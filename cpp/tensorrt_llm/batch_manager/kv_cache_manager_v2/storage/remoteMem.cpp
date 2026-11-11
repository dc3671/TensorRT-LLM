/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kv_cache_manager_v2/storage/remoteMem.h"
#include "kv_cache_manager_v2/exceptions.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/transferAgent.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <optional>
#include <sstream>
#include <string_view>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

namespace kvc = tensorrt_llm::executor::kv_cache;

namespace
{

std::string hostName()
{
    char buf[256] = {};
    if (::gethostname(buf, sizeof(buf) - 1) != 0)
    {
        return "unknown";
    }
    return buf;
}

//! The contents of a lender's file, or nullopt when the file is gone: the lender withdrew after the directory was
//! listed. Any other read failure throws.
std::optional<std::string> readLenderFile(std::filesystem::path const& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        std::error_code ec;
        if (!std::filesystem::exists(path, ec) && !ec)
        {
            return std::nullopt;
        }
        throw RemoteMemError("KVCM2 remote level: cannot read " + path.string());
    }
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

//! Bounds on what a lender's .meta may declare, so a malformed file cannot make a client allocate without limit.
constexpr uint64_t kMinChunkBytes = 1ULL << 20;
constexpr uint64_t kMaxChunksPerLender = 1ULL << 20;

constexpr std::string_view kLenderPrefix = "lender.";
constexpr std::string_view kReadySuffix = ".ready";
constexpr std::string_view kFailedSuffix = ".failed";

std::string cuErrorName(CUresult result)
{
    char const* name = nullptr;
    if (cuGetErrorName(result, &name) != CUDA_SUCCESS || name == nullptr)
    {
        return "CUresult " + std::to_string(static_cast<int>(result));
    }
    return name;
}

} // namespace

// ---------------------------------------------------------------------------
// RemoteMemClient
// ---------------------------------------------------------------------------

RemoteMemClient::RemoteMemClient(RemoteMemOptions options)
    : mOptions(std::move(options))
{
    TLLM_CHECK_WITH_INFO(std::filesystem::is_directory(mOptions.directory),
        "KVCM2 remote level: rendezvous directory '%s' does not exist", mOptions.directory.c_str());
    // Unique among the process's clients, past and present: it names the NIXL agent and the owner of each claim.
    mName = "kvcm2-remote-" + hostName() + "-" + std::to_string(::getpid()) + "-"
        + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-"
        + std::to_string(reinterpret_cast<std::uintptr_t>(this));
    try
    {
        if (mOptions.fabric)
        {
            startFabric();
        }
        else
        {
            kvc::BaseAgentConfig config{};
            config.mName = mName;
            // NIXL's progress thread completes connection set-up with lenders that never call into NIXL themselves.
            config.useProgThread = true;
            config.multiThread = false;
            config.useListenThread = false;
            config.enableTelemetry = false;
            // This agent's own devices, so the process-wide UCX_NET_DEVICES of the KV transceiver's agent stays.
            if (!mOptions.ucxDevices.empty())
            {
                config.backendParams["ucx_devices"] = mOptions.ucxDevices;
            }
            mNixlAgent = kvc::makeTransferAgent("nixl", &config);
            TLLM_CHECK_WITH_INFO(mNixlAgent != nullptr, "KVCM2 remote level: creating the NIXL agent failed");
        }
        std::lock_guard<std::mutex> lock(mMutex);
        refreshLenders();
    }
    catch (...)
    {
        // No destructor runs for a failed constructor, and nothing is claimed yet.
        releaseTransport();
        throw;
    }
    TLLM_LOG_INFO("KVCM2 remote level: client %s, rendezvous %s, %zu lender(s) published, transport %s", mName.c_str(),
        mOptions.directory.c_str(), mLenders.size(), mOptions.fabric ? "fabric" : "nixl");
}

RemoteMemClient::~RemoteMemClient()
{
    // The claims go last: another client may write into a chunk as soon as its claim is gone. A WRITE that timed out
    // may still land in its chunk, so after one the chunks stay claimed.
    bool const releaseClaims = mUncancelled.empty();
    releaseTransport();
    if (!releaseClaims)
    {
        TLLM_LOG_WARNING("KVCM2 remote level: %s keeps its %zu claim(s) in %s after uncancelled transfers",
            mName.c_str(), mClaims.size(), mOptions.directory.c_str());
        return;
    }
    for (auto const& path : mClaims)
    {
        ::unlink(path.c_str());
    }
}

void RemoteMemClient::releaseTransport() noexcept
{
    if (mOptions.fabric)
    {
        // Copies still queued on callers' streams must finish before their mappings go.
        if (CUresult const result = synchronizeContext(); result != CUDA_SUCCESS)
        {
            TLLM_LOG_WARNING("KVCM2 remote level: teardown of %s: synchronizing failed: %s", mName.c_str(),
                cuErrorName(result).c_str());
        }
        for (auto const& mapping : mMappings)
        {
            cuMemUnmap(mapping.address, mapping.bytes);
            cuMemAddressFree(mapping.address, mapping.bytes);
            cuMemRelease(mapping.handle);
        }
        mMappings.clear();
        mMappingEnd.clear();
        if (mContext != nullptr)
        {
            cuDevicePrimaryCtxRelease(mDevice);
            mContext = nullptr;
        }
        return;
    }
    if (mNixlAgent == nullptr)
    {
        return;
    }
    try
    {
        for (auto const& [base, info] : mLocalRegions)
        {
            mNixlAgent->deregisterMemory(
                kvc::MemoryDescs(kvc::MemoryType::kDRAM, {kvc::MemoryDesc(base, info.first, /*deviceId=*/0)}));
        }
        for (auto const& lender : mLenders)
        {
            if (lender.loaded)
            {
                mNixlAgent->invalidateRemoteAgent(lender.name);
            }
        }
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_WARNING("KVCM2 remote level: teardown of %s: %s", mName.c_str(), e.what());
    }
    mUncancelled.clear();
    mNixlAgent.reset();
}

void RemoteMemClient::refreshLenders()
{
    namespace fs = std::filesystem;
    std::error_code ec;
    std::vector<std::string> ready;
    size_t failed = 0;
    for (auto const& entry : fs::directory_iterator(mOptions.directory, ec))
    {
        auto const file = entry.path().filename().string();
        if (file.size() > kLenderPrefix.size() + kReadySuffix.size() && file.starts_with(kLenderPrefix)
            && file.ends_with(kReadySuffix))
        {
            ready.push_back(
                file.substr(kLenderPrefix.size(), file.size() - kLenderPrefix.size() - kReadySuffix.size()));
        }
        else if (file.size() > kLenderPrefix.size() + kFailedSuffix.size() && file.starts_with(kLenderPrefix)
            && file.ends_with(kFailedSuffix))
        {
            ++failed;
        }
    }
    if (ec)
    {
        throw RemoteMemError("KVCM2 remote level: cannot list " + mOptions.directory + ": " + ec.message());
    }
    mReportedLenders = ready.size() + failed;
    std::sort(ready.begin(), ready.end());
    for (auto& lender : mLenders)
    {
        if (!std::binary_search(ready.begin(), ready.end(), lender.name))
        {
            stillPublished(lender);
        }
    }
    for (auto const& name : ready)
    {
        if (mLenderIndex.count(name) != 0)
        {
            continue;
        }
        fs::path const dir(mOptions.directory);
        Lender lender;
        auto const metaPath = dir / ("lender." + name + ".meta");
        // Lender files hold memory registration keys and fabric handles: only this user's lenders are trusted.
        struct stat metaStat = {};
        if (::stat(metaPath.c_str(), &metaStat) != 0)
        {
            continue; // withdrawn since the listing
        }
        if (metaStat.st_uid != ::geteuid())
        {
            TLLM_LOG_WARNING("KVCM2 remote level: ignoring lender %s: owned by uid %u, not this user", name.c_str(),
                static_cast<unsigned>(metaStat.st_uid));
            continue;
        }
        auto const metaText = readLenderFile(metaPath);
        if (!metaText)
        {
            continue; // withdrawn since the listing
        }
        std::istringstream meta(*metaText);
        std::string line;
        int version = 0;
        std::string agentName;
        std::string transport = "nixl";
        while (std::getline(meta, line))
        {
            std::istringstream fields(line);
            std::string key;
            fields >> key;
            if (key == "version")
            {
                fields >> version;
            }
            else if (key == "name")
            {
                fields >> agentName;
            }
            else if (key == "transport")
            {
                fields >> transport;
            }
            else if (key == "host")
            {
                fields >> lender.host;
            }
            else if (key == "chunk_bytes")
            {
                fields >> lender.chunkBytes;
            }
            else if (key == "segment")
            {
                uint64_t base = 0;
                uint64_t bytes = 0;
                fields >> base >> bytes;
                lender.segments.emplace_back(base, bytes);
            }
        }
        if (version != 1 || agentName != name || lender.chunkBytes == 0 || lender.segments.empty())
        {
            TLLM_LOG_WARNING("KVCM2 remote level: ignoring lender %s: malformed meta (version %d, name '%s')",
                name.c_str(), version, agentName.c_str());
            continue;
        }
        if ((transport == "fabric") != mOptions.fabric)
        {
            TLLM_LOG_WARNING("KVCM2 remote level: ignoring lender %s: it lends over %s, this client uses %s",
                name.c_str(), transport.c_str(), mOptions.fabric ? "fabric" : "nixl");
            continue;
        }
        uint64_t numChunks = 0;
        bool segmentsValid = lender.chunkBytes >= kMinChunkBytes;
        for (auto const& [base, bytes] : lender.segments)
        {
            segmentsValid = segmentsValid && base <= UINT64_MAX - bytes;
            numChunks += segmentsValid ? bytes / lender.chunkBytes : 0;
        }
        if (!segmentsValid || numChunks > kMaxChunksPerLender)
        {
            TLLM_LOG_WARNING("KVCM2 remote level: ignoring lender %s: chunk size %llu B or its segments out of bounds",
                name.c_str(), static_cast<unsigned long long>(lender.chunkBytes));
            continue;
        }
        lender.name = name;
        for (auto const& [base, bytes] : lender.segments)
        {
            for (uint64_t chunk = 0; chunk < bytes / lender.chunkBytes; ++chunk)
            {
                lender.chunks.emplace_back(base + chunk * lender.chunkBytes, lender.chunkBytes);
            }
        }
        auto payload = readLenderFile(dir / ("lender." + name + (mOptions.fabric ? ".fabric" : ".desc")));
        if (!payload)
        {
            continue; // withdrawn since the listing
        }
        if (mOptions.fabric)
        {
            lender.fabricHandles = std::move(*payload);
            if (lender.fabricHandles.size() != lender.chunks.size() * sizeof(CUmemFabricHandle))
            {
                TLLM_LOG_WARNING(
                    "KVCM2 remote level: ignoring lender %s: %zu B of fabric handles for %zu chunk(s), "
                    "expected %zu B each",
                    name.c_str(), lender.fabricHandles.size(), lender.chunks.size(), sizeof(CUmemFabricHandle));
                continue;
            }
        }
        else
        {
            lender.desc = std::move(*payload);
        }
        TLLM_LOG_INFO("KVCM2 remote level: found lender %s on %s, %zu chunk(s) of %.2f GiB", name.c_str(),
            lender.host.c_str(), lender.chunks.size(),
            static_cast<double>(lender.chunkBytes) / static_cast<double>(1ULL << 30));
        mLenderIndex.emplace(name, mLenders.size());
        mLenders.push_back(std::move(lender));
    }
}

bool RemoteMemClient::stillPublished(Lender& lender)
{
    if (!lender.withdrawn)
    {
        std::error_code ec;
        auto const ready = std::filesystem::path(mOptions.directory) / ("lender." + lender.name + ".ready");
        if (!std::filesystem::exists(ready, ec) && !ec)
        {
            // Lender names are unique per lender process, so a withdrawn one never comes back.
            lender.withdrawn = true;
            TLLM_LOG_WARNING("KVCM2 remote level: lender %s (%s) withdrew; no new claims from it", lender.name.c_str(),
                lender.host.c_str());
        }
    }
    return !lender.withdrawn;
}

void RemoteMemClient::throwIfFailed() const
{
    if (mFailed.load(std::memory_order_relaxed))
    {
        throw RemoteMemError(
            "KVCM2 remote level: " + mOptions.directory + " takes no transfers after an earlier failure: " + mFailure);
    }
}

void RemoteMemClient::fail(std::string const& message)
{
    if (!mFailed.load(std::memory_order_relaxed))
    {
        mFailure = message;
        mFailed.store(true, std::memory_order_release);
        TLLM_LOG_ERROR(
            "%s; the remote level in %s takes no further transfers", message.c_str(), mOptions.directory.c_str());
    }
    throw RemoteMemError(message);
}

void RemoteMemClient::ensureLoaded(Lender& lender)
{
    if (lender.loaded)
    {
        return;
    }
    if (mOptions.fabric)
    {
        lender.loaded = true; // fabric chunks are imported one by one as they are claimed (mapChunk)
        return;
    }
    try
    {
        mNixlAgent->loadRemoteAgent(lender.name, kvc::AgentDesc::deserialize(lender.desc));
    }
    catch (std::exception const& e)
    {
        throw RemoteMemError(
            "KVCM2 remote level: loading lender " + lender.name + " (" + lender.host + ") failed: " + e.what());
    }
    lender.loaded = true;
}

bool RemoteMemClient::tryClaim(size_t minBytes, RemoteChunk& out)
{
    if (mLenders.empty())
    {
        return false;
    }
    std::string const owner = mName + " " + hostName() + " " + std::to_string(::getpid()) + "\n";
    // Clients start at different lenders, so the first claims of many clients spread out.
    size_t const start = std::hash<std::string>{}(mName) % mLenders.size();
    for (size_t k = 0; k < mLenders.size(); ++k)
    {
        uint32_t const index = static_cast<uint32_t>((start + k) % mLenders.size());
        auto& lender = mLenders[index];
        if (lender.chunkBytes < minBytes || lender.full || !stillPublished(lender))
        {
            continue;
        }
        // Chunks are probed from the last one this client claimed or found taken, wrapping around once: claims
        // last as long as their client, so earlier chunks rarely free up, and the shared file system sees one
        // exclusive create per chunk per client instead of one per claim.
        size_t const numChunks = lender.chunks.size();
        for (size_t probe = 0; probe < numChunks; ++probe)
        {
            size_t const chunk = (lender.nextClaim + probe) % numChunks;
            auto const path
                = (std::filesystem::path(mOptions.directory) / ("claim." + lender.name + "." + std::to_string(chunk)))
                      .string();
            int const fd = ::open(path.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0600);
            if (fd < 0)
            {
                if (errno == EEXIST)
                {
                    continue;
                }
                throw RemoteMemError(
                    "KVCM2 remote level: creating claim " + path + " failed: " + std::string(std::strerror(errno)));
            }
            ssize_t const written = ::write(fd, owner.data(), owner.size());
            ::close(fd);
            if (written != static_cast<ssize_t>(owner.size()))
            {
                ::unlink(path.c_str());
                throw RemoteMemError("KVCM2 remote level: writing claim " + path + " failed");
            }
            uint64_t address = lender.chunks[chunk].first;
            try
            {
                ensureLoaded(lender);
                if (mOptions.fabric)
                {
                    // The chunk (its own fabric allocation) becomes a device address of this client.
                    address = mapChunk(lender, chunk, lender.chunks[chunk].second);
                }
            }
            catch (RemoteMemError const& e)
            {
                // E.g. a crashed lender whose .ready file stayed: no claims from it, the next lender is tried.
                ::unlink(path.c_str());
                lender.withdrawn = true;
                mLastClaimError = e.what();
                TLLM_LOG_WARNING("KVCM2 remote level: no claims from lender %s (%s): %s", lender.name.c_str(),
                    lender.host.c_str(), e.what());
                break;
            }
            catch (...)
            {
                ::unlink(path.c_str());
                throw;
            }
            out = RemoteChunk{index, address, lender.chunks[chunk].second};
            lender.nextClaim = chunk + 1;
            mClaims.push_back(path);
            mClaimedBytes += out.bytes;
            return true;
        }
        lender.full = !lender.withdrawn;
    }
    // No lender had a chunk: the next attempt probes the ones found full again, in case a client let go of one.
    for (auto& lender : mLenders)
    {
        lender.full = false;
    }
    return false;
}

RemoteChunk RemoteMemClient::acquireChunk(size_t minBytes)
{
    std::unique_lock<std::mutex> lock(mMutex);
    throwIfFailed();
    for (auto it = mFreeChunks.begin(); it != mFreeChunks.end(); ++it)
    {
        if (it->bytes >= minBytes && !mLenders[it->lender].withdrawn)
        {
            RemoteChunk chunk = std::move(*it);
            mFreeChunks.erase(it);
            return chunk;
        }
    }
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(mOptions.waitTimeoutMs);
    bool warned = false;
    while (true)
    {
        RemoteChunk chunk;
        if (tryClaim(minBytes, chunk))
        {
            return chunk;
        }
        // Once every lender the job starts has reported, no chunk can appear: claims last as long as their client.
        bool const allReported = mOptions.expectedLenders > 0 && mReportedLenders >= mOptions.expectedLenders;
        if (allReported || std::chrono::steady_clock::now() >= deadline)
        {
            throw RemoteMemError("KVCM2 remote level: no lender in " + mOptions.directory + " has a free chunk of "
                + std::to_string(minBytes) + " B or more "
                + (allReported ? "and all " + std::to_string(mOptions.expectedLenders) + " expected lenders reported"
                               : "after " + std::to_string(mOptions.waitTimeoutMs) + " ms")
                + "; this process holds " + std::to_string(mClaimedBytes) + " B of " + std::to_string(mLenders.size())
                + " lender(s)" + (mLastClaimError.empty() ? "" : "; last lender withdrawn: " + mLastClaimError));
        }
        if (!warned)
        {
            TLLM_LOG_INFO("KVCM2 remote level: waiting for lender capacity in %s (holding %.2f GiB, %zu lender(s))",
                mOptions.directory.c_str(), static_cast<double>(mClaimedBytes) / static_cast<double>(1ULL << 30),
                mLenders.size());
            warned = true;
        }
        lock.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        lock.lock();
        refreshLenders();
    }
}

void RemoteMemClient::releaseChunk(RemoteChunk chunk) noexcept
{
    std::lock_guard<std::mutex> lock(mMutex);
    mFreeChunks.push_back(std::move(chunk));
}

void RemoteMemClient::registerLocal(MemAddress base, size_t bytes)
{
    if (mOptions.fabric)
    {
        return; // copy engines reach pinned host memory through unified addressing; nothing to register
    }
    std::lock_guard<std::mutex> lock(mMutex);
    auto it = mLocalRegions.find(base);
    if (it != mLocalRegions.end())
    {
        TLLM_CHECK_WITH_INFO(it->second.first == bytes,
            "KVCM2 remote level: local region re-registered with %zu B, was %zu B", bytes, it->second.first);
        ++it->second.second;
        return;
    }
    mNixlAgent->registerMemory(
        kvc::MemoryDescs(kvc::MemoryType::kDRAM, {kvc::MemoryDesc(base, bytes, /*deviceId=*/0)}));
    mLocalRegions.emplace(base, std::make_pair(bytes, 1));
}

void RemoteMemClient::deregisterLocal(MemAddress base, size_t bytes) noexcept
{
    if (mOptions.fabric)
    {
        return;
    }
    std::lock_guard<std::mutex> lock(mMutex);
    auto it = mLocalRegions.find(base);
    if (it == mLocalRegions.end() || --it->second.second > 0)
    {
        return;
    }
    try
    {
        mNixlAgent->deregisterMemory(
            kvc::MemoryDescs(kvc::MemoryType::kDRAM, {kvc::MemoryDesc(base, bytes, /*deviceId=*/0)}));
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_WARNING("KVCM2 remote level: deregistering local memory failed: %s", e.what());
    }
    mLocalRegions.erase(it);
}

void RemoteMemClient::transfer(bool write, std::vector<RemoteIo> const& ios, size_t bytes)
{
    if (ios.empty() || bytes == 0)
    {
        return;
    }
    // A manager's remote levels share one transport (KVCacheManagerConfig::validate), so fabric copies always go
    // on the caller's stream (transferOnStream).
    TLLM_CHECK_WITH_INFO(!mOptions.fabric, "KVCM2 remote level: synchronous transfer on a fabric client");
    std::lock_guard<std::mutex> lock(mMutex);
    throwIfFailed();
    std::vector<std::vector<size_t>> byLender(mLenders.size());
    for (size_t i = 0; i < ios.size(); ++i)
    {
        TLLM_CHECK_DEBUG(ios[i].lender < mLenders.size());
        byLender[ios[i].lender].push_back(i);
    }
    auto const opName = write ? std::string("WRITE to") : std::string("READ from");
    std::vector<std::pair<uint32_t, std::unique_ptr<kvc::TransferStatus>>> pending;
    std::string error; // the first failure; the requests already submitted settle before it is thrown
    for (uint32_t lenderIdx = 0; lenderIdx < byLender.size(); ++lenderIdx)
    {
        auto const& entries = byLender[lenderIdx];
        if (entries.empty())
        {
            continue;
        }
        std::vector<kvc::MemoryDesc> local;
        std::vector<kvc::MemoryDesc> remote;
        local.reserve(entries.size());
        remote.reserve(entries.size());
        for (size_t i : entries)
        {
            local.emplace_back(ios[i].local, bytes, /*deviceId=*/0);
            remote.emplace_back(static_cast<uintptr_t>(ios[i].remoteAddr), bytes, /*deviceId=*/0);
        }
        // NIXL's request names the local descriptors first and the remote ones second, whichever way data flows.
        kvc::TransferRequest request(write ? kvc::TransferOp::kWRITE : kvc::TransferOp::kREAD,
            kvc::MemoryDescs(kvc::MemoryType::kDRAM, std::move(local)),
            kvc::MemoryDescs(kvc::MemoryType::kDRAM, std::move(remote)), mLenders[lenderIdx].name);
        try
        {
            pending.emplace_back(lenderIdx, mNixlAgent->submitTransferRequests(request));
        }
        catch (std::exception const& e)
        {
            error = opName + " lender " + mLenders[lenderIdx].name + " (" + mLenders[lenderIdx].host
                + "): submitting failed: " + e.what();
            break;
        }
    }
    // Wait for every request, cancelling the ones still in flight at the deadline, before returning or throwing:
    // the caller reuses the local buffers, which a late request could still write.
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(mOptions.transferTimeoutMs);
    for (auto& [lenderIdx, status] : pending)
    {
        auto const remaining = std::max<int64_t>(0,
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count());
        auto const state = status->wait(remaining);
        if (state == kvc::TransferState::kSUCCESS)
        {
            continue;
        }
        if (state == kvc::TransferState::kIN_PROGRESS)
        {
            // release() returning true does not prove the remote side is quiescent (UCX reports success regardless),
            // so a timed-out request counts as uncancelled: this client keeps its claims until it is destroyed.
            if (!status->release())
            {
                TLLM_LOG_ERROR("KVCM2 remote level: a timed-out %s lender %s (%s) could not be cancelled",
                    opName.c_str(), mLenders[lenderIdx].name.c_str(), mLenders[lenderIdx].host.c_str());
            }
            mUncancelled.push_back(std::move(status));
        }
        if (error.empty())
        {
            std::string const why = state == kvc::TransferState::kIN_PROGRESS
                ? "timed out after " + std::to_string(mOptions.transferTimeoutMs) + " ms"
                : "failed (" + status->getLastStatusStr() + ")";
            error = opName + " lender " + mLenders[lenderIdx].name + " (" + mLenders[lenderIdx].host + ") of "
                + std::to_string(byLender[lenderIdx].size()) + " page(s) x " + std::to_string(bytes) + " B " + why;
        }
    }
    if (!error.empty())
    {
        fail("KVCM2 remote level: " + error);
    }
}

void RemoteMemClient::transferOnStream(bool write, std::vector<RemoteIo> const& ios, size_t bytes, CUstream stream)
{
    TLLM_CHECK_WITH_INFO(mOptions.fabric, "KVCM2 remote level: stream-ordered transfer on a NIXL client");
    if (ios.empty() || bytes == 0)
    {
        return;
    }
    std::lock_guard<std::mutex> lock(mMutex);
    throwIfFailed();
    fabricCopy(write, ios, bytes, stream);
}

void RemoteMemClient::drain() noexcept
{
    if (!mOptions.fabric)
    {
        return;
    }
    // Copies on callers' streams reach memory that is about to be released or reused.
    if (CUresult const result = synchronizeContext(); result != CUDA_SUCCESS)
    {
        TLLM_LOG_ERROR("KVCM2 remote level: draining the fabric copies of %s failed: %s", mOptions.directory.c_str(),
            cuErrorName(result).c_str());
    }
}

CUresult RemoteMemClient::synchronizeContext() const noexcept
{
    if (mContext == nullptr)
    {
        return CUDA_SUCCESS;
    }
    CUresult result = cuCtxPushCurrent(mContext);
    if (result != CUDA_SUCCESS)
    {
        return result;
    }
    result = cuCtxSynchronize();
    CUcontext popped = nullptr;
    cuCtxPopCurrent(&popped);
    return result;
}

// ---------------------------------------------------------------------------
// Fabric transport
// ---------------------------------------------------------------------------

void RemoteMemClient::startFabric()
{
    CUcontext context = nullptr;
    cuCheck(cuCtxGetCurrent(&context));
    if (context == nullptr)
    {
        throw RemoteMemError(
            "KVCM2 remote level: the fabric transport needs a current CUDA context on the calling "
            "thread");
    }
    cuCheck(cuCtxGetDevice(&mDevice));
    // Callers enqueue copies in the device's primary context (as PyTorch does); teardown synchronizes it.
    CUcontext primary = nullptr;
    cuCheck(cuDevicePrimaryCtxRetain(&primary, mDevice));
    if (primary != context)
    {
        cuDevicePrimaryCtxRelease(mDevice);
        throw RemoteMemError("KVCM2 remote level: the fabric transport needs the device's primary context current");
    }
    mContext = primary; // released by releaseTransport()
}

CUdeviceptr RemoteMemClient::mapChunk(Lender const& lender, size_t chunk, uint64_t bytes)
{
    CUmemFabricHandle fabricHandle{};
    std::memcpy(&fabricHandle, lender.fabricHandles.data() + chunk * sizeof(CUmemFabricHandle), sizeof(fabricHandle));
    CUmemGenericAllocationHandle handle = 0;
    CUresult result = cuMemImportFromShareableHandle(&handle, &fabricHandle, CU_MEM_HANDLE_TYPE_FABRIC);
    if (result != CUDA_SUCCESS)
    {
        throw RemoteMemError("KVCM2 remote level: importing chunk " + std::to_string(chunk) + " of lender "
            + lender.name + " (" + lender.host + ") failed: " + cuErrorName(result)
            + "; fabric memory needs both trays in one NVLink domain (NVL72 rack), the job's IMEX channel and a live "
              "lender");
    }
    CUdeviceptr address = 0;
    result = cuMemAddressReserve(&address, bytes, 0, 0, 0);
    if (result == CUDA_SUCCESS)
    {
        // An imported fabric allocation maps only whole: offset 0, its full size.
        result = cuMemMap(address, bytes, 0, handle, 0);
        if (result == CUDA_SUCCESS)
        {
            CUmemAccessDesc access{};
            access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access.location.id = static_cast<int>(mDevice);
            access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            result = cuMemSetAccess(address, bytes, &access, 1);
            if (result != CUDA_SUCCESS)
            {
                cuMemUnmap(address, bytes);
            }
        }
        if (result != CUDA_SUCCESS)
        {
            cuMemAddressFree(address, bytes);
        }
    }
    if (result != CUDA_SUCCESS)
    {
        cuMemRelease(handle);
        throw RemoteMemError("KVCM2 remote level: mapping chunk " + std::to_string(chunk) + " (" + std::to_string(bytes)
            + " B) of lender " + lender.name + " (" + lender.host + ") failed: " + cuErrorName(result));
    }
    mMappings.push_back(FabricMapping{address, bytes, handle});
    mMappingEnd.emplace(address, address + bytes);
    return address;
}

void RemoteMemClient::fabricCopy(bool write, std::vector<RemoteIo> const& ios, size_t bytes, CUstream stream)
{
    // Runs of pages that are contiguous on both sides go as one copy, within one mapped chunk: two chunks' separate
    // mappings may happen to be adjacent in the address space.
    size_t i = 0;
    while (i < ios.size())
    {
        auto const mapping = mMappingEnd.upper_bound(static_cast<CUdeviceptr>(ios[i].remoteAddr));
        TLLM_CHECK_WITH_INFO(mapping != mMappingEnd.begin(), "KVCM2 remote level: fabric address outside every chunk");
        CUdeviceptr const mappingEnd = std::prev(mapping)->second;
        size_t j = i + 1;
        while (j < ios.size() && ios[j].local == ios[j - 1].local + bytes
            && ios[j].remoteAddr == ios[j - 1].remoteAddr + bytes && ios[j].remoteAddr + bytes <= mappingEnd)
        {
            ++j;
        }
        auto const local = static_cast<CUdeviceptr>(ios[i].local);
        auto const remote = static_cast<CUdeviceptr>(ios[i].remoteAddr);
        size_t const n = (j - i) * bytes;
        CUresult const result
            = write ? cuMemcpyAsync(remote, local, n, stream) : cuMemcpyAsync(local, remote, n, stream);
        if (result != CUDA_SUCCESS)
        {
            fail(std::string("KVCM2 remote level: enqueuing a fabric ") + (write ? "WRITE" : "READ") + " of "
                + std::to_string(n) + " B failed: " + cuErrorName(result));
        }
        i = j;
    }
}

// ---------------------------------------------------------------------------
// RemoteRange
// ---------------------------------------------------------------------------

RemoteRange::RemoteRange(SharedPtr<RemoteMemClient> client, size_t slotSize)
    : mClient(std::move(client))
    , mSlotSize(slotSize)
{
    TLLM_CHECK(mClient != nullptr);
    TLLM_CHECK_WITH_INFO(mSlotSize > 0, "KVCM2 remote level: zero slot size");
}

RemoteRange::~RemoteRange()
{
    for (auto& chunk : mChunks)
    {
        mClient->releaseChunk(std::move(chunk));
    }
}

void RemoteRange::reserve(SlotCount numSlots)
{
    while (mCapacity < numSlots)
    {
        RemoteChunk chunk = mClient->acquireChunk(mSlotSize);
        SlotCount const slots = static_cast<SlotCount>(chunk.bytes / mSlotSize);
        TLLM_CHECK(slots > 0);
        mFirstSlot.push_back(mCapacity);
        mCapacity += slots;
        mChunks.push_back(std::move(chunk));
    }
}

std::pair<uint32_t, uint64_t> RemoteRange::locate(SlotId slot) const
{
    SlotCount const s = slot.value();
    TLLM_CHECK_WITH_INFO(s >= 0 && s < mCapacity, "KVCM2 remote level: slot %lld outside the %lld reserved",
        static_cast<long long>(s), static_cast<long long>(mCapacity));
    auto const it = std::upper_bound(mFirstSlot.begin(), mFirstSlot.end(), s);
    size_t const index = static_cast<size_t>(std::distance(mFirstSlot.begin(), it)) - 1;
    auto const& chunk = mChunks[index];
    return {chunk.lender, chunk.remoteAddr + static_cast<uint64_t>(s - mFirstSlot[index]) * mSlotSize};
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
