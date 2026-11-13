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

#include <algorithm>
#include <atomic>
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
#include <stdexcept>
#include <string_view>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

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

//! The contents of a donor's file, or nullopt when it cannot be read: the donor withdrew after the directory was
//! listed, or the file is unreadable, which only passes over that donor until the next refresh.
std::optional<std::string> readDonorFile(std::filesystem::path const& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        std::error_code ec;
        if (std::filesystem::exists(path, ec) || ec)
        {
            TLLM_LOG_WARNING("KVCM2 remote level: cannot read %s; passing over its donor", path.c_str());
        }
        return std::nullopt;
    }
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

//! The transport a donor's .failed file names, or nullopt when it names none.
std::optional<std::string> failedDonorTransport(std::string const& text)
{
    std::istringstream lines(text);
    std::string line;
    while (std::getline(lines, line))
    {
        std::istringstream fields(line);
        std::string key;
        std::string value;
        fields >> key >> value;
        if (key == "transport")
        {
            return value;
        }
    }
    return std::nullopt;
}

//! Bounds on what a donor's .meta may declare, so a malformed file cannot make a client allocate without limit.
constexpr uint64_t kMinChunkBytes = 1ULL << 20;
constexpr uint64_t kMaxChunksPerDonor = 1ULL << 20;

constexpr std::string_view kDonorPrefix = "donor.";
constexpr std::string_view kReadySuffix = ".ready";
constexpr std::string_view kFailedSuffix = ".failed";

} // namespace

// ---------------------------------------------------------------------------
// RemoteMemClient
// ---------------------------------------------------------------------------

RemoteMemClient::RemoteMemClient(std::string name, RemoteMemOptions options, RemoteTransferConfig transferConfig,
    std::unique_ptr<RemoteTransferBackend> backend)
    : mName(std::move(name))
    , mOptions(std::move(options))
    , mTransferConfig(std::move(transferConfig))
    , mBackend(std::move(backend))
{
    TLLM_CHECK(mBackend != nullptr);
    mStreamOrdered = mBackend->streamOrdered();
    {
        std::lock_guard<std::mutex> lock(mMutex);
        refreshDonors();
    }
    TLLM_LOG_INFO("KVCM2 remote level: client %s, rendezvous %s, %zu donor(s) published, transport %s", mName.c_str(),
        mOptions.directory.c_str(), mDonors.size(), mBackend->kind());
}

RemoteMemClient::~RemoteMemClient()
{
    if (mAbandoned)
    {
        // A transfer that never quiesced may still reach the chunks and the backend: both outlive this client.
        static_cast<void>(mBackend.release());
        TLLM_LOG_WARNING(
            "KVCM2 remote level: %s keeps its %zu claim(s) in %s and its transport after transfers "
            "that never quiesced",
            mName.c_str(), mClaims.size(), mOptions.directory.c_str());
        return;
    }
    // The claims go last, after the backend's teardown: another client may write into a chunk as soon as its claim
    // is gone.
    mBackend.reset();
    for (auto const& path : mClaims)
    {
        ::unlink(path.c_str());
    }
}

void RemoteMemClient::refreshDonors()
{
    namespace fs = std::filesystem;
    std::error_code ec;
    std::vector<std::string> ready;
    size_t failed = 0;
    for (auto const& entry : fs::directory_iterator(mOptions.directory, ec))
    {
        auto const file = entry.path().filename().string();
        if (file.size() > kDonorPrefix.size() + kReadySuffix.size() && file.starts_with(kDonorPrefix)
            && file.ends_with(kReadySuffix))
        {
            ready.push_back(file.substr(kDonorPrefix.size(), file.size() - kDonorPrefix.size() - kReadySuffix.size()));
        }
        else if (file.size() > kDonorPrefix.size() + kFailedSuffix.size() && file.starts_with(kDonorPrefix)
            && file.ends_with(kFailedSuffix))
        {
            // A failed donor of another transport is not one of the donors this client counts on.
            auto const text = readDonorFile(entry.path());
            auto const transport = text ? failedDonorTransport(*text) : std::nullopt;
            failed += !transport || *transport == mBackend->kind() ? 1 : 0;
        }
    }
    if (ec)
    {
        throw RemoteMemError("KVCM2 remote level: cannot list " + mOptions.directory + ": " + ec.message());
    }
    std::sort(ready.begin(), ready.end());
    for (auto& donor : mDonors)
    {
        if (!std::binary_search(ready.begin(), ready.end(), donor.name))
        {
            stillPublished(donor);
        }
    }
    for (auto const& name : ready)
    {
        if (mDonorIndex.count(name) != 0 || mForeignDonors.count(name) != 0)
        {
            continue;
        }
        fs::path const dir(mOptions.directory);
        RemoteDonor donor;
        auto const metaPath = dir / ("donor." + name + ".meta");
        // Donor files hold memory registration keys and fabric handles: only this user's donors are trusted.
        struct stat metaStat = {};
        if (::stat(metaPath.c_str(), &metaStat) != 0)
        {
            continue; // withdrawn since the listing
        }
        if (metaStat.st_uid != ::geteuid())
        {
            TLLM_LOG_WARNING("KVCM2 remote level: ignoring donor %s: owned by uid %u, not this user", name.c_str(),
                static_cast<unsigned>(metaStat.st_uid));
            continue;
        }
        auto const metaText = readDonorFile(metaPath);
        if (!metaText)
        {
            continue; // withdrawn since the listing
        }
        std::istringstream meta(*metaText);
        std::string line;
        int version = 0;
        std::string agentName;
        std::string transport;
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
                fields >> donor.host;
            }
            else if (key == "chunk_bytes")
            {
                fields >> donor.chunkBytes;
            }
            else if (key == "segment")
            {
                uint64_t base = 0;
                uint64_t bytes = 0;
                fields >> base >> bytes;
                donor.segments.emplace_back(base, bytes);
            }
        }
        // Donors publish one segment (remote_donor.py): more could overlap or sum past the chunk bound.
        if (version != 1 || agentName != name || transport.empty() || donor.chunkBytes == 0
            || donor.segments.size() != 1)
        {
            TLLM_LOG_WARNING(
                "KVCM2 remote level: ignoring donor %s: malformed meta (version %d, name '%s', transport '%s')",
                name.c_str(), version, agentName.c_str(), transport.c_str());
            continue;
        }
        if (transport != mBackend->kind())
        {
            TLLM_LOG_WARNING(
                "KVCM2 remote level: ignoring donor %s: it donates over %s, this client uses %s; it is not counted as "
                "one of the expected donors",
                name.c_str(), transport.c_str(), mBackend->kind());
            mForeignDonors.insert(name);
            continue;
        }
        auto const [segmentBase, segmentBytes] = donor.segments.front();
        bool const segmentsValid = donor.chunkBytes >= kMinChunkBytes && segmentBase <= UINT64_MAX - segmentBytes;
        uint64_t const numChunks = segmentsValid ? segmentBytes / donor.chunkBytes : 0;
        if (!segmentsValid || numChunks > kMaxChunksPerDonor)
        {
            TLLM_LOG_WARNING("KVCM2 remote level: ignoring donor %s: chunk size %llu B or its segments out of bounds",
                name.c_str(), static_cast<unsigned long long>(donor.chunkBytes));
            continue;
        }
        donor.name = name;
        for (auto const& [base, bytes] : donor.segments)
        {
            for (uint64_t chunk = 0; chunk < bytes / donor.chunkBytes; ++chunk)
            {
                donor.chunks.emplace_back(base + chunk * donor.chunkBytes, donor.chunkBytes);
            }
        }
        auto payload = readDonorFile(dir / ("donor." + name + "." + mBackend->payloadSuffix()));
        if (!payload)
        {
            continue; // withdrawn since the listing
        }
        donor.payload = std::move(*payload);
        if (!mBackend->acceptPayload(donor))
        {
            continue;
        }
        TLLM_LOG_INFO("KVCM2 remote level: found donor %s on %s, %zu chunk(s) of %.2f GiB", name.c_str(),
            donor.host.c_str(), donor.chunks.size(),
            static_cast<double>(donor.chunkBytes) / static_cast<double>(1ULL << 30));
        mDonorIndex.emplace(name, mDonors.size());
        mDonors.push_back(std::move(donor));
    }
    // Counted once the new ready donors are parsed: a donor of another transport is not reported to this client.
    mReportedDonors = failed
        + static_cast<size_t>(std::count_if(
            ready.begin(), ready.end(), [this](std::string const& name) { return mForeignDonors.count(name) == 0; }));
}

bool RemoteMemClient::stillPublished(RemoteDonor& donor)
{
    if (!donor.withdrawn)
    {
        std::error_code ec;
        auto const ready = std::filesystem::path(mOptions.directory) / ("donor." + donor.name + ".ready");
        if (!std::filesystem::exists(ready, ec) && !ec)
        {
            // Donor names are unique per donor process, so a withdrawn one never comes back.
            donor.withdrawn = true;
            TLLM_LOG_WARNING("KVCM2 remote level: donor %s (%s) withdrew; no new claims from it", donor.name.c_str(),
                donor.host.c_str());
        }
    }
    return !donor.withdrawn;
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
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if (!mFailed.load(std::memory_order_relaxed))
        {
            mFailure = message;
            mFailed.store(true, std::memory_order_release);
            TLLM_LOG_ERROR(
                "%s; the remote level in %s takes no further transfers", message.c_str(), mOptions.directory.c_str());
        }
    }
    throw RemoteMemError(message);
}

bool RemoteMemClient::tryClaim(size_t minBytes, RemoteChunk& out)
{
    if (mDonors.empty())
    {
        return false;
    }
    std::string const owner = mName + " " + hostName() + " " + std::to_string(::getpid()) + "\n";
    // Clients start at different donors, so the first claims of many clients spread out.
    size_t const start = std::hash<std::string>{}(mName) % mDonors.size();
    for (size_t k = 0; k < mDonors.size(); ++k)
    {
        uint32_t const index = static_cast<uint32_t>((start + k) % mDonors.size());
        auto& donor = mDonors[index];
        if (donor.chunkBytes < minBytes || donor.full || !stillPublished(donor))
        {
            continue;
        }
        // Chunks are probed from the last one this client claimed or found taken, wrapping around once: claims
        // last as long as their client, so earlier chunks rarely free up, and the shared file system sees one
        // exclusive create per chunk per client instead of one per claim.
        size_t const numChunks = donor.chunks.size();
        for (size_t probe = 0; probe < numChunks; ++probe)
        {
            size_t const chunk = (donor.nextClaim + probe) % numChunks;
            auto const path
                = (std::filesystem::path(mOptions.directory) / ("claim." + donor.name + "." + std::to_string(chunk)))
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
            uint64_t address = 0;
            try
            {
                address = mBackend->attachChunk(donor, chunk);
            }
            catch (RemoteMemError const& e)
            {
                // E.g. a crashed donor whose .ready file stayed: no claims from it, the next donor is tried.
                ::unlink(path.c_str());
                donor.withdrawn = true;
                mLastClaimError = e.what();
                TLLM_LOG_WARNING("KVCM2 remote level: no claims from donor %s (%s): %s", donor.name.c_str(),
                    donor.host.c_str(), e.what());
                break;
            }
            catch (...)
            {
                ::unlink(path.c_str());
                throw;
            }
            out = RemoteChunk{index, address, donor.chunks[chunk].second};
            donor.nextClaim = chunk + 1;
            mClaims.push_back(path);
            mClaimedBytes += out.bytes;
            return true;
        }
        donor.full = !donor.withdrawn;
    }
    // No donor had a chunk: the next attempt probes the ones found full again, in case a client let go of one.
    for (auto& donor : mDonors)
    {
        donor.full = false;
    }
    return false;
}

RemoteChunk RemoteMemClient::acquireChunk(size_t minBytes)
{
    std::unique_lock<std::mutex> lock(mMutex);
    throwIfFailed();
    for (auto it = mFreeChunks.begin(); it != mFreeChunks.end(); ++it)
    {
        if (it->bytes >= minBytes && !mDonors[it->donor].withdrawn)
        {
            RemoteChunk chunk = std::move(*it);
            mFreeChunks.erase(it);
            return chunk;
        }
    }
    auto const deadline = remoteDeadlineAfter(mOptions.waitTimeoutMs);
    bool warned = false;
    while (true)
    {
        RemoteChunk chunk;
        if (tryClaim(minBytes, chunk))
        {
            return chunk;
        }
        // Once every donor the job starts has reported, no chunk can appear: claims last as long as their client.
        bool const allReported = mOptions.expectedDonors > 0 && mReportedDonors >= mOptions.expectedDonors;
        if (allReported || std::chrono::steady_clock::now() >= deadline)
        {
            throw RemoteMemError("KVCM2 remote level: no donor in " + mOptions.directory + " has a free chunk of "
                + std::to_string(minBytes) + " B or more "
                + (allReported ? "and all " + std::to_string(mOptions.expectedDonors)
                            + " expected donors reported (each layer group of a rank claims whole chunks, so the "
                              "donation "
                              "must exceed the summed cache_size by up to one chunk per layer group per rank)"
                               : "after " + std::to_string(mOptions.waitTimeoutMs) + " ms")
                + "; this process holds " + std::to_string(mClaimedBytes) + " B of " + std::to_string(mDonors.size())
                + " donor(s)" + (mLastClaimError.empty() ? "" : "; last donor withdrawn: " + mLastClaimError));
        }
        if (!warned)
        {
            TLLM_LOG_INFO("KVCM2 remote level: waiting for donor capacity in %s (holding %.2f GiB, %zu donor(s))",
                mOptions.directory.c_str(), static_cast<double>(mClaimedBytes) / static_cast<double>(1ULL << 30),
                mDonors.size());
            warned = true;
        }
        lock.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        lock.lock();
        refreshDonors();
    }
}

void RemoteMemClient::checkUsable() const
{
    std::lock_guard<std::mutex> lock(mMutex);
    throwIfFailed();
}

void RemoteMemClient::releaseChunk(RemoteChunk chunk) noexcept
{
    std::lock_guard<std::mutex> lock(mMutex);
    mFreeChunks.push_back(std::move(chunk));
}

std::unique_ptr<RemoteTransferAttempt> RemoteMemClient::submit(
    bool write, std::vector<RemoteIo> const& ios, size_t bytes, CUstream stream)
{
    std::lock_guard<std::mutex> lock(mMutex);
    throwIfFailed();
    return mBackend->submit(write, mDonors, ios, bytes, stream);
}

void RemoteMemClient::registerLocal(MemAddress base, size_t bytes)
{
    std::lock_guard<std::mutex> lock(mMutex);
    mBackend->registerLocal(base, bytes);
}

void RemoteMemClient::deregisterLocal(MemAddress base, size_t bytes) noexcept
{
    std::lock_guard<std::mutex> lock(mMutex);
    if (!mAbandoned)
    {
        mBackend->deregisterLocal(base, bytes);
    }
}

void RemoteMemClient::drain() noexcept
{
    std::lock_guard<std::mutex> lock(mMutex);
    mBackend->drain();
}

void RemoteMemClient::abandon() noexcept
{
    std::lock_guard<std::mutex> lock(mMutex);
    mAbandoned = true;
}

SharedPtr<RemoteMemClient> makeRemoteMemClient(RemoteMemOptions options, RemoteTransferConfig const& transferConfig)
{
    TLLM_CHECK_WITH_INFO(std::filesystem::is_directory(options.directory),
        "KVCM2 remote level: rendezvous directory '%s' does not exist", options.directory.c_str());
    // Unique among the process's clients, past and present: it names the transfer agent and the owner of each claim.
    static std::atomic<uint64_t> sClients{0};
    std::string name = "kvcm2-remote-" + hostName() + "-" + std::to_string(::getpid()) + "-"
        + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-"
        + std::to_string(sClients.fetch_add(1));
    auto backend = makeRemoteTransferBackend(transferConfig, name);
    return makeShared<RemoteMemClient>(std::move(name), std::move(options), transferConfig, std::move(backend));
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
    return {chunk.donor, chunk.remoteAddr + static_cast<uint64_t>(s - mFirstSlot[index]) * mSlotSize};
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
