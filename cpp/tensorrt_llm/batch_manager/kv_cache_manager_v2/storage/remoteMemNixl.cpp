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

#include "kv_cache_manager_v2/storage/remoteMemNixl.h"
#include "kv_cache_manager_v2/exceptions.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/transferAgent.h"

#include <algorithm>
#include <chrono>
#include <string>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

namespace kvc = tensorrt_llm::executor::kv_cache;

NixlRemoteMemClient::NixlRemoteMemClient(RemoteMemOptions options)
    : RemoteMemClient(std::move(options))
{
    try
    {
        kvc::BaseAgentConfig config{};
        config.mName = name();
        // NIXL's progress thread completes connection set-up with donors that never call into NIXL themselves.
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
        discoverDonors();
    }
    catch (...)
    {
        // No destructor runs for a failed constructor, and nothing is claimed yet.
        releaseTransport();
        throw;
    }
}

NixlRemoteMemClient::~NixlRemoteMemClient()
{
    // A WRITE that timed out may still land in its chunk, so after one the chunks stay claimed.
    if (!mUncancelled.empty())
    {
        keepClaims();
    }
    releaseTransport();
}

void NixlRemoteMemClient::releaseTransport() noexcept
{
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
        for (auto const& donor : mDonors)
        {
            if (donor.loaded)
            {
                mNixlAgent->invalidateRemoteAgent(donor.name);
            }
        }
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_WARNING("KVCM2 remote level: teardown of %s: %s", name().c_str(), e.what());
    }
    mUncancelled.clear();
    mNixlAgent.reset();
}

uint64_t NixlRemoteMemClient::attachChunk(Donor& donor, size_t chunk)
{
    if (!donor.loaded)
    {
        try
        {
            mNixlAgent->loadRemoteAgent(donor.name, kvc::AgentDesc::deserialize(donor.payload));
        }
        catch (std::exception const& e)
        {
            throw RemoteMemError(
                "KVCM2 remote level: loading donor " + donor.name + " (" + donor.host + ") failed: " + e.what());
        }
        donor.loaded = true;
    }
    return donor.chunks[chunk].first;
}

void NixlRemoteMemClient::registerLocal(MemAddress base, size_t bytes)
{
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

void NixlRemoteMemClient::deregisterLocal(MemAddress base, size_t bytes) noexcept
{
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

void NixlRemoteMemClient::doTransfer(bool write, std::vector<RemoteIo> const& ios, size_t bytes, CUstream /*stream*/)
{
    std::vector<std::vector<size_t>> byDonor(mDonors.size());
    for (size_t i = 0; i < ios.size(); ++i)
    {
        TLLM_CHECK_DEBUG(ios[i].donor < mDonors.size());
        byDonor[ios[i].donor].push_back(i);
    }
    auto const opName = write ? std::string("WRITE to") : std::string("READ from");
    std::vector<std::pair<uint32_t, std::unique_ptr<kvc::TransferStatus>>> pending;
    std::string error; // the first failure; the requests already submitted settle before it is thrown
    for (uint32_t donorIdx = 0; donorIdx < byDonor.size(); ++donorIdx)
    {
        auto const& entries = byDonor[donorIdx];
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
            kvc::MemoryDescs(kvc::MemoryType::kDRAM, std::move(remote)), mDonors[donorIdx].name);
        try
        {
            pending.emplace_back(donorIdx, mNixlAgent->submitTransferRequests(request));
        }
        catch (std::exception const& e)
        {
            error = opName + " donor " + mDonors[donorIdx].name + " (" + mDonors[donorIdx].host
                + "): submitting failed: " + e.what();
            break;
        }
    }
    // Wait for every request, cancelling the ones still in flight at the deadline, before returning or throwing:
    // the caller reuses the local buffers, which a late request could still write.
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(mOptions.transferTimeoutMs);
    for (auto& [donorIdx, status] : pending)
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
                TLLM_LOG_ERROR("KVCM2 remote level: a timed-out %s donor %s (%s) could not be cancelled",
                    opName.c_str(), mDonors[donorIdx].name.c_str(), mDonors[donorIdx].host.c_str());
            }
            mUncancelled.push_back(std::move(status));
        }
        if (error.empty())
        {
            std::string const why = state == kvc::TransferState::kIN_PROGRESS
                ? "timed out after " + std::to_string(mOptions.transferTimeoutMs) + " ms"
                : "failed (" + status->getLastStatusStr() + ")";
            error = opName + " donor " + mDonors[donorIdx].name + " (" + mDonors[donorIdx].host + ") of "
                + std::to_string(byDonor[donorIdx].size()) + " page(s) x " + std::to_string(bytes) + " B " + why;
        }
    }
    if (!error.empty())
    {
        fail("KVCM2 remote level: " + error);
    }
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
