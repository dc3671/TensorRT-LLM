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

#pragma once

#include "kv_cache_manager_v2/storage/remoteMem.h"

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tensorrt_llm::executor::kv_cache
{
class BaseTransferAgent;
class TransferStatus;
} // namespace tensorrt_llm::executor::kv_cache

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

//! The NIXL transport: one-sided RDMA READ / WRITE through this client's own agent, finished on the host before a
//! transfer returns. Donors publish their agent's AgentDesc in donor.<name>.desc.
class NixlRemoteMemClient final : public RemoteMemClient
{
public:
    explicit NixlRemoteMemClient(RemoteMemOptions options);
    ~NixlRemoteMemClient() override;

    [[nodiscard]] bool streamOrdered() const noexcept override
    {
        return false;
    }

    void registerLocal(MemAddress base, size_t bytes) override;
    void deregisterLocal(MemAddress base, size_t bytes) noexcept override;

protected:
    [[nodiscard]] char const* transportName() const noexcept override
    {
        return "nixl";
    }

    [[nodiscard]] char const* payloadSuffix() const noexcept override
    {
        return "desc";
    }

    [[nodiscard]] bool acceptPayload(Donor const& /*donor*/) const override
    {
        return true;
    }

    uint64_t attachChunk(Donor& donor, size_t chunk) override;
    //! Waits for every request it submitted, cancelling the ones still running at the transfer timeout, before it
    //! returns or throws.
    void doTransfer(bool write, std::vector<RemoteIo> const& ios, size_t bytes, CUstream stream) override;

private:
    //! Deregister local memory, forget the donors and destroy the agent; the destructor and a failed constructor
    //! run it.
    void releaseTransport() noexcept;

    std::unique_ptr<executor::kv_cache::BaseTransferAgent> mNixlAgent;
    std::unordered_map<MemAddress, std::pair<size_t, int>> mLocalRegions; // base -> (bytes, refs)
    // Requests that timed out, cancelled or not (a cancel does not prove the remote side quiescent); kept until the
    // agent goes, and the claims with them.
    std::vector<std::unique_ptr<executor::kv_cache::TransferStatus>> mUncancelled;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
