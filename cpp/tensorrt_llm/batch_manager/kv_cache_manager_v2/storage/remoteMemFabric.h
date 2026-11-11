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

#include <cuda.h>
#include <map>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

//! The NVLink fabric transport: each claimed chunk (one fabric allocation of the lender) is imported and mapped into
//! this client's device, and pages are copied on the caller's stream. Lenders publish one CUmemFabricHandle per chunk
//! in lender.<name>.fabric.
class FabricRemoteMemClient final : public RemoteMemClient
{
public:
    //! Needs the device's primary context current on the calling thread.
    explicit FabricRemoteMemClient(RemoteMemOptions options);
    ~FabricRemoteMemClient() override;

    [[nodiscard]] bool streamOrdered() const noexcept override
    {
        return true;
    }

    void drain() noexcept override;

protected:
    [[nodiscard]] char const* transportName() const noexcept override
    {
        return "fabric";
    }

    [[nodiscard]] char const* payloadSuffix() const noexcept override
    {
        return "fabric";
    }

    [[nodiscard]] bool acceptPayload(Lender const& lender) const override;
    //! Imports the chunk and maps it whole, readable and writable by this client's device; returns the device address.
    uint64_t attachChunk(Lender& lender, size_t chunk) override;
    //! Enqueues the copies on `stream`, merging runs that are contiguous on both sides within one mapped chunk.
    void doTransfer(bool write, std::vector<RemoteIo> const& ios, size_t bytes, CUstream stream) override;

private:
    //! One imported, mapped fabric chunk: unmapped, its address range freed and its handle released on close.
    struct FabricMapping
    {
        CUdeviceptr address = 0;
        size_t bytes = 0;
        CUmemGenericAllocationHandle handle = 0;
    };

    //! Retain the current device's primary context, which callers enqueue their copies in.
    void retainContext();
    //! Unmap and release what the constructor and the claims acquired; the destructor and a failed constructor run it.
    void releaseTransport() noexcept;
    //! Synchronize mContext, where callers enqueue their copies.
    [[nodiscard]] CUresult synchronizeContext() const noexcept;

    CUdevice mDevice = 0;
    CUcontext mContext = nullptr;                   // mDevice's primary context, retained
    std::vector<FabricMapping> mMappings;           // every mapped chunk
    std::map<CUdeviceptr, CUdeviceptr> mMappingEnd; // start -> end of every mapped chunk
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
