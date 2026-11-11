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

// Remote memory behind a cold cache level: host memory that lenders (remote_lender.py, which describes the files)
// publish in a rendezvous directory, claimed in whole chunks with O_EXCL claim.<lender>.<chunk> files. NIXL moves
// pages with synchronous one-sided RDMA; fabric maps claimed chunks and copies on the caller's stream.

#include "kv_cache_manager_v2/common.h"
#include "kv_cache_manager_v2/exceptions.h"
#include "kv_cache_manager_v2/storage/config.h"
#include "kv_cache_manager_v2/utils/sharedPtr.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
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

struct RemoteMemOptions
{
    //! Rendezvous directory the lenders publish into.
    std::string directory;
    //! How long a claim waits for a lender with a free chunk (DiskCacheTierConfig::remoteWaitTimeoutMs).
    int64_t waitTimeoutMs = 0;
    //! How long one NIXL transfer may take before it counts as failed (DiskCacheTierConfig::remoteTransferTimeoutMs).
    int64_t transferTimeoutMs = 0;
    //! Move pages over NVLink fabric memory instead of NIXL.
    bool fabric = false;
    //! NIXL only: the UCX devices of this client's agent; empty uses the process's UCX environment.
    std::string ucxDevices;
    //! Lenders the job starts (DiskCacheTierConfig::remoteExpectedLenders); 0 if unknown.
    uint32_t expectedLenders = 0;
};

//! One chunk of lent memory this process owns.
struct RemoteChunk
{
    uint32_t lender = 0;     // index into the client's lender table
    uint64_t remoteAddr = 0; // NIXL: address in the lender's memory; fabric: address of our mapping
    uint64_t bytes = 0;
};

//! One page to move: `bytes` between local memory at `local` and lender `lender` at `remoteAddr`.
struct RemoteIo
{
    MemAddress local;
    uint32_t lender;
    uint64_t remoteAddr;
};

//! A connection to the lenders of one rendezvous directory: the lender table, the chunks this client claimed and,
//! for NIXL, one agent. Owned by a StorageManager and shared by its levels that name the directory; the last one to
//! go releases the claims.
class RemoteMemClient
{
public:
    explicit RemoteMemClient(RemoteMemOptions options);
    ~RemoteMemClient();

    RemoteMemClient(RemoteMemClient const&) = delete;
    RemoteMemClient& operator=(RemoteMemClient const&) = delete;

    //! A chunk that holds at least one slot of `minBytes`: from this process's free list, else claimed from a lender,
    //! waiting up to waitTimeoutMs for one, or until all expectedLenders have reported. Throws RemoteMemError when
    //! none appears.
    [[nodiscard]] RemoteChunk acquireChunk(size_t minBytes);
    //! Return a chunk to this process's free list. The claim is kept until the client is destroyed.
    void releaseChunk(RemoteChunk chunk) noexcept;

    //! NIXL: register local memory that transfers read from or write into; reference-counted by base address.
    void registerLocal(MemAddress base, size_t bytes);
    void deregisterLocal(MemAddress base, size_t bytes) noexcept;

    //! NIXL only: move `bytes` for every entry, local -> remote (`write`) or remote -> local, and wait for all of it.
    //! Throws RemoteMemError on any failure or timeout, after waiting for every request it submitted (cancelling the
    //! ones still running at the timeout); the client has failed() from then on.
    void transfer(bool write, std::vector<RemoteIo> const& ios, size_t bytes);

    //! Fabric only: enqueue the copies of one transfer on `stream`, which orders them after the work that wrote the
    //! source and before the work that reads the destination.
    void transferOnStream(bool write, std::vector<RemoteIo> const& ios, size_t bytes, CUstream stream);

    //! Whether a transfer failed. A failed client throws RemoteMemError on every transfer and claim, since the
    //! contents of its remote slots can no longer be trusted.
    [[nodiscard]] bool failed() const noexcept
    {
        return mFailed.load(std::memory_order_acquire);
    }

    //! Whether this client moves pages over NVLink fabric memory (stream-ordered) instead of NIXL (synchronous).
    [[nodiscard]] bool fabric() const noexcept
    {
        return mOptions.fabric;
    }

    //! Wait until every copy enqueued on a stream so far has finished; a failure is logged (teardown).
    void drain() noexcept;

    [[nodiscard]] std::string const& directory() const noexcept
    {
        return mOptions.directory;
    }

    [[nodiscard]] RemoteMemOptions const& options() const noexcept
    {
        return mOptions;
    }

private:
    struct Lender
    {
        std::string name;
        std::string host;
        uint64_t chunkBytes = 0;
        std::vector<std::pair<uint64_t, uint64_t>> segments; // (base, bytes)
        std::vector<std::pair<uint64_t, uint64_t>> chunks;   // (address, bytes), in claim-index order; fabric: offsets
        std::string desc;                                    // NIXL: serialized AgentDesc
        std::string fabricHandles;                           // fabric: one CUmemFabricHandle per chunk, in order
        bool loaded = false;                                 // NIXL: loaded into our agent
        bool withdrawn = false;                              // its .ready file is gone: no new claims from it
        size_t nextClaim = 0; // chunks before it were claimed (by anyone) when last probed: claims start here
        bool full = false;    // a probe found every chunk claimed; skipped until a claim finds no lender at all
    };

    //! One imported, mapped fabric chunk: unmapped, its address range freed and its handle released on close.
    struct FabricMapping
    {
        CUdeviceptr address = 0;
        size_t bytes = 0;
        CUmemGenericAllocationHandle handle = 0;
    };

    //! Add the lenders newly published in the directory, mark the ones whose .ready file is gone as withdrawn, and
    //! count the lenders that reported (.ready or .failed).
    void refreshLenders();
    //! Whether `lender` still publishes its .ready file; marks it withdrawn when not.
    bool stillPublished(Lender& lender);
    //! Claim a free chunk; a lender whose chunk cannot be loaded or mapped (e.g. it crashed) is marked withdrawn.
    bool tryClaim(size_t minBytes, RemoteChunk& out);
    //! Caller holds mMutex. Throws RemoteMemError when an earlier transfer failed.
    void throwIfFailed() const;
    //! Caller holds mMutex. Record the first transfer failure and throw it as RemoteMemError.
    [[noreturn]] void fail(std::string const& message);
    void ensureLoaded(Lender& lender);

    void startFabric();
    //! Unmap and release what the constructor acquired; the destructor and a failed constructor run it.
    void releaseTransport() noexcept;
    //! Fabric: synchronize mContext, where callers enqueue their copies.
    [[nodiscard]] CUresult synchronizeContext() const noexcept;
    //! Import chunk `chunk` of the lender (its own fabric allocation of `bytes`) and map it whole, readable and
    //! writable by this client's device; returns the device address.
    CUdeviceptr mapChunk(Lender const& lender, size_t chunk, uint64_t bytes);
    //! Enqueue the copies of one transfer on `stream`, merging runs that are contiguous on both sides within one
    //! mapped chunk.
    void fabricCopy(bool write, std::vector<RemoteIo> const& ios, size_t bytes, CUstream stream);

    CUdevice mDevice = 0;
    CUcontext mContext = nullptr;                   // fabric: mDevice's primary context, retained
    std::vector<FabricMapping> mMappings;           // fabric: every mapped chunk
    std::map<CUdeviceptr, CUdeviceptr> mMappingEnd; // fabric: start -> end of every mapped chunk

    RemoteMemOptions mOptions;
    std::string mName;
    std::unique_ptr<executor::kv_cache::BaseTransferAgent> mNixlAgent;
    std::vector<Lender> mLenders;
    std::unordered_map<std::string, size_t> mLenderIndex;
    std::vector<RemoteChunk> mFreeChunks;
    std::vector<std::string> mClaims;
    uint64_t mClaimedBytes = 0;
    size_t mReportedLenders = 0; // lenders with a .ready or .failed file at the last refresh
    std::string mLastClaimError; // why the last lender was withdrawn at claim time
    std::unordered_map<MemAddress, std::pair<size_t, int>> mLocalRegions; // base -> (bytes, refs)
    // NIXL requests that timed out, cancelled or not (a cancel does not prove the remote side quiescent); kept until
    // the agent goes, and the claims with them.
    std::vector<std::unique_ptr<executor::kv_cache::TransferStatus>> mUncancelled;
    std::string mFailure; // the first transfer failure
    std::atomic<bool> mFailed{false};
    mutable std::mutex mMutex;
};

//! A pool's slots in remote memory: whole chunks, each holding floor(chunk bytes / slot size) slots, in order.
class RemoteRange
{
public:
    RemoteRange(SharedPtr<RemoteMemClient> client, size_t slotSize);
    ~RemoteRange();

    RemoteRange(RemoteRange const&) = delete;
    RemoteRange& operator=(RemoteRange const&) = delete;

    //! Hold enough chunks for `numSlots` slots. Growing claims chunks; shrinking keeps them (a later grow reuses them).
    void reserve(SlotCount numSlots);

    //! Where slot `slot` lives: (lender index, remote address).
    [[nodiscard]] std::pair<uint32_t, uint64_t> locate(SlotId slot) const;

    [[nodiscard]] RemoteMemClient& client() const noexcept
    {
        return *mClient;
    }

private:
    SharedPtr<RemoteMemClient> mClient;
    size_t mSlotSize = 0;
    std::vector<RemoteChunk> mChunks;
    std::vector<SlotCount> mFirstSlot; // first slot of each chunk; mFirstSlot.size() == mChunks.size()
    SlotCount mCapacity{0};
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
