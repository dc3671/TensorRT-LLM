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

// Remote memory behind a remote cache level: host memory that lenders (remote_lender.py, which describes the files)
// publish in a rendezvous directory, claimed in whole chunks with O_EXCL claim.<lender>.<chunk> files. How pages move
// is a transport's: remoteMemNixl.h (one-sided RDMA, host-synchronous) or remoteMemFabric.h (stream-ordered copies).

#include "kv_cache_manager_v2/common.h"
#include "kv_cache_manager_v2/config.h"
#include "kv_cache_manager_v2/exceptions.h"
#include "kv_cache_manager_v2/storage/config.h"
#include "kv_cache_manager_v2/utils/sharedPtr.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

struct RemoteMemOptions
{
    //! Rendezvous directory the lenders publish into.
    std::string directory;
    //! How long a claim waits for a lender with a free chunk (RemoteCacheTierConfig::waitTimeoutMs).
    int64_t waitTimeoutMs = 0;
    //! How long one NIXL transfer may take before it counts as failed (RemoteCacheTierConfig::transferTimeoutMs).
    int64_t transferTimeoutMs = 0;
    RemoteTransport transport = RemoteTransport::NIXL;
    //! NIXL only: the UCX devices of this client's agent; empty uses the process's UCX environment.
    std::string ucxDevices;
    //! Lenders the job starts (RemoteCacheTierConfig::expectedLenders); 0 if unknown.
    uint32_t expectedLenders = 0;

    bool operator==(RemoteMemOptions const&) const = default;
};

//! One chunk of lent memory this process owns.
struct RemoteChunk
{
    uint32_t lender = 0;     // index into the client's lender table
    uint64_t remoteAddr = 0; // the address transfers name the chunk by (RemoteMemClient::attachChunk)
    uint64_t bytes = 0;
};

//! One page to move: `bytes` between local memory at `local` and lender `lender` at `remoteAddr`.
struct RemoteIo
{
    MemAddress local;
    uint32_t lender;
    uint64_t remoteAddr;
};

//! A connection to the lenders of one rendezvous directory. The lender table and the claimed chunks are common to
//! every transport; how pages move is the transport's (makeRemoteMemClient). Owned by a StorageManager and shared by
//! its levels that name the directory; the last one to go releases the claims.
class RemoteMemClient
{
public:
    virtual ~RemoteMemClient();

    RemoteMemClient(RemoteMemClient const&) = delete;
    RemoteMemClient& operator=(RemoteMemClient const&) = delete;

    //! A chunk that holds at least one slot of `minBytes`: from this process's free list, else claimed from a lender,
    //! waiting up to waitTimeoutMs for one, or until all expectedLenders have reported. Throws RemoteMemError when
    //! none appears.
    [[nodiscard]] RemoteChunk acquireChunk(size_t minBytes);
    //! Return a chunk to this process's free list. The claim is kept until the client is destroyed.
    void releaseChunk(RemoteChunk chunk) noexcept;

    //! Whether transfers are copies enqueued on the caller's stream. Otherwise a transfer runs on the host and has
    //! finished when transfer() returns, so its local memory must be ready, and free of stream work, before the call.
    [[nodiscard]] virtual bool streamOrdered() const noexcept = 0;

    //! Register local memory that transfers read from or write into, reference-counted by base address. A transport
    //! that reaches local memory without registration ignores it.
    virtual void registerLocal(MemAddress /*base*/, size_t /*bytes*/) {}

    virtual void deregisterLocal(MemAddress /*base*/, size_t /*bytes*/) noexcept {}

    //! Move `bytes` for every entry, local -> remote (`write`) or remote -> local: enqueued on `stream`, or finished
    //! on return (streamOrdered()). Throws RemoteMemError on any failure; the client has failed() from then on.
    void transfer(bool write, std::vector<RemoteIo> const& ios, size_t bytes, CUstream stream);

    //! Whether a transfer failed. A failed client throws RemoteMemError on every transfer and claim, since the
    //! contents of its remote slots can no longer be trusted.
    [[nodiscard]] bool failed() const noexcept
    {
        return mFailed.load(std::memory_order_acquire);
    }

    //! Wait until every transfer enqueued on a stream so far has finished; a failure is logged (teardown).
    virtual void drain() noexcept {}

    [[nodiscard]] std::string const& directory() const noexcept
    {
        return mOptions.directory;
    }

    [[nodiscard]] RemoteMemOptions const& options() const noexcept
    {
        return mOptions;
    }

protected:
    struct Lender
    {
        std::string name;
        std::string host;
        uint64_t chunkBytes = 0;
        std::vector<std::pair<uint64_t, uint64_t>> segments; // (base, bytes)
        std::vector<std::pair<uint64_t, uint64_t>> chunks;   // (address, bytes), in claim-index order; fabric: offsets
        std::string payload;    // the lender's payloadSuffix() file: what the transport needs to reach it
        bool loaded = false;    // the transport has reached it (attachChunk)
        bool withdrawn = false; // its .ready file is gone: no new claims from it
        size_t nextClaim = 0;   // chunks before it were claimed (by anyone) when last probed: claims start here
        bool full = false;      // a probe found every chunk claimed; skipped until a claim finds no lender at all
    };

    explicit RemoteMemClient(RemoteMemOptions options);

    //! Called by the transport's constructor once the transport is ready: reads the lenders published so far.
    void discoverLenders();

    //! The transport's name, as lenders write it in their .meta ("nixl" when absent).
    [[nodiscard]] virtual char const* transportName() const noexcept = 0;
    //! The suffix of the lender file holding what this transport needs to reach a lender.
    [[nodiscard]] virtual char const* payloadSuffix() const noexcept = 0;
    //! Whether `lender`'s payload fits its chunks; logs why not.
    [[nodiscard]] virtual bool acceptPayload(Lender const& lender) const = 0;
    //! Make the newly claimed chunk `chunk` of `lender` reachable; returns the address transfers name it by. Throws
    //! RemoteMemError when the lender cannot be reached, which passes over that lender.
    virtual uint64_t attachChunk(Lender& lender, size_t chunk) = 0;
    //! The transport's half of transfer(): `ios` is non-empty, mMutex is held and the client has not failed.
    virtual void doTransfer(bool write, std::vector<RemoteIo> const& ios, size_t bytes, CUstream stream) = 0;

    //! Caller holds mMutex. Record the first transfer failure and throw it as RemoteMemError.
    [[noreturn]] void fail(std::string const& message);

    //! The claims outlive this client: a transfer the transport could not stop may still land in a chunk.
    void keepClaims() noexcept
    {
        mKeepClaims = true;
    }

    [[nodiscard]] std::string const& name() const noexcept
    {
        return mName;
    }

    RemoteMemOptions mOptions;
    std::vector<Lender> mLenders;
    mutable std::mutex mMutex;

private:
    //! Add the lenders newly published in the directory, mark the ones whose .ready file is gone as withdrawn, and
    //! count the lenders that reported (.ready or .failed).
    void refreshLenders();
    //! Whether `lender` still publishes its .ready file; marks it withdrawn when not.
    bool stillPublished(Lender& lender);
    //! Claim a free chunk; a lender whose chunk cannot be attached (e.g. it crashed) is marked withdrawn.
    bool tryClaim(size_t minBytes, RemoteChunk& out);
    //! Caller holds mMutex. Throws RemoteMemError when an earlier transfer failed.
    void throwIfFailed() const;

    std::string mName;
    std::unordered_map<std::string, size_t> mLenderIndex;
    std::vector<RemoteChunk> mFreeChunks;
    std::vector<std::string> mClaims;
    uint64_t mClaimedBytes = 0;
    size_t mReportedLenders = 0; // lenders with a .ready or .failed file at the last refresh
    std::string mLastClaimError; // why the last lender was withdrawn at claim time
    bool mKeepClaims = false;
    std::string mFailure;        // the first transfer failure
    std::atomic<bool> mFailed{false};
};

//! The client of `options.directory` over `options.transport`.
[[nodiscard]] SharedPtr<RemoteMemClient> makeRemoteMemClient(RemoteMemOptions options);

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
