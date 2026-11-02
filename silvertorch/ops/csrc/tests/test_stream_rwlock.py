# pyre-unsafe

"""Tests for StreamRWLock (stream_rwlock.cuh).

StreamRWLock is a header-only CUDA synchronization primitive, so it is exercised
by compiling a small harness against the header with
``torch.utils.cpp_extension.load_inline`` -- nothing is registered as a
``torch.ops.st.*`` op. The harness runs reader/writer scenarios on real CUDA
streams, including one where every reader and writer lives on its own host
thread with its own stream, and returns the values observed on the device.

The kernels spin deliberately, so a broken cross-stream fence lets the competing
stream race ahead and yields the wrong value; with the lock in place every
scenario is deterministic regardless of the spin duration.
"""

import os

import pytest
import torch
from torch.utils.cpp_extension import load_inline

CSRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# A large-enough device spin so that, without the lock, the fast kernel on the
# competing stream would almost certainly win the race. Correctness with the
# lock does not depend on this value.
SPIN_CYCLES = 20_000_000

# The threaded scenario serializes every writer, so keep its spin shorter.
THREADED_SPIN_CYCLES = 2_000_000

# Compiling the harness and spinning on the device takes minutes, so these are
# opt-in via `pytest --run-slow`.
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="StreamRWLock requires CUDA"),
    pytest.mark.slow,
]

CPP_SOURCE = """
std::vector<int64_t> writer_then_reader(int64_t spin_cycles);
std::vector<int64_t> reader_then_writer(int64_t spin_cycles);
std::vector<int64_t> serialized_writers(int64_t num_writers, int64_t spin_cycles);
std::map<std::string, std::vector<int64_t>> threaded_readers_and_writers(
    int64_t num_readers, int64_t num_writers, int64_t spin_cycles);
"""

CUDA_SOURCE = """
#include "stream_rwlock.cuh"

#include <algorithm>
#include <atomic>
#include <exception>
#include <map>
#include <string>
#include <thread>
#include <vector>

namespace {

using st::ops::stream_rwlock::LockGuard;
using st::ops::stream_rwlock::StreamRWLock;

constexpr int kNumElems = 1024;
constexpr int kThreadsPerBlock = 256;
constexpr int kBlocks = (kNumElems + kThreadsPerBlock - 1) / kThreadsPerBlock;

// Stored by windowed_read_kernel when the value changed mid-read.
constexpr int kTornRead = -1;

#define CUDA_CHECK(expr)                                             \\
  do {                                                               \\
    cudaError_t err__ = (expr);                                      \\
    TORCH_CHECK(                                                     \\
        err__ == cudaSuccess, #expr, " failed: ",                    \\
        cudaGetErrorString(err__));                                  \\
  } while (0)

// Busy-wait on the device for `spin` SM clock cycles. Keeps a fast kernel on a
// competing stream from silently winning a race if ordering were broken.
__device__ __forceinline__ void busy_wait(long long spin) {
  long long start = clock64();
  while (clock64() - start < spin) {
    // spin
  }
}

__global__ void fill_kernel(int* buf, int n, int value) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    buf[i] = value;
  }
}

__global__ void copy_kernel(const int* src, int* dst, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    dst[i] = src[i];
  }
}

__global__ void add_kernel(int* buf, const int* addend, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    buf[i] += addend[i];
  }
}

// Slow write: spin, then stamp `value` into every slot.
__global__ void slow_fill_kernel(int* buf, int n, int value, long long spin) {
  busy_wait(spin);
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    buf[i] = value;
  }
}

// Slow read: spin, then snapshot `src` into `dst`.
__global__ void
slow_copy_kernel(const int* src, int* dst, int n, long long spin) {
  busy_wait(spin);
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    dst[i] = src[i];
  }
}

// Slow read-modify-write: tmp = buf; spin; buf = tmp + 1. Concurrent RMWs that
// are not mutually exclusive would lose updates.
__global__ void slow_increment_kernel(int* buf, int n, long long spin) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int tmp = buf[i];
    busy_wait(spin);
    buf[i] = tmp + 1;
  }
}

// Reads `src` twice with a spin in between, so the read spans a long window
// instead of a single instant, and stores kTornRead if the value changed
// underneath it. `volatile` keeps the second read from being folded into the
// first. A write landing anywhere inside the window is therefore visible to the
// caller, which a single-instant read would almost always miss.
__global__ void windowed_read_kernel(
    const volatile int* src,
    int* dst,
    int n,
    long long spin) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int first = src[i];
    busy_wait(spin);
    int second = src[i];
    dst[i] = (first == second) ? first : kTornRead;
  }
}

// Owning handle for a plain (non-caching-allocator) device int buffer.
struct DeviceInts {
  int* ptr = nullptr;
  int size = 0;

  explicit DeviceInts(int n) : size(n) {
    CUDA_CHECK(cudaMalloc(&ptr, static_cast<size_t>(n) * sizeof(int)));
  }
  DeviceInts(const DeviceInts&) = delete;
  DeviceInts& operator=(const DeviceInts&) = delete;
  ~DeviceInts() { cudaFree(ptr); }

  void fill(int value) {
    fill_kernel<<<(size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                  kThreadsPerBlock>>>(ptr, size, value);
  }

  // [min, max] over `count` elements starting at `offset`.
  std::vector<int64_t> min_max(int offset = 0, int count = -1) const {
    int n = count < 0 ? size : count;
    std::vector<int> host(n);
    CUDA_CHECK(cudaMemcpy(
        host.data(), ptr + offset, n * sizeof(int), cudaMemcpyDeviceToHost));
    auto [mn, mx] = std::minmax_element(host.begin(), host.end());
    return {static_cast<int64_t>(*mn), static_cast<int64_t>(*mx)};
  }
};

// CUDA loads a kernel lazily on its first launch, and that load synchronizes
// with work already queued on the device -- enough to hide a missing fence on
// whichever scenario happens to run first. Touch every kernel up front so the
// scenarios below measure stream ordering rather than module loading.
void warm_up_kernels() {
  DeviceInts src(kNumElems);
  DeviceInts dst(kNumElems);
  fill_kernel<<<kBlocks, kThreadsPerBlock>>>(src.ptr, kNumElems, 0);
  copy_kernel<<<kBlocks, kThreadsPerBlock>>>(src.ptr, dst.ptr, kNumElems);
  add_kernel<<<kBlocks, kThreadsPerBlock>>>(dst.ptr, src.ptr, kNumElems);
  slow_fill_kernel<<<kBlocks, kThreadsPerBlock>>>(src.ptr, kNumElems, 0, 0);
  slow_copy_kernel<<<kBlocks, kThreadsPerBlock>>>(src.ptr, dst.ptr, kNumElems, 0);
  slow_increment_kernel<<<kBlocks, kThreadsPerBlock>>>(src.ptr, kNumElems, 0);
  windowed_read_kernel<<<kBlocks, kThreadsPerBlock>>>(
      src.ptr, dst.ptr, kNumElems, 0);
  CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace

// Scenario: a writer publishes a value, then a reader reads it. The reader's
// r_lock waits on mutation_done_, so the (fast) read on the reader stream must
// observe the (slow) write on the mutation stream.
//
// Returns [min, max] of the value the reader read.
std::vector<int64_t> writer_then_reader(int64_t spin_cycles) {
  constexpr int kWritten = 7;
  StreamRWLock rwlock;
  DeviceInts buf(kNumElems);
  DeviceInts out(kNumElems);
  cudaStream_t reader = nullptr;
  cudaStream_t writer = nullptr;
  CUDA_CHECK(cudaStreamCreate(&reader));
  CUDA_CHECK(cudaStreamCreate(&writer));

  warm_up_kernels();
  buf.fill(0);
  out.fill(-1);
  CUDA_CHECK(cudaDeviceSynchronize());

  {
    auto wlock = rwlock.get_wlock(writer);
    LockGuard<StreamRWLock::WLock> guard(wlock);
    slow_fill_kernel<<<kBlocks, kThreadsPerBlock, 0, guard.get_stream()>>>(
        buf.ptr, kNumElems, kWritten, spin_cycles);
  }
  {
    auto rlock = rwlock.get_rlock(reader);
    LockGuard<StreamRWLock::RLock> guard(rlock);
    copy_kernel<<<kBlocks, kThreadsPerBlock, 0, guard.get_stream()>>>(
        buf.ptr, out.ptr, kNumElems);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  auto result = out.min_max();
  CUDA_CHECK(cudaStreamDestroy(reader));
  CUDA_CHECK(cudaStreamDestroy(writer));
  return result;
}

// Scenario: a reader is in flight (still registered) when a writer acquires the
// lock. w_lock's wait_for_all_read_streams makes the mutation stream wait for
// the reader, so the (slow) read observes the pre-write value and the (fast)
// overwrite only lands afterwards.
//
// Uses the RLock handle's lock()/unlock() directly (not LockGuard) so the
// reader stays registered across the write -- LockGuard holds the lock by value
// and would unregister the reader when its copy is destroyed.
//
// Returns [read_min, read_max, buf_min, buf_max].
std::vector<int64_t> reader_then_writer(int64_t spin_cycles) {
  constexpr int kBefore = 5;
  constexpr int kAfter = 9;
  StreamRWLock rwlock;
  DeviceInts buf(kNumElems);
  DeviceInts out(kNumElems);
  cudaStream_t reader = nullptr;
  cudaStream_t writer = nullptr;
  CUDA_CHECK(cudaStreamCreate(&reader));
  CUDA_CHECK(cudaStreamCreate(&writer));

  warm_up_kernels();
  buf.fill(kBefore);
  out.fill(-1);
  CUDA_CHECK(cudaDeviceSynchronize());

  {
    auto rlock = rwlock.get_rlock(reader);
    // Launch the slow read while holding the read lock, then release the launch
    // mutex but keep `rlock` alive (and the reader registered) across the write.
    rlock.lock();
    slow_copy_kernel<<<kBlocks, kThreadsPerBlock, 0, rlock.get_stream()>>>(
        buf.ptr, out.ptr, kNumElems, spin_cycles);
    rlock.unlock();

    {
      auto wlock = rwlock.get_wlock(writer);
      LockGuard<StreamRWLock::WLock> guard(wlock);
      fill_kernel<<<kBlocks, kThreadsPerBlock, 0, guard.get_stream()>>>(
          buf.ptr, kNumElems, kAfter);
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  auto result = out.min_max();
  auto after = buf.min_max();
  result.insert(result.end(), after.begin(), after.end());

  CUDA_CHECK(cudaStreamDestroy(reader));
  CUDA_CHECK(cudaStreamDestroy(writer));
  return result;
}

// Scenario: num_writers writers, each on its own stream, do a slow
// read-modify-write increment from a single host thread. Writers are mutually
// exclusive, so no update is lost.
//
// Returns [min, max] of the accumulated counter.
std::vector<int64_t> serialized_writers(int64_t num_writers, int64_t spin_cycles) {
  TORCH_CHECK(num_writers > 0, "num_writers must be positive");
  StreamRWLock rwlock;
  DeviceInts buf(kNumElems);
  std::vector<cudaStream_t> streams(num_writers, nullptr);
  for (auto& s : streams) {
    CUDA_CHECK(cudaStreamCreate(&s));
  }

  warm_up_kernels();
  buf.fill(0);
  CUDA_CHECK(cudaDeviceSynchronize());

  for (auto& stream : streams) {
    auto wlock = rwlock.get_wlock(stream);
    LockGuard<StreamRWLock::WLock> guard(wlock);
    slow_increment_kernel<<<kBlocks, kThreadsPerBlock, 0, guard.get_stream()>>>(
        buf.ptr, kNumElems, spin_cycles);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  auto result = buf.min_max();
  for (auto& s : streams) {
    CUDA_CHECK(cudaStreamDestroy(s));
  }
  return result;
}

// Scenario: num_readers + num_writers host threads, each owning its own CUDA
// stream, all race for the same lock. Every writer produces its addend on its
// own stream, adds it to the shared buffer under the write lock, then reads the
// buffer back on its own stream; every reader takes a windowed snapshot under
// the read lock. Together they exercise all four fences of the lock:
//
//   * r_lock waits for the in-flight mutation  -> readers see complete writes
//   * w_lock waits for registered readers      -> no write lands mid-read
//   * w_lock waits for the caller's stream     -> the addend is ready, so the
//                                                 total is num_writers
//   * w_unlock waits for the mutation          -> a writer's own read-back
//                                                 already includes its addend
//
// Returns a dict of [min, max] pairs: "buffer" for the final accumulator,
// "reader_<i>" for reader i's snapshot, "writer_<i>" for what writer i read
// back after its own write.
std::map<std::string, std::vector<int64_t>> threaded_readers_and_writers(
    int64_t num_readers,
    int64_t num_writers,
    int64_t spin_cycles) {
  TORCH_CHECK(num_readers >= 0, "num_readers must be non-negative");
  TORCH_CHECK(num_writers >= 0, "num_writers must be non-negative");
  const int readers = static_cast<int>(num_readers);
  const int writers = static_cast<int>(num_writers);
  const int total = readers + writers;
  TORCH_CHECK(total > 0, "need at least one reader or writer");

  StreamRWLock rwlock;
  DeviceInts buf(kNumElems);
  DeviceInts addends(std::max(writers, 1) * kNumElems);
  DeviceInts snapshots(std::max(readers, 1) * kNumElems);
  DeviceInts read_backs(std::max(writers, 1) * kNumElems);

  warm_up_kernels();
  buf.fill(0);
  addends.fill(0);
  snapshots.fill(kTornRead);
  read_backs.fill(-1);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Release every thread at once so they really contend for the lock.
  std::atomic<int> arrived{0};
  std::atomic<bool> go{false};
  auto wait_for_start = [&] {
    arrived.fetch_add(1, std::memory_order_release);
    while (!go.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
  };

  // TORCH_CHECK throws, and an exception escaping a thread would terminate the
  // process, so hand failures back to the caller instead.
  std::vector<std::exception_ptr> errors(total);
  std::vector<std::thread> threads;
  threads.reserve(total);

  for (int r = 0; r < readers; ++r) {
    threads.emplace_back([&, r] {
      try {
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));
        wait_for_start();
        {
          auto rlock = rwlock.get_rlock(stream);
          // Keep the read lock (and this stream's registration) alive until the
          // read has finished, so no writer can overwrite what it is reading.
          rlock.lock();
          windowed_read_kernel<<<kBlocks, kThreadsPerBlock, 0,
                                 rlock.get_stream()>>>(
              buf.ptr, snapshots.ptr + r * kNumElems, kNumElems, spin_cycles);
          rlock.unlock();
          CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
      } catch (...) {
        errors[r] = std::current_exception();
      }
    });
  }

  for (int w = 0; w < writers; ++w) {
    threads.emplace_back([&, w] {
      try {
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));
        wait_for_start();
        int* addend = addends.ptr + w * kNumElems;
        // Produce this writer's addend on its own stream, slowly: the mutation
        // must wait for it, or it would add zeros instead of ones.
        slow_fill_kernel<<<kBlocks, kThreadsPerBlock, 0, stream>>>(
            addend, kNumElems, 1, spin_cycles);
        {
          auto wlock = rwlock.get_wlock(stream);
          LockGuard<StreamRWLock::WLock> guard(wlock);
          add_kernel<<<kBlocks, kThreadsPerBlock, 0, guard.get_stream()>>>(
              buf.ptr, addend, kNumElems);
        }
        // w_unlock made this stream wait for the mutation, so a read issued
        // here must already include this writer's own addend.
        copy_kernel<<<kBlocks, kThreadsPerBlock, 0, stream>>>(
            buf.ptr, read_backs.ptr + w * kNumElems, kNumElems);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
      } catch (...) {
        errors[readers + w] = std::current_exception();
      }
    });
  }

  while (arrived.load(std::memory_order_acquire) < total) {
    std::this_thread::yield();
  }
  go.store(true, std::memory_order_release);
  for (auto& t : threads) {
    t.join();
  }
  for (auto& error : errors) {
    if (error) {
      std::rethrow_exception(error);
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  std::map<std::string, std::vector<int64_t>> results;
  results["buffer"] = buf.min_max();
  for (int r = 0; r < readers; ++r) {
    results["reader_" + std::to_string(r)] =
        snapshots.min_max(r * kNumElems, kNumElems);
  }
  for (int w = 0; w < writers; ++w) {
    results["writer_" + std::to_string(w)] =
        read_backs.min_max(w * kNumElems, kNumElems);
  }
  return results;
}
"""


@pytest.fixture(scope="module")
def lock_harness():
    """Compile the StreamRWLock harness against the header (cached by torch)."""
    return load_inline(
        name="stream_rwlock_harness",
        cpp_sources=CPP_SOURCE,
        cuda_sources=CUDA_SOURCE,
        functions=[
            "writer_then_reader",
            "reader_then_writer",
            "serialized_writers",
            "threaded_readers_and_writers",
        ],
        extra_include_paths=[CSRC_DIR],
        extra_cuda_cflags=["--expt-relaxed-constexpr", "-O3"],
        verbose=False,
    )


def test_writer_then_reader_visibility(lock_harness) -> None:
    """A reader must observe a write that a prior writer published."""
    assert lock_harness.writer_then_reader(SPIN_CYCLES) == [7, 7]


def test_writer_waits_for_inflight_reader(lock_harness) -> None:
    """A writer must not overwrite data an in-flight reader is still reading."""
    read_min, read_max, buf_min, buf_max = lock_harness.reader_then_writer(SPIN_CYCLES)
    # The (slow) reader snapshotted the pre-write value everywhere ...
    assert (read_min, read_max) == (5, 5)
    # ... and the write still landed afterwards.
    assert (buf_min, buf_max) == (9, 9)


@pytest.mark.parametrize("num_writers", [1, 2, 8, 16])
def test_writers_are_serialized(lock_harness, num_writers: int) -> None:
    """Mutually exclusive writers must not lose read-modify-write updates."""
    assert lock_harness.serialized_writers(num_writers, SPIN_CYCLES) == [num_writers, num_writers]


def test_zero_spin_still_correct(lock_harness) -> None:
    """Ordering holds even with no artificial delay in the kernels."""
    assert lock_harness.writer_then_reader(0) == [7, 7]
    assert lock_harness.reader_then_writer(0) == [5, 5, 9, 9]


@pytest.mark.parametrize(
    "num_readers,num_writers",
    [(0, 4), (4, 0), (1, 1), (4, 4), (8, 8)],
)
def test_threaded_readers_and_writers(lock_harness, num_readers: int, num_writers: int) -> None:
    """Threads that each own a stream must still see serialized, atomic writes."""
    results = lock_harness.threaded_readers_and_writers(num_readers, num_writers, THREADED_SPIN_CYCLES)

    # Every writer's addend was ready before its write and no write was lost,
    # however the threads interleaved.
    assert results.pop("buffer") == [num_writers, num_writers]

    for reader in range(num_readers):
        snapshot_min, snapshot_max = results.pop(f"reader_{reader}")
        # The read is a consistent snapshot: uniform, never straddling a write
        # (which would leave the torn-read marker -1) ...
        assert snapshot_min == snapshot_max
        # ... and the value it saw is one a serialized writer actually published.
        assert 0 <= snapshot_min <= num_writers

    for writer in range(num_writers):
        read_back_min, read_back_max = results.pop(f"writer_{writer}")
        # Reading back on the caller's own stream sees at least this writer's
        # own addend, and at most every writer's.
        assert read_back_min >= 1
        assert read_back_max <= num_writers

    assert not results
