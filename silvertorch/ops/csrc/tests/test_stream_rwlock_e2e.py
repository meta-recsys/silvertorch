# pyre-unsafe

"""End-to-end test for StreamRWLock (stream_rwlock.cuh).

The workload is a sum that must not tear:

* the shared resource is a device buffer holding ``n`` distinct floats;
* a **reader** sums the buffer, one element at a time, spinning between elements
  so the read spans a long window instead of a single instant;
* a **writer** swaps two random elements, so every mutation leaves the buffer a
  permutation of itself and therefore leaves the *sum* invariant.

Because a swap preserves the sum, any reader that gets a consistent view -- the
buffer entirely before or entirely after each swap -- must return the initial
sum. A reader that straddles a mutation does not: it either sees the swap
half-applied (one value counted twice, the other not at all) or sees the source
element before the swap and the destination element after it. Both show up as a
sum that differs from the invariant, which makes "was this view consistent?" a
single exact float comparison.

Four reader threads and two writer threads, each owning its own CUDA stream, run
the workload concurrently. The same scenario also runs with the lock bypassed
(``use_lock=False``, every thread launching straight onto its own stream), which
is expected to tear -- that is what shows the invariant is held by the lock and
not by the workload.

StreamRWLock is a header-only CUDA primitive, so the harness is compiled against
the header with ``torch.utils.cpp_extension.load_inline``; nothing is registered
as a ``torch.ops.st.*`` op.

The last test profiles the locked and unlocked runs and exports Chrome traces, so
the read/read overlap and the read/write exclusion are also checkable on a
timeline. It asserts both properties from the trace, and prints the trace paths
(run pytest with ``-s`` to see them) for viewing at https://ui.perfetto.dev or
``chrome://tracing``. Traces are written to a pytest temp dir unless
``STREAM_RWLOCK_TRACE_DIR`` says otherwise; one run of each configuration is
checked in under ``tests/traces/`` for reference.

These tests are marked ``slow``, so they need ``-S`` / ``--run-slow``::

    pytest silvertorch/ops/csrc/tests/test_stream_rwlock_e2e.py -S -s
"""

import json
import os
import warnings
from itertools import combinations

import pytest
import torch
from torch.profiler import ProfilerActivity
from torch.profiler import profile

CSRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Compiling the harness and spinning on the device takes minutes, so these are
# opt-in via `pytest --run-slow`.
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="StreamRWLock requires CUDA"),
    pytest.mark.slow,
]

# Workload shape: 4 readers + 2 writers, as a demonstration size. The rounds are
# picked so the writers stay busy for as long as the readers do (a reader round
# costs about as much as three writer rounds), leaving no read uncontended.
NUM_ELEMS = 1024
NUM_READERS = 4
NUM_WRITERS = 2
READS_PER_READER = 4
WRITES_PER_WRITER = 12

# Device spin, in SM clock cycles. The reader pauses READ_SPIN cycles after every
# element, so a read spans roughly NUM_ELEMS * READ_SPIN cycles (~1 ms) -- wide
# enough for several concurrent swaps to land inside it. The writer spends
# WRITE_SPIN cycles producing its index pair and another WRITE_SPIN with the swap
# half-applied. Correctness under the lock does not depend on these values.
READ_SPIN = 2_000
WRITE_SPIN = 400_000

# Tearing without the lock is a race, so retry a few times before calling it
# absent. In practice the first attempt tears.
UNLOCKED_ATTEMPTS = 5

# Kernel overlap, in microseconds, is measured from CUPTI timestamps in the
# trace. Anything above this counts as real overlap rather than boundary jitter;
# a genuine violation would be hundreds of microseconds wide.
OVERLAP_TOLERANCE_US = 5.0

CPP_SOURCE = """
#include <map>
#include <string>
#include <vector>

void warm_up_kernels();

std::map<std::string, std::vector<double>> sum_and_swap(
    at::Tensor data,
    int64_t num_readers,
    int64_t num_writers,
    int64_t reads_per_reader,
    int64_t writes_per_writer,
    int64_t read_spin,
    int64_t write_spin,
    bool use_lock,
    int64_t seed);
"""

CUDA_SOURCE = """
#include "stream_rwlock.cuh"

#include <atomic>
#include <exception>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

using st::ops::stream_rwlock::LockGuard;
using st::ops::stream_rwlock::StreamRWLock;

#define CUDA_CHECK(expr)                                             \\
  do {                                                               \\
    cudaError_t err__ = (expr);                                      \\
    TORCH_CHECK(                                                     \\
        err__ == cudaSuccess, #expr, " failed: ",                    \\
        cudaGetErrorString(err__));                                  \\
  } while (0)

// Busy-wait on the device for `spin` SM clock cycles.
__device__ __forceinline__ void busy_wait(long long spin) {
  long long start = clock64();
  while (clock64() - start < spin) {
    // spin
  }
}

// The read: sum the buffer with one thread, pausing `spin` cycles after every
// element so the read covers a long window rather than a single instant. A
// mutation landing anywhere inside the window changes the result, so the sum
// doubles as a consistency check. `volatile` keeps the loads from being hoisted
// out of the loop, and the double accumulator makes the sum exact for the
// integer-valued input, so the comparison on the Python side can be exact.
__global__ void windowed_sum_kernel(
    const volatile float* buf,
    int n,
    double* out,
    long long spin) {
  double acc = 0.0;
  for (int i = 0; i < n; ++i) {
    acc += buf[i];
    busy_wait(spin);
  }
  *out = acc;
}

// The writer's input data: the index pair for the next swap, produced on the
// writer's own stream. Deliberately slow -- the mutation has to wait for it, or
// the swap would read a stale or half-written pair.
__global__ void publish_pair_kernel(int* pair, int i, int j, long long spin) {
  busy_wait(spin);
  pair[0] = i;
  pair[1] = j;
}

// The mutation: swap two elements, which keeps the buffer a permutation of
// itself and so keeps its sum invariant. The spin between the two stores widens
// the half-applied window, where buf[j]'s value sits in both slots and the sum
// is short by one element.
__global__ void swap_kernel(volatile float* buf, const int* pair, long long spin) {
  const int i = pair[0];
  const int j = pair[1];
  const float a = buf[i];
  const float b = buf[j];
  buf[i] = b;
  busy_wait(spin);
  buf[j] = a;
}

// Owning handle for a plain (non-caching-allocator) device buffer, so the raw
// streams below are the only thing touching this memory.
template <typename T>
struct DeviceBuffer {
  T* ptr = nullptr;
  int64_t size = 0;

  explicit DeviceBuffer(int64_t n) : size(n) {
    CUDA_CHECK(cudaMalloc(&ptr, static_cast<size_t>(n) * sizeof(T)));
    CUDA_CHECK(cudaMemset(ptr, 0, static_cast<size_t>(n) * sizeof(T)));
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() { cudaFree(ptr); }

  std::vector<T> to_host() const {
    std::vector<T> host(size);
    CUDA_CHECK(cudaMemcpy(
        host.data(), ptr, static_cast<size_t>(size) * sizeof(T),
        cudaMemcpyDeviceToHost));
    return host;
  }
};

} // namespace

// CUDA loads a kernel lazily on its first launch, and that load synchronizes
// with work already queued on the device -- enough to hide a missing fence.
// Warm every kernel up front, and only once per process, so that a profiled run
// contains exactly the kernels its scenario launched.
void warm_up_kernels() {
  static std::once_flag warmed;
  std::call_once(warmed, [] {
    DeviceBuffer<float> buf(2);
    DeviceBuffer<double> out(1);
    DeviceBuffer<int> pair(2);
    publish_pair_kernel<<<1, 1>>>(pair.ptr, 0, 1, 0);
    windowed_sum_kernel<<<1, 1>>>(buf.ptr, 2, out.ptr, 0);
    swap_kernel<<<1, 1>>>(buf.ptr, pair.ptr, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
  });
}

// Scenario: `num_readers` reader threads and `num_writers` writer threads, each
// owning its own CUDA stream, hammer one shared copy of `data`.
//
// Each reader repeatedly sums the buffer under a read lock; each writer
// repeatedly swaps two random elements under a write lock, publishing the index
// pair on its own stream first. Swaps preserve the sum, so every consistent read
// returns the initial sum -- see the module docstring.
//
// With `use_lock == false` the same threads launch straight onto their own
// streams with no cross-stream ordering at all, which is the racy baseline.
//
// Returns {"sums": every sum observed, reader-major; "final": the buffer as it
// ended up}.
std::map<std::string, std::vector<double>> sum_and_swap(
    at::Tensor data,
    int64_t num_readers,
    int64_t num_writers,
    int64_t reads_per_reader,
    int64_t writes_per_writer,
    int64_t read_spin,
    int64_t write_spin,
    bool use_lock,
    int64_t seed) {
  TORCH_CHECK(data.is_cuda(), "data must be a CUDA tensor");
  TORCH_CHECK(data.scalar_type() == at::kFloat, "data must be float32");
  TORCH_CHECK(data.dim() == 1, "data must be 1-D");
  TORCH_CHECK(data.is_contiguous(), "data must be contiguous");
  TORCH_CHECK(data.numel() >= 2, "data needs at least two elements to swap");
  TORCH_CHECK(num_readers >= 0, "num_readers must be non-negative");
  TORCH_CHECK(num_writers >= 0, "num_writers must be non-negative");
  TORCH_CHECK(num_readers + num_writers > 0, "need at least one thread");
  TORCH_CHECK(reads_per_reader >= 0, "reads_per_reader must be non-negative");
  TORCH_CHECK(writes_per_writer >= 0, "writes_per_writer must be non-negative");

  const int n = static_cast<int>(data.numel());
  const int readers = static_cast<int>(num_readers);
  const int writers = static_cast<int>(num_writers);
  const int reads = static_cast<int>(reads_per_reader);
  const int writes = static_cast<int>(writes_per_writer);
  const int total_threads = readers + writers;

  warm_up_kernels();

  StreamRWLock rwlock;
  DeviceBuffer<float> buf(n);
  // One sum slot per read, and one index pair per writer.
  DeviceBuffer<double> sums(std::max(readers * reads, 1));
  DeviceBuffer<int> pairs(std::max(writers, 1) * 2);
  CUDA_CHECK(cudaMemcpy(
      buf.ptr, data.data_ptr<float>(), static_cast<size_t>(n) * sizeof(float),
      cudaMemcpyDeviceToDevice));
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
  std::vector<std::exception_ptr> errors(total_threads);
  std::vector<std::thread> threads;
  threads.reserve(total_threads);

  for (int r = 0; r < readers; ++r) {
    threads.emplace_back([&, r] {
      try {
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));
        wait_for_start();
        for (int round = 0; round < reads; ++round) {
          double* out = sums.ptr + r * reads + round;
          if (use_lock) {
            {
              auto rlock = rwlock.get_rlock(stream);
              LockGuard<StreamRWLock::RLock> guard(rlock);
              windowed_sum_kernel<<<1, 1, 0, guard.get_stream()>>>(
                  buf.ptr, n, out, read_spin);
            }
            // The guard and the RLock are both gone by now, while the sum kernel
            // is still running on the device -- and this thread is parked in the
            // synchronize below, so a writer is free to take the lock during
            // exactly that window. It must still wait for this read, which it
            // can only do via the event r_unlock recorded; a lock that tracked
            // live RLock objects instead would see no reader here and mutate the
            // buffer mid-read.
            CUDA_CHECK(cudaStreamSynchronize(stream));
          } else {
            windowed_sum_kernel<<<1, 1, 0, stream>>>(
                buf.ptr, n, out, read_spin);
            CUDA_CHECK(cudaStreamSynchronize(stream));
          }
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
        // Each writer picks its own swaps, so the mutations differ per thread
        // and per round while staying reproducible for a given seed. The two
        // indices are drawn from opposite halves of the buffer: they are then
        // always distinct and far apart, so a reader scanning in index order is
        // likely to sit between them when a swap lands, which is what makes a
        // missing fence show up as a torn sum instead of a rare fluke. Under the
        // lock the spread changes nothing -- no swap lands mid-read at all.
        std::mt19937 rng(static_cast<uint32_t>(seed * 1000 + w));
        std::uniform_int_distribution<int> pick_low(0, n / 2 - 1);
        std::uniform_int_distribution<int> pick_high(n / 2, n - 1);
        int* pair = pairs.ptr + 2 * w;
        wait_for_start();
        for (int round = 0; round < writes; ++round) {
          const int i = pick_low(rng);
          const int j = pick_high(rng);
          // The input data, on the writer's own stream.
          publish_pair_kernel<<<1, 1, 0, stream>>>(pair, i, j, write_spin);
          if (use_lock) {
            auto wlock = rwlock.get_wlock(stream);
            LockGuard<StreamRWLock::WLock> guard(wlock);
            swap_kernel<<<1, 1, 0, guard.get_stream()>>>(
                buf.ptr, pair, write_spin);
          } else {
            swap_kernel<<<1, 1, 0, stream>>>(buf.ptr, pair, write_spin);
          }
          // w_unlock made this stream wait for the mutation, so this also waits
          // for the swap and the pair slot is free to reuse next round.
          CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
      } catch (...) {
        errors[readers + w] = std::current_exception();
      }
    });
  }

  while (arrived.load(std::memory_order_acquire) < total_threads) {
    std::this_thread::yield();
  }
  go.store(true, std::memory_order_release);
  for (auto& thread : threads) {
    thread.join();
  }
  for (auto& error : errors) {
    if (error) {
      std::rethrow_exception(error);
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  const std::vector<float> final_host = buf.to_host();
  std::map<std::string, std::vector<double>> results;
  results["sums"] = sums.to_host();
  results["sums"].resize(readers * reads);
  results["final"] = std::vector<double>(final_host.begin(), final_host.end());
  return results;
}
"""


@pytest.fixture(scope="module")
def harness():
    """Compile the sum/swap harness against the header (cached by torch)."""
    from torch.utils.cpp_extension import load_inline

    module = load_inline(
        name="stream_rwlock_e2e_harness",
        cpp_sources=CPP_SOURCE,
        cuda_sources=CUDA_SOURCE,
        functions=["warm_up_kernels", "sum_and_swap"],
        extra_include_paths=[CSRC_DIR],
        extra_cuda_cflags=["--expt-relaxed-constexpr", "-O3"],
        verbose=False,
    )
    # Warm up (and load the CUDA module) outside of any profiled region.
    module.warm_up_kernels()
    return module


@pytest.fixture
def data():
    """n distinct integer-valued floats, so sums are exact and swaps observable."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    return (torch.randperm(NUM_ELEMS, device="cuda", generator=generator) + 1).float()


def run_workload(
    harness,
    data: torch.Tensor,
    *,
    use_lock: bool,
    seed: int = 0,
    read_spin: int = READ_SPIN,
    write_spin: int = WRITE_SPIN,
) -> tuple[list[float], list[float]]:
    """Run the 4-reader / 2-writer workload; return the sums read and final buffer."""
    results = harness.sum_and_swap(
        data,
        NUM_READERS,
        NUM_WRITERS,
        READS_PER_READER,
        WRITES_PER_WRITER,
        read_spin,
        write_spin,
        use_lock,
        seed,
    )
    return results["sums"], results["final"]


@pytest.mark.parametrize("read_spin,write_spin", [(READ_SPIN, WRITE_SPIN), (0, 0)])
def test_locked_readers_always_see_the_same_sum(harness, data, read_spin: int, write_spin: int) -> None:
    """Under the lock, every read is a consistent view, so every sum is the invariant."""
    expected = data.double().sum().item()
    sums, final = run_workload(harness, data, use_lock=True, read_spin=read_spin, write_spin=write_spin)

    assert len(sums) == NUM_READERS * READS_PER_READER
    # Swaps preserve the sum, so a consistent read can only return `expected`.
    assert sums == [expected] * len(sums)
    # Writers were mutually exclusive too: no swap was lost or half-applied, so
    # the buffer is still a permutation of the input.
    assert sorted(final) == sorted(data.double().tolist())


def test_unlocked_readers_see_torn_sums(harness, data) -> None:
    """Without the lock the same workload tears: some read is not a consistent view."""
    expected = data.double().sum().item()
    torn: list[float] = []
    for attempt in range(UNLOCKED_ATTEMPTS):
        sums, _ = run_workload(harness, data, use_lock=False, seed=attempt)
        torn = [total for total in sums if total != expected]
        if torn:
            break

    assert torn, (
        f"expected at least one torn read across {UNLOCKED_ATTEMPTS} unlocked runs; "
        "the readers and writers may not have overlapped -- try raising READ_SPIN"
    )


def kernel_events(trace_path: str, name_fragment: str) -> list[dict]:
    """GPU kernel events whose name contains `name_fragment`, as {ts, dur, stream}."""
    with open(trace_path) as trace_file:
        trace = json.load(trace_file)
    return [
        {"ts": event["ts"], "dur": event["dur"], "stream": event.get("args", {}).get("stream")}
        for event in trace["traceEvents"]
        if event.get("ph") == "X" and event.get("cat") == "kernel" and name_fragment in event.get("name", "")
    ]


def overlap_us(first: dict, second: dict) -> float:
    """Microseconds during which both kernels were running (<= 0 if disjoint)."""
    start = max(first["ts"], second["ts"])
    end = min(first["ts"] + first["dur"], second["ts"] + second["dur"])
    return end - start


def profile_workload(harness, data: torch.Tensor, trace_path: str, *, use_lock: bool) -> None:
    """Run the workload under the profiler and export a Chrome trace."""
    with warnings.catch_warnings():
        # torch warns that a profiler without `acc_events` drops events at the end
        # of every cycle. Each run here gets a fresh profiler and is exported
        # immediately, so there is only ever one cycle and nothing to accumulate.
        warnings.filterwarnings("ignore", message="Warning: Profiler clears events")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            run_workload(harness, data, use_lock=use_lock)
    prof.export_chrome_trace(trace_path)
    print(f"\nstream rwlock trace ({'locked' if use_lock else 'unlocked'}): {trace_path}")


def test_profile_shows_shared_reads_and_exclusive_writes(harness, data, tmp_path) -> None:
    """The timeline must show reads overlapping each other but never a write."""
    trace_dir = os.environ.get("STREAM_RWLOCK_TRACE_DIR", str(tmp_path))
    os.makedirs(trace_dir, exist_ok=True)
    locked_trace = os.path.join(trace_dir, "stream_rwlock_locked.json")
    profile_workload(harness, data, locked_trace, use_lock=True)

    reads = kernel_events(locked_trace, "windowed_sum_kernel")
    writes = kernel_events(locked_trace, "swap_kernel")
    assert len(reads) == NUM_READERS * READS_PER_READER
    assert len(writes) == NUM_WRITERS * WRITES_PER_WRITER

    # writer ⟂ reader: no mutation ran while any read was in flight.
    read_write_overlaps = [
        (read, write) for read in reads for write in writes if overlap_us(read, write) > OVERLAP_TOLERANCE_US
    ]
    assert not read_write_overlaps, f"reads and writes overlapped in {locked_trace}: {read_write_overlaps}"

    # reader ∥ reader: reads on different streams did run at the same time, so
    # the exclusion above is not just everything running serially.
    concurrent_reads = [
        (first, second)
        for first, second in combinations(reads, 2)
        if first["stream"] != second["stream"] and overlap_us(first, second) > OVERLAP_TOLERANCE_US
    ]
    assert concurrent_reads, f"no two reads overlapped in {locked_trace}"

    # The exclusion is the lock's doing, not the workload's: the same threads
    # without the lock do put reads and writes on the GPU at the same time.
    unlocked_trace = os.path.join(trace_dir, "stream_rwlock_unlocked.json")
    for _ in range(UNLOCKED_ATTEMPTS):
        profile_workload(harness, data, unlocked_trace, use_lock=False)
        unlocked_overlaps = [
            (read, write)
            for read in kernel_events(unlocked_trace, "windowed_sum_kernel")
            for write in kernel_events(unlocked_trace, "swap_kernel")
            if overlap_us(read, write) > OVERLAP_TOLERANCE_US
        ]
        if unlocked_overlaps:
            break
    assert unlocked_overlaps, f"expected reads and writes to overlap without the lock in {unlocked_trace}"
