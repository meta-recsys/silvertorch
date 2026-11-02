#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <vector>


namespace st::ops::stream_rwlock {

#define _CUDA_CHECK(expr)                                                    \
do {                                                                         \
  const cudaError_t error_ = (expr);                                         \
  if (error_ != cudaSuccess) {                                               \
    std::fprintf(stderr, "CUDA error %s (%s) at %s:%d in %s\n",              \
                  cudaGetErrorName(error_), cudaGetErrorString(error_),      \
                  __FILE__, __LINE__, #expr);                                \
    std::abort();                                                            \
  }                                                                          \
} while (0)

// RAII guard that locks on construction and unlocks on destruction. Works with
// any lock exposing lock()/unlock()/get_stream() (e.g. RLock, WLock). Holds
// the lock by reference: exactly one lock()/unlock() pair per guard.
template <typename LockType> struct LockGuard {
  LockType &lock_;
  explicit LockGuard(LockType &lock) : lock_(lock) { lock_.lock(); }
  ~LockGuard() { lock_.unlock(); }

  LockGuard(const LockGuard &) = delete;
  LockGuard &operator=(const LockGuard &) = delete;

  // Convenience passthrough to the stream callers should launch on.
  cudaStream_t get_stream() { return lock_.get_stream(); }
};

// A stream based read-write lock that can be used to synchronize access to
// shared resources between streams. Two roles: reader stream and writer stream.
// Reader stream:
//  - Acquire the read lock on the shared resource before launching kernels.
//  - Release the read lock after the read kernels have been launched.
//  - Reader stream asynchronously waits for the writer stream to finish if any,
//    but does not block other reader streams.
// Writer stream:
//  - Acquire the write lock on the shared resource before launching kernels.
//  - Release the write lock after the write kernels have been launched.
//  - Writer stream asynchronously waits for reader streams and other writer
//    streams to finish.
//
// The writer discovers readers through read-done events, not a registry of
// live RLock objects: r_unlock records an event on the read stream (capturing
// the kernels enqueued by that section), and the next w_lock waits on the
// recorded events. An RLock can therefore be destroyed at launch time — while
// its kernels are still running on the device — without any writer losing the
// dependency, and without the caller ever synchronizing the CPU with the GPU.
//
// Usage:
// 1. Reader stream:
//    RLock rlock = rwlock.get_rlock(stream);
//    LockGuard guard(rlock);
//    auto read_stream = guard.get_stream();
//    launch_reading_kernels_on(read_stream);
// 2. Writer stream:
//    WLock wlock = rwlock.get_wlock(stream);
//    LockGuard guard(wlock);
//    auto mutation_stream = guard.get_stream();
//    launch_writing_kernels_on(mutation_stream);
class StreamRWLock {

private:
  // Serializes the host-side "set up sync + launch" critical section.
  std::mutex launch_mutex_;

  // read streams with their done events since the previous write.
  std::unordered_map<cudaStream_t, cudaEvent_t> read_streams_;

  // Pool of events for reuse.
  std::vector<cudaEvent_t> event_pool_;

  // Marks completion of the latest write; readers and the next writer wait on it.
  cudaEvent_t mutation_done_;

  // Private stream on which mutation kernels run.
  cudaStream_t mutation_stream_;

  // Helper functions.
  void record_mutation_done() {
    _CUDA_CHECK(cudaEventRecord(this->mutation_done_, this->mutation_stream_));
  }

  void wait_for_mutation_done(cudaStream_t stream) {
    _CUDA_CHECK(cudaStreamWaitEvent(stream, this->mutation_done_));
  }

  cudaEvent_t acquire_event() {
    if (!this->event_pool_.empty()) {
      cudaEvent_t event = this->event_pool_.back();
      this->event_pool_.pop_back();
      return event;
    }
    cudaEvent_t event;
    _CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    return event;
  }

  void release_event(cudaEvent_t event) {
    this->event_pool_.push_back(event);
  }

  // Mutation stream waits for every read stream since the previous write.
  void wait_for_all_read_streams() {
    for (const auto &[read_stream, done_event] : this->read_streams_) {
      _CUDA_CHECK(cudaStreamWaitEvent(this->mutation_stream_, done_event));
      this->release_event(done_event);
    }
    this->read_streams_.clear();
  }

  // Record read-done on the read stream.
  void record_read_stream(cudaStream_t read_stream) {
    auto it = this->read_streams_.find(read_stream);
    cudaEvent_t done_event = (it != this->read_streams_.end()) ? it->second : this->acquire_event();
    _CUDA_CHECK(cudaEventRecord(done_event, read_stream));
    this->read_streams_[read_stream] = done_event;
  }

  // The waiter stream waits for the waitee stream to finish.
  void wait_for_stream(cudaStream_t waiter, cudaStream_t waitee) {
    if (waiter != waitee) {
      // Place an event on the waitee and let the waiter wait for it asynchronously.
      cudaEvent_t event = this->acquire_event();
      _CUDA_CHECK(cudaEventRecord(event, waitee));
      _CUDA_CHECK(cudaStreamWaitEvent(waiter, event));
      this->release_event(event);
    }
  }

public:
  StreamRWLock() {
    _CUDA_CHECK(cudaEventCreateWithFlags(&mutation_done_, cudaEventDisableTiming));
    _CUDA_CHECK(cudaStreamCreate(&mutation_stream_));
  }

  ~StreamRWLock() {
    for (const auto &[read_stream, done_event] : read_streams_) {
      _CUDA_CHECK(cudaEventDestroy(done_event));
    }
    for (const auto &event : event_pool_) {
      _CUDA_CHECK(cudaEventDestroy(event));
    }
    _CUDA_CHECK(cudaEventDestroy(mutation_done_));
    _CUDA_CHECK(cudaStreamDestroy(mutation_stream_));
  }

  StreamRWLock(const StreamRWLock &) = delete;
  StreamRWLock &operator=(const StreamRWLock &) = delete;

  struct RLock {
    StreamRWLock *rwlock_;
    cudaStream_t read_stream_;

    RLock(StreamRWLock *rwlock, cudaStream_t read_stream)
        : rwlock_(rwlock), read_stream_(read_stream) {}
    ~RLock() = default;

    void lock() { rwlock_->r_lock(read_stream_); }

    void unlock() { rwlock_->r_unlock(read_stream_); }

    cudaStream_t get_stream() { return read_stream_; }
  };

  struct WLock {
    StreamRWLock *rwlock_;
    // Caller's stream, where the input data for the update comes from.
    cudaStream_t write_stream_;

    WLock(StreamRWLock *rwlock, cudaStream_t write_stream)
        : rwlock_(rwlock), write_stream_(write_stream) {}
    ~WLock() = default;

    void lock() { rwlock_->w_lock(write_stream_); }

    void unlock() { rwlock_->w_unlock(write_stream_); }

    cudaStream_t get_stream() { return rwlock_->mutation_stream_; }
  };

  RLock get_rlock(cudaStream_t read_stream) { return {this, read_stream}; }

  WLock get_wlock(cudaStream_t write_stream) { return {this, write_stream}; }

  // Reader lock/unlock logic, executed on behalf of RLock.
  void r_lock(cudaStream_t read_stream) {
    // Hold the launch mutex to prevent concurrent kernel launches.
    this->launch_mutex_.lock();
    // Wait for any ongoing mutation to finish asynchronously.
    this->wait_for_mutation_done(read_stream);
    // From now on, the read stream is free to launch reading kernels.
  }

  void r_unlock(cudaStream_t read_stream) {
    // Leave a device-side marker for the next writer to wait on.
    this->record_read_stream(read_stream);
    this->launch_mutex_.unlock();
  }

  // Writer lock/unlock logic, executed on behalf of WLock.
  void w_lock(cudaStream_t write_stream) {
    // Hold the launch mutex to prevent concurrent kernel launches.
    this->launch_mutex_.lock();

    // Mutation stream waits for all read streams since the previous write to finish asynchronously.
    this->wait_for_all_read_streams();

    // Mutation stream waits for the previous mutation to finish asynchronously.
    this->wait_for_mutation_done(this->mutation_stream_);

    // Mutation stream waits for the write stream so the input data is ready.
    this->wait_for_stream(this->mutation_stream_, write_stream);

    // From now on, the mutation stream is free to launch mutation kernels.
  }

  void w_unlock(cudaStream_t write_stream) {
    // Record the write-done event on the mutation stream.
    this->record_mutation_done();

    // Write stream waits for the mutation done so the caller sees the result.
    this->wait_for_mutation_done(write_stream);

    this->launch_mutex_.unlock();
  }
};
#undef _CUDA_CHECK

} // namespace st::ops::stream_rwlock
