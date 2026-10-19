#pragma once
// Hand-written bounded single-producer / single-consumer queue.
// Replaces boost::lockfree::spsc_queue between the event-tap callback (producer,
// main run-loop thread) and the bridge thread (consumer) that publishes into
// shared memory.
//
// Design, and why each choice:
//   * Fixed power-of-two capacity N as a template parameter: `i & (N-1)` instead
//     of `%`, and the storage lives inline (no heap allocation on the tap path).
//   * Monotonic 64-bit indices (never wrapped back to 0): full = tail-head == N,
//     empty = tail == head. No sacrificed slot, no ABA question: at 1 event/ns
//     a 64-bit counter wraps in ~584 years.
//   * tail_ is written only by the producer, head_ only by the consumer. Each
//     side publishes its index with a release store and reads the other side's
//     with an acquire load. The release on tail_ orders the slot write before
//     the index bump; the consumer's acquire on tail_ makes the slot contents
//     visible before it reads them. Symmetrically, head_ release/acquire hands
//     the slot back to the producer only after the consumer finished copying.
//   * Cached copy of the *other* side's index (Rigtorp / Vyukov trick): the
//     producer only re-reads head_ (a cache line the consumer owns) when its
//     stale copy says the queue looks full, and vice versa. In the common case
//     each operation touches only lines its own thread owns.
//   * Every independently-written field on its own cache line, so the
//     producer's stores to tail_ don't invalidate the consumer's head_ line
//     (false sharing). Apple M-series cores use 128-byte lines, x86 uses 64.
//   * try_push never blocks and reports failure: the tap callback must return
//     fast, so on a full queue the caller counts a drop (newest-drop policy).
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace bib
{

#if defined(__APPLE__) && defined(__aarch64__)
constexpr std::size_t CACHE_LINE = 128;
#else
constexpr std::size_t CACHE_LINE = 64;
#endif

template <typename T, std::size_t N>
class SpscQueue
{
    static_assert(N >= 2 && (N & (N - 1)) == 0, "capacity must be a power of two");
    static_assert(std::is_trivially_copyable<T>::value, "T is copied by value across threads");
    static_assert(std::atomic<uint64_t>::is_always_lock_free, "need lock-free 64-bit atomics");
public:
    SpscQueue() = default;
    SpscQueue(const SpscQueue &) = delete;
    SpscQueue &operator=(const SpscQueue &) = delete;
    // Producer thread only.
    bool try_push(const T &v)
    {
        const uint64_t t = tail_.load(std::memory_order_relaxed); // we are the only writer
        if (t - head_cache_ == N)
        {
            head_cache_ = head_.load(std::memory_order_acquire);
            if (t - head_cache_ == N)
                return false;
        }
        buf_[t & (N - 1)] = v;
        tail_.store(t + 1, std::memory_order_release);
        return true;
    }
    // Consumer thread only.
    bool try_pop(T &out)
    {
        const uint64_t h = head_.load(std::memory_order_relaxed); // we are the only writer
        if (h == tail_cache_)
        {
            tail_cache_ = tail_.load(std::memory_order_acquire);
            if (h == tail_cache_)
                return false;
        }
        out = buf_[h & (N - 1)];
        head_.store(h + 1, std::memory_order_release);
        return true;
    }
    // Approximate; exact only when called from one of the two threads with the other idle.
    std::size_t size_approx() const
    {
        return static_cast<std::size_t>(tail_.load(std::memory_order_acquire) - head_.load(std::memory_order_acquire));
    }
    static constexpr std::size_t capacity() { return N; }
private:
    alignas(CACHE_LINE) std::atomic<uint64_t> tail_{0}; // producer-owned
    alignas(CACHE_LINE) uint64_t head_cache_ = 0;       // producer-private copy of head_
    alignas(CACHE_LINE) std::atomic<uint64_t> head_{0}; // consumer-owned
    alignas(CACHE_LINE) uint64_t tail_cache_ = 0;       // consumer-private copy of tail_
    alignas(CACHE_LINE) T buf_[N];
};

} // namespace bib
