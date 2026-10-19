#pragma once
// Bridge: drains the in-process SpscQueue into the shared-memory ring.
//
// Why two queues instead of having the tap callback write the shm ring
// directly: the callback runs on the main run loop and macOS disables a tap
// whose callback stalls, so the callback does the minimum (one copy into
// memory this process owns). Everything that touches the cross-process mapping
// and the drop accounting happens on this thread, and a second sink (e.g. a
// disk recorder in C++) can be added here without touching the callback.
// The cost is one extra copy (80 B) per event and up to `idle_sleep` latency.
#include "bib/input_event.hpp"
#include "bib/shared_ring.hpp"
#include "bib/spsc_queue.hpp"
#include <atomic>
#include <chrono>
#include <thread>

namespace bib
{

constexpr std::size_t TAP_QUEUE_CAPACITY = 8192;
typedef SpscQueue<InputEvent, TAP_QUEUE_CAPACITY> TapQueue;

class Bridge
{
public:
    explicit Bridge(SharedRing *ring, std::chrono::microseconds idle_sleep = std::chrono::microseconds(2000))
        : ring_(ring), idle_sleep_(idle_sleep)
    {
    }
    ~Bridge() { stop(); }
    void start()
    {
        running_.store(true, std::memory_order_relaxed);
        thread_ = std::thread([this]() { run(); });
    }
    // Stops after draining whatever the producer already pushed.
    void stop()
    {
        if (!thread_.joinable())
            return;
        running_.store(false, std::memory_order_release);
        thread_.join();
    }
    // Called from the producer (tap callback) thread only.
    void submit(const InputEvent &ev)
    {
        if (!queue_.try_push(ev))
            tap_dropped_.fetch_add(1, std::memory_order_relaxed);
    }
    uint64_t published() const { return published_.load(std::memory_order_relaxed); }
private:
    void drain()
    {
        InputEvent ev;
        while (queue_.try_pop(ev))
        {
            if (ring_publish(ring_, ev))
                published_.fetch_add(1, std::memory_order_relaxed);
        }
        ring_->dropped_tap.store(tap_dropped_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    }
    void run()
    {
        while (running_.load(std::memory_order_acquire))
        {
            drain();
            std::this_thread::sleep_for(idle_sleep_);
        }
        drain(); // final flush
    }
    SharedRing *ring_;
    std::chrono::microseconds idle_sleep_;
    TapQueue queue_;
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> tap_dropped_{0};
    std::atomic<uint64_t> published_{0};
    std::thread thread_;
};

} // namespace bib
