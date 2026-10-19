#pragma once
// SharedRing: the SPSC ring that lives in POSIX shared memory and carries
// InputEvents from the C++ producer process to the Python consumer process.
//
// Protocol (same algorithm as SpscQueue, different address space):
//   producer (bridge thread, C++):
//       w = write_idx (relaxed, sole writer); r = read_idx (ACQUIRE)
//       if w - r == CAPACITY  -> drop newest, ++dropped     (never overwrite unread data)
//       slots[w & mask] = ev; write_idx = w + 1 (RELEASE)
//   consumer (Python via libbibring):
//       w = write_idx (ACQUIRE)  -> slots [r, w) are fully written and visible
//       copy slots [r, w)
//       read_idx = w (RELEASE)   -> copies finished before the producer may reuse those slots
//
// Why newest-drop instead of overwrite-oldest: overwriting would let the
// producer write a slot while Python is mid-copy (a torn event) unless we add
// per-slot sequence numbers. Dropping the newest keeps every delivered event
// intact and makes loss *observable* through the counters, which the analysis
// side uses to discard windows that straddle a gap.
//
// Layout is fixed-alignment (128) on every platform so offsets don't depend on
// the compiler's idea of a cache line; Python re-checks them at attach time.
#include "bib/input_event.hpp"
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <new>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace bib
{

constexpr uint32_t RING_MAGIC = 0x31424942; // "BIB1" little-endian
constexpr uint32_t RING_VERSION = 2;
constexpr uint64_t RING_CAPACITY = 4096;    // ~41 s of 100 Hz mouse movement if Python stalls
constexpr const char *RING_NAME = "/BIBSharedMem"; // macOS limits shm names to 31 chars
static_assert((RING_CAPACITY & (RING_CAPACITY - 1)) == 0, "capacity must be a power of two");
static_assert(std::atomic<uint64_t>::is_always_lock_free, "cross-process atomics must be lock-free");
static_assert(std::atomic<uint32_t>::is_always_lock_free, "cross-process atomics must be lock-free");

typedef struct SharedRing
{
    // header: written once before `magic` is published, immutable afterwards
    std::atomic<uint32_t> magic;
    uint32_t version;
    uint32_t capacity;
    uint32_t event_size;
    std::atomic<uint32_t> producer_alive;      // 0 after a clean producer shutdown
    alignas(128) std::atomic<uint64_t> write_idx; // producer-owned
    alignas(128) std::atomic<uint64_t> read_idx;  // consumer-owned
    alignas(128) std::atomic<uint64_t> dropped;   // lost at the shm boundary (Python too slow)
    std::atomic<uint64_t> dropped_tap;            // lost at the in-process queue (bridge too slow)
    alignas(128) InputEvent slots[RING_CAPACITY];
} SharedRing;

static_assert(std::is_standard_layout<SharedRing>::value, "SharedRing must have C layout");
static_assert(offsetof(SharedRing, write_idx) == 128, "layout");
static_assert(offsetof(SharedRing, read_idx) == 256, "layout");
static_assert(offsetof(SharedRing, dropped) == 384, "layout");
static_assert(offsetof(SharedRing, dropped_tap) == 392, "layout");
static_assert(offsetof(SharedRing, slots) == 512, "layout");

// Producer side: publish one event. Returns false (and counts it) on a full ring.
static inline bool ring_publish(SharedRing *r, const InputEvent &ev)
{
    const uint64_t w = r->write_idx.load(std::memory_order_relaxed);
    const uint64_t rd = r->read_idx.load(std::memory_order_acquire);
    if (w - rd >= RING_CAPACITY)
    {
        r->dropped.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    r->slots[w & (RING_CAPACITY - 1)] = ev;
    r->write_idx.store(w + 1, std::memory_order_release);
    return true;
}

// Producer side: create a fresh segment. Any stale segment from a crashed run is
// unlinked first: on macOS ftruncate() only succeeds once per shm object, so
// re-opening an old one and resizing it fails with EINVAL.
// Mode 0600: this segment carries every keystroke you type; other local users
// must not be able to map it (the original 0666 allowed that).
static inline SharedRing *ring_create(const char *name, int *out_fd)
{
    shm_unlink(name);
    int fd = shm_open(name, O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd < 0)
        return nullptr;
    if (ftruncate(fd, sizeof(SharedRing)) != 0)
    {
        int e = errno;
        close(fd);
        shm_unlink(name);
        errno = e;
        return nullptr;
    }
    void *p = mmap(nullptr, sizeof(SharedRing), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED)
    {
        int e = errno;
        close(fd);
        shm_unlink(name);
        errno = e;
        return nullptr;
    }
    SharedRing *r = new (p) SharedRing(); // value-init: atomics and slots start at zero
    r->version = RING_VERSION;
    r->capacity = static_cast<uint32_t>(RING_CAPACITY);
    r->event_size = static_cast<uint32_t>(sizeof(InputEvent));
    r->producer_alive.store(1, std::memory_order_relaxed);
    r->magic.store(RING_MAGIC, std::memory_order_release); // publishes the header
    *out_fd = fd;
    return r;
}

static inline void ring_destroy(SharedRing *r, int fd, const char *name)
{
    if (r)
    {
        r->producer_alive.store(0, std::memory_order_release);
        munmap(r, sizeof(SharedRing));
    }
    if (fd >= 0)
        close(fd);
    shm_unlink(name);
}

} // namespace bib
