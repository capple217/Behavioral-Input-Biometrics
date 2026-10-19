// libbibring: the consumer half of the shared-memory protocol, exported with C
// linkage for Python's ctypes.
//
// Why a shim at all: Python has no acquire/release primitives. A plain ctypes
// read of write_idx is an ordinary load; on Apple Silicon (weakly ordered) the
// CPU may then serve slot reads that are older than the index it just saw, i.e.
// Python can read an event the producer has not finished writing. These
// functions compile to LDAR (load-acquire) / STLR (store-release) on arm64 and
// plain MOVs on x86 (where TSO already gives those guarantees).
//
// Python keeps the protocol logic (it decides what to read and when to hand
// slots back); the shim only supplies the ordered loads/stores and the mapping.
#include "bib/input_event.hpp"
#include "bib/shared_ring.hpp"
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace bib;

extern "C"
{

// Fills `out` with layout facts Python cross-checks against its own dtype.
// Returns the number of entries written.
int bib_ring_layout(int64_t *out, int n)
{
    const int64_t v[] = {
        (int64_t)sizeof(InputEvent), (int64_t)offsetof(InputEvent, kind), (int64_t)offsetof(InputEvent, dop),
        (int64_t)offsetof(InputEvent, key_code), (int64_t)offsetof(InputEvent, flags),
        (int64_t)offsetof(InputEvent, mouse_side), (int64_t)offsetof(InputEvent, aux), (int64_t)offsetof(InputEvent, x),
        (int64_t)offsetof(InputEvent, y), (int64_t)offsetof(InputEvent, delta_x), (int64_t)offsetof(InputEvent, delta_y),
        (int64_t)offsetof(InputEvent, timestamp), (int64_t)offsetof(InputEvent, event_time), (int64_t)RING_CAPACITY,
        (int64_t)sizeof(SharedRing), (int64_t)offsetof(SharedRing, slots), (int64_t)RING_MAGIC, (int64_t)RING_VERSION};
    const int m = (int)(sizeof(v) / sizeof(v[0]));
    for (int i = 0; i < m && i < n; i++)
        out[i] = v[i];
    return m;
}

// Maps an existing ring read/write (the consumer must store read_idx).
// Returns NULL and sets *err (errno, or -1 for bad magic/version) on failure.
// Deliberately not Python's multiprocessing.shared_memory: before 3.13 it
// registers attached segments with resource_tracker, which shm_unlink()s them
// when the Python process exits, deleting the producer's segment.
void *bib_ring_attach(const char *name, int *err)
{
    *err = 0;
    int fd = shm_open(name, O_RDWR, 0);
    if (fd < 0)
    {
        *err = errno;
        return nullptr;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || (size_t)st.st_size < sizeof(SharedRing))
    {
        *err = (errno != 0) ? errno : EINVAL;
        close(fd);
        return nullptr;
    }
    void *p = mmap(nullptr, sizeof(SharedRing), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd); // the mapping keeps the object alive
    if (p == MAP_FAILED)
    {
        *err = errno;
        return nullptr;
    }
    SharedRing *r = static_cast<SharedRing *>(p);
    if (r->magic.load(std::memory_order_acquire) != RING_MAGIC || r->version != RING_VERSION ||
        r->event_size != sizeof(InputEvent) || r->capacity != RING_CAPACITY)
    {
        munmap(p, sizeof(SharedRing));
        *err = -1;
        return nullptr;
    }
    return p;
}

void bib_ring_detach(void *ring)
{
    if (ring)
        munmap(ring, sizeof(SharedRing));
}

// ACQUIRE: every slot below the returned index is fully written and visible.
uint64_t bib_ring_load_write_idx(void *ring)
{
    return static_cast<SharedRing *>(ring)->write_idx.load(std::memory_order_acquire);
}

uint64_t bib_ring_load_read_idx(void *ring)
{
    return static_cast<SharedRing *>(ring)->read_idx.load(std::memory_order_relaxed);
}

// RELEASE: our copies of slots below `idx` are complete; the producer may reuse them.
void bib_ring_store_read_idx(void *ring, uint64_t idx)
{
    static_cast<SharedRing *>(ring)->read_idx.store(idx, std::memory_order_release);
}

void bib_ring_counters(void *ring, uint64_t *dropped, uint64_t *dropped_tap, uint32_t *alive)
{
    SharedRing *r = static_cast<SharedRing *>(ring);
    *dropped = r->dropped.load(std::memory_order_relaxed);
    *dropped_tap = r->dropped_tap.load(std::memory_order_relaxed);
    *alive = r->producer_alive.load(std::memory_order_acquire);
}

void *bib_ring_slots(void *ring)
{
    return static_cast<SharedRing *>(ring)->slots;
}

} // extern "C"
