import ctypes
import mmap
import time
from multiprocessing import shared_memory


class InputEvent(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_int),
        ("dop", ctypes.c_int),
        ("key_code", ctypes.c_int64),
        ("flags", ctypes.c_uint64),
        ("mouse_side", ctypes.c_int),
        ("x", ctypes.c_double),
        ("y", ctypes.c_double),
        ("delta_x", ctypes.c_int64),
        ("delta_y", ctypes.c_int64),
        ("timestamp", ctypes.c_int64),
    ]


CAPACITY = 50


class SharedRing(ctypes.Structure):
    _fields_ = [
        ("write_idx", ctypes.c_uint64),
        ("read_idx", ctypes.c_uint64),
        ("dropped", ctypes.c_uint64),
        ("slots", InputEvent * CAPACITY),
    ]


shm_name = "BIBSharedMem"

try:
    shm = shared_memory.SharedMemory(name=shm_name)
    ring = SharedRing.from_buffer(shm.buf)
    # Continously read data from buffer
    while True:
        w = ring.write_idx
        r = ring.read_idx
        if r == w:
            time.sleep(0.001)
            continue
        ev = ring.slots[r % CAPACITY]
        print(f"x {ev.x} y{ev.y}")
        ring.read_idx = r + 1

except KeyboardInterrupt:
    print("Stopped reading.")

finally:
    # Cleanup
    del ring
    shm.close()
