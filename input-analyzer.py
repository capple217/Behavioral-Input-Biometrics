import ctypes
import time
from multiprocessing import shared_memory

import atomics


class InputEvent(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_int),
        ("dop", ctypes.c_int),
        ("key_code", ctypes.c_int64),
        ("flags", ctypes.c_uint64),
        ("mouse_side", ctypes.c_int),
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
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

    # Continously read data from buffer
    while True:
        SharedRing = SharedRing.from_buffer(shm.buf)
        # print(f"Kind {SharedRing.slots[1].kind}, key_code {input_events.key_code}")
        # print(f"x {input_events.x}, y {input_events.y}")
        # print(f"deltaX {input_events.delta_x}, deltaY {input_events.delta_y}")
        # print(f"Time {input_events.timestamp}")
        # time.sleep(0.01)  # 100Hz

except KeyboardInterrupt:
    print("Stopped reading.")

finally:
    # Cleanup
    shm.close()
