import ctypes
import time
from multiprocessing import shared_memory


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


shm_name = "BIBSharedMem"

try:
    shm = shared_memory.SharedMemory(name=shm_name)

    # Continously read data from buffer
    while True:
        input_events = InputEvent.from_buffer(shm.buf)
        print(f"Kind {input_events.kind}, key_code {input_events.key_code}")
        print(f"x {input_events.x}, y {input_events.y}")
        print(f"deltaX {input_events.delta_x}, deltaY {input_events.delta_y}")
        print(f"Time {input_events.timestamp}")
        time.sleep(0.01)  # 100Hz

except KeyboardInterrupt:
    print("Stopped reading.")

finally:
    # Cleanup
    shm.close()
