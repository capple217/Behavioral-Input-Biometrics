#include "/opt/homebrew/include/boost/lockfree/spsc_queue.hpp"
#include <ApplicationServices/ApplicationServices.h>
#include <CoreFoundation/CFBase.h>
#include <CoreFoundation/CFMachPort.h>
#include <CoreFoundation/CFRunLoop.h>
#include <CoreGraphics/CGEvent.h>
#include <CoreGraphics/CGEventTypes.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fcntl.h>
#include <iostream>
#include <string.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>

constexpr int CAPACITY = 50;

// Going to use boost.Lockfree for initial IMPL; later versions can utilize DIY
// SPSC queue

// May have to convert CGFloat
struct InputEvent {
  int kind; // 0 for keyboard, 1 for mouse
  int dop;  // (down or up) 0 for down, 1 for up, 2 for dragged, 3 for neither
  int64_t key_code;
  CGEventFlags flags;
  int mouse_side; // 0 for neither, 1 for left, 2 for right, 3 for scrollwheel
  CGFloat x;
  CGFloat y;
  int64_t delta_x;
  int64_t delta_y;
  std::chrono::time_point<std::chrono::steady_clock> timestamp;
  // ADD current window open
};

static CGEventRef input_sensor(CGEventTapProxy proxy, CGEventType type,
                               CGEventRef event, void *data) {
  auto *q = static_cast<boost::lockfree::spsc_queue<InputEvent> *>(data);
  auto now = std::chrono::steady_clock::now();
  if (type == CGEventType::kCGEventKeyDown) {
    auto key_code = CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode);
    auto flags = CGEventGetFlags(event);
    q->push(InputEvent{0, 0, key_code, flags, 0, 0, 0, 0, 0, now});
  } else if (type == CGEventType::kCGEventKeyUp) {
    auto key_code = CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode);
    auto flags = CGEventGetFlags(event);
    q->push(InputEvent{0, 1, key_code, flags, 0, 0, 0, 0, 0, now});
  } else if (type == CGEventType::kCGEventMouseMoved) {
    CGPoint loc = CGEventGetLocation(event);
    int64_t deltaX = CGEventGetIntegerValueField(event, kCGMouseEventDeltaX);
    int64_t deltaY = CGEventGetIntegerValueField(event, kCGMouseEventDeltaY);
    q->push(InputEvent{1, 3, 0, 0, 0, loc.x, loc.y, deltaX, deltaY, now});
  } else if (type == CGEventType::kCGEventLeftMouseDown) {
    CGPoint loc = CGEventGetLocation(event);
    q->push(InputEvent{1, 0, 0, 0, 1, loc.x, loc.y, 0, 0, now});
  } else if (type == CGEventType::kCGEventLeftMouseUp) {
    CGPoint loc = CGEventGetLocation(event);
    q->push(InputEvent{1, 1, 0, 0, 1, loc.x, loc.y, 0, 0, now});
  } else if (type == CGEventType::kCGEventLeftMouseDragged) {
    CGPoint loc = CGEventGetLocation(event);
    int64_t deltaX = CGEventGetIntegerValueField(event, kCGMouseEventDeltaX);
    int64_t deltaY = CGEventGetIntegerValueField(event, kCGMouseEventDeltaY);
    q->push(InputEvent{1, 2, 0, 0, 1, loc.x, loc.y, deltaX, deltaY, now});
  } else if (type == CGEventType::kCGEventRightMouseDown) {
    CGPoint loc = CGEventGetLocation(event);
    q->push(InputEvent{1, 0, 0, 0, 2, loc.x, loc.y, 0, 0, now});
  } else if (type == CGEventType::kCGEventRightMouseUp) {
    CGPoint loc = CGEventGetLocation(event);
    q->push(InputEvent{1, 1, 0, 0, 2, loc.x, loc.y, 0, 0, now});
  } else if (type == CGEventType::kCGEventRightMouseDragged) {
    CGPoint loc = CGEventGetLocation(event);
    int64_t deltaX = CGEventGetIntegerValueField(event, kCGMouseEventDeltaX);
    int64_t deltaY = CGEventGetIntegerValueField(event, kCGMouseEventDeltaY);
    q->push(InputEvent{1, 2, 0, 0, 2, loc.x, loc.y, deltaX, deltaY, now});
  }

  return event;
}

struct SharedRing {
  std::atomic<uint64_t> write_idx;
  std::atomic<uint64_t> read_idx;
  std::atomic<int64_t> dropped;
  InputEvent slots[CAPACITY];
};

int main() {
  // Check accessability permissions
  /*
  if (!AXIsProcessTrustedWithOptions())
  {
    std::cout << "Need Permission To Operate" << std::endl;
    return 1;
  }
  */

  // Create queue space
  constexpr std::size_t capacity = 128;
  boost::lockfree::spsc_queue<InputEvent> q(capacity);
  std::atomic<bool> running{true};

  // Create shared memory object
  const char *shm_name = "/BIBSharedMem";
  const size_t SHM_SIZE = sizeof(SharedRing);
  // May want to change mode permissions
  int fd = shm_open(shm_name, O_CREAT | O_RDWR, 0666);
  ftruncate(fd, SHM_SIZE);

  // Map shared mem into address space
  SharedRing *shm_ptr = (SharedRing *)mmap(0, SHM_SIZE, PROT_READ | PROT_WRITE,
                                           MAP_SHARED, fd, 0);
  shm_ptr->read_idx = 0;
  shm_ptr->write_idx = 0;
  shm_ptr->dropped = 0;

  // Consumer thread
  std::thread consumer([&running, &q, shm_ptr]() {
    while (running.load()) {
      InputEvent ev;
      while (q.pop(ev)) {
        // Send information to database at current write_idx
        uint64_t r = shm_ptr->read_idx.load(std::memory_order_acquire);
        uint64_t w = shm_ptr->write_idx.load(std::memory_order_relaxed);
        if (w - r >= CAPACITY) {
          // write_idx is too far ahead, so we skip all incoming events until we
          // have space and update drop counter for data to know
          shm_ptr->dropped.fetch_add(1, std::memory_order_relaxed);
          continue;
        }
        shm_ptr->slots[w % CAPACITY].key_code = ev.key_code;
        shm_ptr->slots[w % CAPACITY].flags = ev.flags;
        shm_ptr->slots[w % CAPACITY].kind = ev.kind;
        shm_ptr->slots[w % CAPACITY].timestamp = ev.timestamp;
        shm_ptr->slots[w % CAPACITY].x = ev.x;
        shm_ptr->slots[w % CAPACITY].y = ev.y;
        shm_ptr->slots[w % CAPACITY].dop = ev.dop;
        shm_ptr->slots[w % CAPACITY].mouse_side = ev.mouse_side;
        shm_ptr->slots[w % CAPACITY].delta_x = ev.delta_x;
        shm_ptr->slots[w % CAPACITY].delta_y = ev.delta_y;

        // Increment write_idx
        shm_ptr->write_idx.store(w + 1, std::memory_order_release);
      }
      usleep(10000); // Send at 100hz, may change
    }
  });

  // Event tap
  CGEventMask mask =
      (1 << kCGEventKeyDown) | (1 << kCGEventKeyUp) |
      (1 << kCGEventMouseMoved) | (1 << kCGEventLeftMouseDown) |
      (1 << kCGEventLeftMouseUp) | (1 << kCGEventLeftMouseDragged) |
      (1 << kCGEventRightMouseDown) | (1 << kCGEventRightMouseUp) |
      (1 << kCGEventRightMouseDragged);
  CFMachPortRef handle =
      CGEventTapCreate(kCGHIDEventTap, kCGTailAppendEventTap,
                       kCGEventTapOptionListenOnly, mask, input_sensor, &q);

  CFRunLoopSourceRef source =
      CFMachPortCreateRunLoopSource(kCFAllocatorDefault, handle, 0);
  CFRunLoopAddSource(CFRunLoopGetMain(), source, kCFRunLoopCommonModes);

  // Indefintely running the loop
  CFRunLoopRun();

  // If loop ever exists, leave
  running.store(false);
  consumer.join();
  CFRelease(source);
  CFRelease(handle);
  munmap(shm_ptr, SHM_SIZE);
  close(fd);
  shm_unlink(shm_name);

  return 0;
}
