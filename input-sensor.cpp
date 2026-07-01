#include "/opt/homebrew/include/boost/lockfree/spsc_queue.hpp"
#include <ApplicationServices/ApplicationServices.h>
#include <CoreFoundation/CFBase.h>
#include <CoreFoundation/CFMachPort.h>
#include <CoreFoundation/CFRunLoop.h>
#include <CoreGraphics/CGEvent.h>
#include <CoreGraphics/CGEventTypes.h>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <chrono>
#include <cstdint>
#include <fcntl.h>
#include <iostream>
#include <string.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>

// Going to use boost.Lockfree for initial IMPL; later versions can utilize DIY
// SPSC queue

// May have to convert CGFloat
struct InputEvent {
  int kind; // 0 for keyboard, 1 for mouse
  int64_t key_code;
  CGEventFlags flags;
  CGFloat x;
  CGFloat y;
  std::chrono::time_point<std::chrono::steady_clock> timestamp;
  // ADD mouse dragged, mouse down, mouse let go
  // ADD current window open
};

static CGEventRef input_sensor(CGEventTapProxy proxy, CGEventType type,
                               CGEventRef event, void *data) {
  auto *q = static_cast<boost::lockfree::spsc_queue<InputEvent> *>(data);
  auto now = std::chrono::steady_clock::now();
  if (type == CGEventType::kCGEventKeyDown) {
    auto key_code = CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode);
    auto flags = CGEventGetFlags(event);
    q->push(InputEvent{0, key_code, flags, 0, 0, now});
  } else if (type == CGEventType::kCGEventMouseMoved) {
    CGPoint loc = CGEventGetLocation(event);
    q->push(InputEvent{1, 0, 0, loc.x, loc.y, now});
  }

  return event;
}

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
  const size_t SHM_SIZE = sizeof(InputEvent);
  // May want to change mode permissions
  int fd = shm_open(shm_name, O_CREAT | O_RDWR, 0666);
  ftruncate(fd, SHM_SIZE);

  // Map shared mem into address space
  InputEvent *shm_ptr = (InputEvent *)mmap(0, SHM_SIZE, PROT_READ | PROT_WRITE,
                                           MAP_SHARED, fd, 0);

  // Consumer thread
  std::thread consumer([&running, &q, shm_ptr]() {
    while (running.load()) {
      InputEvent ev;
      while (q.pop(ev)) {
        // MAY NEED MORE SYNCHRONIZING SAFETY NETS
        // Send information to database
        shm_ptr->key_code = ev.key_code;
        shm_ptr->flags = ev.flags;
        shm_ptr->kind = ev.kind;
        shm_ptr->timestamp = ev.timestamp;
        shm_ptr->x = ev.x;
        shm_ptr->y = ev.y;

        usleep(10000); // Send at 100hz, may change
      }
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  });

  // Event tap
  CGEventMask mask = (1 << kCGEventKeyDown) | (1 << kCGEventMouseMoved);
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
