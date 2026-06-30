#include "/opt/homebrew/include/boost/lockfree/spsc_queue.hpp"
#include <ApplicationServices/ApplicationServices.h>
#include <CoreFoundation/CFBase.h>
#include <CoreFoundation/CFMachPort.h>
#include <CoreFoundation/CFRunLoop.h>
#include <CoreGraphics/CGEvent.h>
#include <CoreGraphics/CGEventTypes.h>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <string.h>
#include <thread>

// Going to use boost.Lockfree for initial IMPL; later versions can utilize DIY
// SPSC queue

enum class EventKind { Key, Mouse };

// May have to convert CGFloat
struct InputEvent {
  EventKind kind;
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
    q->push(InputEvent{EventKind::Key, key_code, flags, 0, 0, now});
  } else if (type == CGEventType::kCGEventMouseMoved) {
    CGPoint loc = CGEventGetLocation(event);
    q->push(InputEvent{EventKind::Mouse, 0, 0, loc.x, loc.y, now});
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

  // Create shared space
  constexpr std::size_t capacity = 128;
  boost::lockfree::spsc_queue<InputEvent> q(capacity);
  std::atomic<bool> running{true};

  // Consumer thread
  std::thread consumer([&running, &q]() {
    while (running.load()) {
      InputEvent ev;
      while (q.pop(ev)) {
        // Process information
        // Send information to database
        if (ev.key_code != 0)
          std::cout << ev.key_code << std::endl;
      }
      std::this_thread::sleep_for(std::chrono::seconds(3));
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
  return 0;
}
