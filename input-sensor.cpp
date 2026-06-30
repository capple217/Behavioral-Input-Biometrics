#include "/opt/homebrew/include/boost/lockfree/spsc_queue.hpp"
#include <ApplicationServices/ApplicationServices.h>
#include <CoreGraphics/CGEvent.h>
#include <CoreGraphics/CGEventTypes.h>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <string.h>
#include <thread>

// Initial IMPL will only focus on keys being pressed down, not on the
// difference in time between being pressed up HOWEVER, that is an important
// characteristic to be added

// Going to use boost.Lockfree for initial IMPL; later versions can utilize DIY
// SPSC queue

// std::string is not acceptable for boost.lockfree.spsc_queue so instead using
// char struct
enum class EventKind { Key, Mouse };

// May have to convert CGFloat
struct InputEvent {
  EventKind kind;
  int64_t key_code;
  CGEventFlags flags;
  CGFloat x;
  CGFloat y;
  std::chrono::time_point<std::chrono::steady_clock> timestamp;
  // To add mouse dragged
  // Similarly, left/right mouse down (aka pressed)
  // Consequently, also when let go
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
  // std::thread consumer([])

  // Event tap
  CGEventMask mask = (1 << kCGEventKeyDown) | (1 << kCGEventMouseMoved);
  // CFMachPortRef handle = CGEventTapCreate(kCGHIDEventTap, kCGTail)
  return 0;
}
