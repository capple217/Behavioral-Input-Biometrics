// input_sensor: macOS producer. Taps HID input with CGEventTapCreate, records
// raw integers + timestamps (no interpretation), and hands them to the Bridge,
// which publishes them into the shared-memory ring read by Python.
//
// Build: cmake -B build && cmake --build build   (see README)
// Permission: System Settings > Privacy & Security > Input Monitoring must list
// the binary (or the terminal it runs from).
#include "bib/bridge.hpp"
#include "bib/input_event.hpp"
#include "bib/shared_ring.hpp"
#include <ApplicationServices/ApplicationServices.h>
#include <dispatch/dispatch.h>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>

using namespace bib;

// Passed through CGEventTapCreate's void* user-data pointer. The original passed
// the queue directly; the callback now also needs the tap handle to re-enable
// it after a timeout, so both travel together.
typedef struct TapContext
{
    Bridge *bridge;
    CFMachPortRef tap;
    uint64_t reenabled;
} TapContext;

static int64_t steady_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static InputEvent make_event(int32_t kind, int32_t dop, int32_t side, CGEventRef event)
{
    InputEvent ev;
    std::memset(&ev, 0, sizeof(ev));
    ev.kind = kind;
    ev.dop = dop;
    ev.mouse_side = side;
    ev.timestamp = steady_ns();
    ev.event_time = CGEventGetTimestamp(event);
    return ev;
}

static InputEvent key_event(int32_t kind, int32_t dop, CGEventRef event)
{
    InputEvent ev = make_event(kind, dop, SIDE_NONE, event);
    ev.key_code = CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode);
    ev.flags = CGEventGetFlags(event);
    // Holding a key makes the OS synthesize repeated keyDowns. They are not
    // new presses and would corrupt dwell/flight times, so they are tagged here
    // (one field read) and filtered by the consumer.
    ev.aux = static_cast<int32_t>(CGEventGetIntegerValueField(event, kCGKeyboardEventAutorepeat));
    return ev;
}

static InputEvent mouse_event(int32_t dop, int32_t side, CGEventRef event, bool with_delta)
{
    InputEvent ev = make_event(KIND_MOUSE, dop, side, event);
    CGPoint loc = CGEventGetLocation(event);
    ev.x = loc.x;
    ev.y = loc.y;
    ev.flags = CGEventGetFlags(event);
    if (with_delta)
    {
        ev.delta_x = CGEventGetIntegerValueField(event, kCGMouseEventDeltaX);
        ev.delta_y = CGEventGetIntegerValueField(event, kCGMouseEventDeltaY);
    }
    return ev;
}

static CGEventRef input_sensor(CGEventTapProxy proxy, CGEventType type, CGEventRef event, void *data)
{
    (void)proxy;
    TapContext *ctx = static_cast<TapContext *>(data);
    switch (type)
    {
    // macOS silently disables a tap whose callback is too slow (or on some
    // secure-input transitions). Without this, capture just stops.
    case kCGEventTapDisabledByTimeout:
    case kCGEventTapDisabledByUserInput:
        CGEventTapEnable(ctx->tap, true);
        ctx->reenabled++;
        break;
    case kCGEventKeyDown:
        ctx->bridge->submit(key_event(KIND_KEY, DOP_DOWN, event));
        break;
    case kCGEventKeyUp:
        ctx->bridge->submit(key_event(KIND_KEY, DOP_UP, event));
        break;
    // Shift/Ctrl/Opt/Cmd/Caps/Fn never produce keyDown/keyUp, only this. The
    // consumer decides press vs release from the device-dependent flag bits.
    case kCGEventFlagsChanged:
        ctx->bridge->submit(key_event(KIND_FLAGS, DOP_NONE, event));
        break;
    case kCGEventMouseMoved:
        ctx->bridge->submit(mouse_event(DOP_NONE, SIDE_NONE, event, true));
        break;
    case kCGEventLeftMouseDown:
        ctx->bridge->submit(mouse_event(DOP_DOWN, SIDE_LEFT, event, false));
        break;
    case kCGEventLeftMouseUp:
        ctx->bridge->submit(mouse_event(DOP_UP, SIDE_LEFT, event, false));
        break;
    case kCGEventLeftMouseDragged:
        ctx->bridge->submit(mouse_event(DOP_DRAG, SIDE_LEFT, event, true));
        break;
    case kCGEventRightMouseDown:
        ctx->bridge->submit(mouse_event(DOP_DOWN, SIDE_RIGHT, event, false));
        break;
    case kCGEventRightMouseUp:
        ctx->bridge->submit(mouse_event(DOP_UP, SIDE_RIGHT, event, false));
        break;
    case kCGEventRightMouseDragged:
        ctx->bridge->submit(mouse_event(DOP_DRAG, SIDE_RIGHT, event, true));
        break;
    case kCGEventScrollWheel:
    {
        InputEvent ev = mouse_event(DOP_NONE, SIDE_SCROLL, event, false);
        ev.delta_y = CGEventGetIntegerValueField(event, kCGScrollWheelEventPointDeltaAxis1);
        ev.delta_x = CGEventGetIntegerValueField(event, kCGScrollWheelEventPointDeltaAxis2);
        ev.aux = static_cast<int32_t>(CGEventGetIntegerValueField(event, kCGScrollWheelEventIsContinuous));
        ctx->bridge->submit(ev);
        break;
    }
    default:
        break;
    }
    return event; // listen-only: never modify or swallow input
}

static void stop_main_loop(void *unused)
{
    (void)unused;
    CFRunLoopStop(CFRunLoopGetMain());
}

static dispatch_source_t stop_on_signal(int sig)
{
    // Signal handlers may only call async-signal-safe functions, and
    // CFRunLoopStop is not one. A dispatch signal source runs the handler as a
    // normal function on the main queue, which CFRunLoopRun services.
    // (_f variant: plain function pointer, so no -fblocks needed in C++.)
    signal(sig, SIG_IGN);
    dispatch_source_t src = dispatch_source_create(DISPATCH_SOURCE_TYPE_SIGNAL, sig, 0, dispatch_get_main_queue());
    dispatch_source_set_event_handler_f(src, stop_main_loop);
    dispatch_resume(src);
    return src;
}

int main()
{
    // Since macOS 10.15 listening to keyboard events needs Input Monitoring.
    // Without it CGEventTapCreate returns NULL (the old code then crashed in
    // CFMachPortCreateRunLoopSource).
    if (!CGPreflightListenEventAccess())
    {
        CGRequestListenEventAccess();
        std::fprintf(stderr, "input_sensor: needs Input Monitoring permission.\n"
                             "  System Settings > Privacy & Security > Input Monitoring: enable this binary or your terminal,\n"
                             "  then re-run.\n");
        return 1;
    }
    int fd = -1;
    SharedRing *ring = ring_create(RING_NAME, &fd);
    if (!ring)
    {
        std::fprintf(stderr, "input_sensor: shared memory setup failed: %s\n", std::strerror(errno));
        return 1;
    }
    Bridge *bridge = new Bridge(ring); // ~650 KB of queue storage: keep it off the stack
    bridge->start();
    TapContext ctx = {bridge, nullptr, 0};
    CGEventMask mask = CGEventMaskBit(kCGEventKeyDown) | CGEventMaskBit(kCGEventKeyUp) |
                       CGEventMaskBit(kCGEventFlagsChanged) | CGEventMaskBit(kCGEventMouseMoved) |
                       CGEventMaskBit(kCGEventLeftMouseDown) | CGEventMaskBit(kCGEventLeftMouseUp) |
                       CGEventMaskBit(kCGEventLeftMouseDragged) | CGEventMaskBit(kCGEventRightMouseDown) |
                       CGEventMaskBit(kCGEventRightMouseUp) | CGEventMaskBit(kCGEventRightMouseDragged) |
                       CGEventMaskBit(kCGEventScrollWheel);
    ctx.tap = CGEventTapCreate(kCGHIDEventTap, kCGTailAppendEventTap, kCGEventTapOptionListenOnly, mask, input_sensor, &ctx);
    if (!ctx.tap)
    {
        std::fprintf(stderr, "input_sensor: CGEventTapCreate failed (permission not yet effective? restart the terminal)\n");
        bridge->stop();
        delete bridge;
        ring_destroy(ring, fd, RING_NAME);
        return 1;
    }
    CFRunLoopSourceRef source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, ctx.tap, 0);
    CFRunLoopAddSource(CFRunLoopGetMain(), source, kCFRunLoopCommonModes);
    CGEventTapEnable(ctx.tap, true);
    dispatch_source_t s_int = stop_on_signal(SIGINT);
    dispatch_source_t s_term = stop_on_signal(SIGTERM);
    std::fprintf(stderr, "input_sensor: capturing into %s (capacity %llu events). Ctrl-C to stop.\n", RING_NAME,
                 (unsigned long long)RING_CAPACITY);
    CFRunLoopRun(); // returns after SIGINT/SIGTERM
    CGEventTapEnable(ctx.tap, false);
    CFRunLoopRemoveSource(CFRunLoopGetMain(), source, kCFRunLoopCommonModes);
    CFRelease(source);
    CFRelease(ctx.tap);
    dispatch_source_cancel(s_int);
    dispatch_source_cancel(s_term);
    bridge->stop();
    std::fprintf(stderr, "input_sensor: published %llu, dropped %llu at ring, %llu at tap queue, tap re-enabled %llu times\n",
                 (unsigned long long)bridge->published(), (unsigned long long)ring->dropped.load(),
                 (unsigned long long)ring->dropped_tap.load(), (unsigned long long)ctx.reenabled);
    delete bridge;
    ring_destroy(ring, fd, RING_NAME);
    return 0;
}
