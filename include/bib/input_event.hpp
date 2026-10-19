#pragma once
// InputEvent: the single record type that crosses every boundary in the system
// (tap callback -> in-process SPSC queue -> shared-memory ring -> Python).
//
// Layout rules, because Python reads this through ctypes/numpy with no compiler
// in the loop:
//   * only fixed-width integer / double fields, no std:: types (the old
//     std::chrono::time_point member relied on libc++ happening to store an
//     int64 of nanoseconds; that is an implementation detail, not an ABI)
//   * padding is explicit (the `aux` field occupies what used to be a hole
//     after mouse_side), so ctypes/numpy and the C++ compiler cannot disagree
//   * every offset is pinned by static_assert below and re-checked at runtime
//     by Python against bib_ring_layout() exported from the shim library
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace bib
{

// kind
constexpr int32_t KIND_KEY = 0;   // keyDown / keyUp
constexpr int32_t KIND_MOUSE = 1; // move / button / drag / scroll
constexpr int32_t KIND_FLAGS = 2; // modifier change (Shift, Cmd, ...): these never arrive as keyDown/keyUp
// dop (down-or-up), kept from the original encoding
constexpr int32_t DOP_DOWN = 0;
constexpr int32_t DOP_UP = 1;
constexpr int32_t DOP_DRAG = 2;
constexpr int32_t DOP_NONE = 3;   // plain move, scroll, flags-changed (consumer infers direction)
// mouse_side, kept from the original encoding
constexpr int32_t SIDE_NONE = 0;
constexpr int32_t SIDE_LEFT = 1;
constexpr int32_t SIDE_RIGHT = 2;
constexpr int32_t SIDE_SCROLL = 3;

typedef struct InputEvent
{
    int32_t kind;
    int32_t dop;
    int64_t key_code;     // CGKeyCode for KIND_KEY / KIND_FLAGS, 0 otherwise
    uint64_t flags;       // raw CGEventFlags (modifier state incl. device-dependent L/R bits)
    int32_t mouse_side;
    int32_t aux;          // key: autorepeat (1 = OS-generated repeat). scroll: 1 = continuous (trackpad)
    double x;             // global display coords, points, origin top-left of main display
    double y;
    int64_t delta_x;      // mouse: integer delta. scroll: horizontal point delta
    int64_t delta_y;      // mouse: integer delta. scroll: vertical point delta
    int64_t timestamp;    // steady_clock nanoseconds, sampled when the callback ran
    uint64_t event_time;  // CGEventGetTimestamp(): when the HID event happened; units resolved by the consumer
} InputEvent;

static_assert(std::is_trivially_copyable<InputEvent>::value, "InputEvent must be memcpy-able");
static_assert(std::is_standard_layout<InputEvent>::value, "InputEvent must have C layout");
static_assert(offsetof(InputEvent, kind) == 0, "layout");
static_assert(offsetof(InputEvent, dop) == 4, "layout");
static_assert(offsetof(InputEvent, key_code) == 8, "layout");
static_assert(offsetof(InputEvent, flags) == 16, "layout");
static_assert(offsetof(InputEvent, mouse_side) == 24, "layout");
static_assert(offsetof(InputEvent, aux) == 28, "layout");
static_assert(offsetof(InputEvent, x) == 32, "layout");
static_assert(offsetof(InputEvent, y) == 40, "layout");
static_assert(offsetof(InputEvent, delta_x) == 48, "layout");
static_assert(offsetof(InputEvent, delta_y) == 56, "layout");
static_assert(offsetof(InputEvent, timestamp) == 64, "layout");
static_assert(offsetof(InputEvent, event_time) == 72, "layout");
static_assert(sizeof(InputEvent) == 80, "layout");

} // namespace bib
