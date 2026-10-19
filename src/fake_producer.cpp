// fake_producer: portable stand-in for input_sensor (no Core Graphics), used to
// test the C++ -> Python path on Linux/macOS and to replay recorded sessions.
//
//   fake_producer --count N [--rate HZ] [--startup S] [--linger S]
//       N synthetic events; key_code carries the sequence number 0..N-1 so the
//       reader can prove ordering and account for every lost event.
//   fake_producer --replay events.bin [--speed X] [--startup S] [--linger S]
//       Replays raw 80-byte InputEvent records with their original spacing
//       (timestamps are re-stamped to "now" so they look live).
// Both go through the same Bridge (SpscQueue -> ring) as the real sensor.
#include "bib/bridge.hpp"
#include "bib/input_event.hpp"
#include "bib/shared_ring.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace bib;
using clk = std::chrono::steady_clock;

static int64_t now_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now().time_since_epoch()).count();
}

static void sleep_s(double s)
{
    std::this_thread::sleep_for(std::chrono::duration<double>(s));
}

static void run_synthetic(Bridge *b, uint64_t count, double rate)
{
    const auto t0 = clk::now();
    for (uint64_t i = 0; i < count; i++)
    {
        InputEvent ev;
        std::memset(&ev, 0, sizeof(ev));
        ev.kind = (int32_t)(i % 3 == 0 ? KIND_MOUSE : KIND_KEY);
        ev.dop = (int32_t)(i % 2);
        ev.key_code = (int64_t)i;
        ev.flags = ~(uint64_t)i; // distinctive bit patterns make torn reads detectable
        ev.x = (double)i * 0.5;
        ev.y = -(double)i;
        ev.delta_x = (int64_t)i * 3;
        ev.delta_y = -(int64_t)i * 3;
        ev.timestamp = now_ns();
        ev.event_time = (uint64_t)ev.timestamp;
        b->submit(ev);
        if (rate > 0)
        {
            const auto due = t0 + std::chrono::duration_cast<clk::duration>(std::chrono::duration<double>((double)(i + 1) / rate));
            std::this_thread::sleep_until(due);
        }
    }
}

static int run_replay(Bridge *b, const char *path, double speed)
{
    FILE *f = std::fopen(path, "rb");
    if (!f)
    {
        std::perror(path);
        return 1;
    }
    std::vector<InputEvent> evs;
    InputEvent ev;
    while (std::fread(&ev, sizeof(ev), 1, f) == 1)
        evs.push_back(ev);
    std::fclose(f);
    if (evs.empty())
        return 0;
    const int64_t src0 = evs[0].timestamp;
    const auto t0 = clk::now();
    const int64_t dst0 = now_ns();
    for (InputEvent &e : evs)
    {
        const double off = (double)(e.timestamp - src0) / 1e9 / speed;
        std::this_thread::sleep_until(t0 + std::chrono::duration_cast<clk::duration>(std::chrono::duration<double>(off)));
        const int64_t shift = dst0 - src0;
        e.timestamp += shift;
        e.event_time = (uint64_t)e.timestamp; // replay: event clock == steady clock
        b->submit(e);
    }
    std::fprintf(stderr, "fake_producer: replayed %zu events\n", evs.size());
    return 0;
}

int main(int argc, char **argv)
{
    uint64_t count = 0;
    double rate = 0, startup = 1.0, linger = 2.0, speed = 1.0;
    const char *replay = nullptr;
    for (int i = 1; i + 1 < argc; i += 2)
    {
        std::string a = argv[i];
        if (a == "--count")
            count = std::strtoull(argv[i + 1], nullptr, 10);
        else if (a == "--rate")
            rate = std::atof(argv[i + 1]);
        else if (a == "--startup")
            startup = std::atof(argv[i + 1]);
        else if (a == "--linger")
            linger = std::atof(argv[i + 1]);
        else if (a == "--replay")
            replay = argv[i + 1];
        else if (a == "--speed")
            speed = std::atof(argv[i + 1]);
    }
    int fd = -1;
    SharedRing *ring = ring_create(RING_NAME, &fd);
    if (!ring)
    {
        std::perror("ring_create");
        return 1;
    }
    Bridge *b = new Bridge(ring, std::chrono::microseconds(200));
    b->start();
    sleep_s(startup); // give the reader time to attach
    int rc = 0;
    if (replay)
        rc = run_replay(b, replay, speed);
    else
        run_synthetic(b, count, rate);
    b->stop();
    std::fprintf(stderr, "fake_producer: published %llu dropped_ring %llu dropped_tap %llu\n",
                 (unsigned long long)b->published(), (unsigned long long)ring->dropped.load(),
                 (unsigned long long)ring->dropped_tap.load());
    sleep_s(linger); // let the reader drain before the segment is unlinked
    delete b;
    ring_destroy(ring, fd, RING_NAME);
    return rc;
}
