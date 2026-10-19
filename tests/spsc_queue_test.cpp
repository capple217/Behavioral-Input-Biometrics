// Tests + microbenchmark for bib::SpscQueue.
//   1. single-thread edge cases: empty pop, fill to capacity, full push fails, FIFO order, index wrap
//   2. two-thread stress: producer pushes N records (spinning when full), consumer checks
//      every record arrives exactly once, in order, untorn. Run under -fsanitize=thread too.
//   3. throughput (ops/s) for the same two-thread transfer, optionally vs boost::lockfree.
#include "bib/input_event.hpp"
#include "bib/spsc_queue.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#ifdef BIB_HAVE_BOOST
#include <boost/lockfree/spsc_queue.hpp>
#endif

using namespace bib;

static int failures = 0;
#define CHECK(c)                                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(c))                                                                                                      \
        {                                                                                                              \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);                                         \
            failures++;                                                                                                \
        }                                                                                                              \
    } while (0)

static InputEvent make(uint64_t i)
{
    InputEvent e;
    std::memset(&e, 0, sizeof(e));
    e.key_code = (int64_t)i;
    e.flags = ~i;
    e.timestamp = (int64_t)(i * 7);
    e.event_time = i ^ 0x5555555555555555ULL;
    return e;
}

static bool intact(const InputEvent &e, uint64_t i)
{
    return e.key_code == (int64_t)i && e.flags == ~i && e.timestamp == (int64_t)(i * 7) &&
           e.event_time == (i ^ 0x5555555555555555ULL);
}

static void test_single_thread()
{
    SpscQueue<InputEvent, 8> *q = new SpscQueue<InputEvent, 8>();
    InputEvent e;
    CHECK(!q->try_pop(e));
    for (uint64_t i = 0; i < 8; i++)
        CHECK(q->try_push(make(i)));
    CHECK(!q->try_push(make(99))); // full: newest is refused, nothing overwritten
    CHECK(q->size_approx() == 8);
    for (uint64_t i = 0; i < 8; i++)
        CHECK(q->try_pop(e) && intact(e, i));
    CHECK(!q->try_pop(e));
    // many laps around the ring with varying fill levels
    uint64_t next_in = 0, next_out = 0;
    for (int round = 0; round < 10000; round++)
    {
        int k = (round * 7) % 9;
        for (int j = 0; j < k; j++)
            if (q->try_push(make(next_in)))
                next_in++;
        int m = (round * 5) % 9;
        for (int j = 0; j < m; j++)
            if (q->try_pop(e))
            {
                CHECK(intact(e, next_out));
                next_out++;
            }
    }
    CHECK(next_in >= next_out);
    delete q;
}

template <typename Q, typename Push, typename Pop>
static double transfer(Q *q, uint64_t n, Push push, Pop pop, bool verify)
{
    auto t0 = std::chrono::steady_clock::now();
    std::thread prod([&]() {
        for (uint64_t i = 0; i < n; i++)
        {
            InputEvent e = make(i);
            while (!push(q, e))
                ; // spin: this test wants zero loss
        }
    });
    uint64_t expect = 0;
    InputEvent e;
    while (expect < n)
    {
        if (pop(q, e))
        {
            if (verify && !intact(e, expect))
            {
                std::fprintf(stderr, "FAIL: record %llu out of order or torn (got key_code %lld)\n",
                             (unsigned long long)expect, (long long)e.key_code);
                failures++;
                break;
            }
            expect++;
        }
    }
    prod.join();
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return (double)n / s;
}

int main(int argc, char **argv)
{
    uint64_t n = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : 5000000ULL;
    test_single_thread();
    typedef SpscQueue<InputEvent, 8192> Q;
    Q *q = new Q();
    auto push = [](Q *qq, const InputEvent &e) { return qq->try_push(e); };
    auto pop = [](Q *qq, InputEvent &e) { return qq->try_pop(e); };
    double r1 = transfer(q, n, push, pop, true);
    std::printf("bib::SpscQueue      %8.2f M events/s (%llu x %zu B, verified)\n", r1 / 1e6, (unsigned long long)n,
                sizeof(InputEvent));
#ifdef BIB_HAVE_BOOST
    typedef boost::lockfree::spsc_queue<InputEvent, boost::lockfree::capacity<8192>> BQ;
    BQ *bq = new BQ();
    auto bpush = [](BQ *qq, const InputEvent &e) { return qq->push(e); };
    auto bpop = [](BQ *qq, InputEvent &e) { return qq->pop(e); };
    double r2 = transfer(bq, n, bpush, bpop, true);
    std::printf("boost::spsc_queue   %8.2f M events/s (same transfer)\n", r2 / 1e6);
    delete bq;
#endif
    delete q;
    if (failures)
    {
        std::printf("%d FAILURES\n", failures);
        return 1;
    }
    std::printf("spsc_queue_test: OK\n");
    return 0;
}
