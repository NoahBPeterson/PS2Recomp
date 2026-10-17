// Log of host-side writes into EE RAM, for finding which runtime subsystem wrote a guest address.
//   PS2X_HOST_WRITE_LOG=<path>       one line per write:
//       v=<vsync> t=<host thread #> op=<writer> addr=0x... size=0x... pc=0x... ra=0x...
//   PS2X_HOST_WRITE_RANGE=lo-hi      keep only writes overlapping [lo, hi) (physical, hex)
// The file is flushed at every new vsync, so a killed run loses at most the current frame.
//   PS2X_RAM_DUMP_VSYNCS=N[,N...]     write EE RAM to <PS2X_RAM_DUMP_DIR or .>/ram_v<N>.bin at those
//                                     vsyncs (on the EE scheduler thread, between guest slices),
//                                     and GS local memory to gs_v<N>.bin at the next GS vsync
#include "ps2_cpu_context.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>
#if !defined(_WIN32)
#include <dlfcn.h>
#include <pthread.h>
#endif

std::atomic<bool> g_ps2HostWriteTraceOn{false};
uint32_t g_ps2GuestWatchLo = 0x40000000u;
uint32_t g_ps2GuestWatchSpan = 0;
extern std::atomic<uint64_t> g_ps2xDebugGsDumpVsync; // gs_frontend.cpp

namespace
{
    std::atomic<uint64_t> s_vsync{0};
    std::atomic<int> s_nextThread{0};
    std::mutex s_mutex;
    FILE *s_file = nullptr;
    uint32_t s_rangeLo = 0;
    uint32_t s_rangeHi = 0xFFFFFFFFu;
    uint64_t s_lastFlushedVsync = 0;

    struct HostWriteTraceInit
    {
        HostWriteTraceInit()
        {
            const char *path = std::getenv("PS2X_HOST_WRITE_LOG");
            if (!path || !*path)
            {
                return;
            }
            s_file = std::fopen(path, "w");
            if (!s_file)
            {
                std::fprintf(stderr, "[host-write] cannot open %s\n", path);
                return;
            }
            if (const char *range = std::getenv("PS2X_HOST_WRITE_RANGE"); range && *range)
            {
                char *end = nullptr;
                s_rangeLo = static_cast<uint32_t>(std::strtoul(range, &end, 16));
                if (end && *end == '-')
                {
                    s_rangeHi = static_cast<uint32_t>(std::strtoul(end + 1, nullptr, 16));
                }
            }
            g_ps2HostWriteTraceOn.store(true, std::memory_order_relaxed);
        }
    } s_init;

    int threadIndex()
    {
        thread_local int index = -1;
        if (index < 0)
        {
            index = s_nextThread.fetch_add(1, std::memory_order_relaxed);
            char name[64] = "?";
#if !defined(_WIN32)
            pthread_getname_np(pthread_self(), name, sizeof(name));
#endif
            std::fprintf(s_file, "thread %d name=%s\n", index, name[0] ? name : "?");
        }
        return index;
    }
}

void ps2HostWriteTraceSetVsync(uint64_t vsync)
{
    s_vsync.store(vsync, std::memory_order_relaxed);
}

void ps2HostWriteTrace(uint32_t guestAddr, uint32_t size, const char *op, const R5900Context *ctx)
{
    const uint32_t physical = guestAddr & 0x1FFFFFFFu;
    if (size == 0 || physical >= s_rangeHi || physical + size <= s_rangeLo)
    {
        return;
    }
    const uint64_t vsync = s_vsync.load(std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(s_mutex);
    const int thread = threadIndex();
    const uint32_t pc = ctx ? ctx->pc : 0u;
    const uint32_t ra = ctx ? static_cast<uint32_t>(_mm_extract_epi32(ctx->r[31], 0)) : 0u;
    std::fprintf(s_file, "v=%llu t=%d op=%s addr=0x%x size=0x%x pc=0x%x ra=0x%x\n",
                 static_cast<unsigned long long>(vsync), thread, op ? op : "?", physical, size, pc, ra);
    if (vsync != s_lastFlushedVsync)
    {
        s_lastFlushedVsync = vsync;
        std::fflush(s_file);
    }
}

void ps2DebugRamDumpAtVsync(const uint8_t *rdram, uint64_t vsync)
{
    static const std::vector<uint64_t> marks = []()
    {
        std::vector<uint64_t> out;
        if (const char *list = std::getenv("PS2X_RAM_DUMP_VSYNCS"); list && *list)
        {
            const char *p = list;
            while (*p)
            {
                char *end = nullptr;
                const unsigned long long v = std::strtoull(p, &end, 10);
                if (end == p)
                {
                    break;
                }
                out.push_back(v);
                p = (*end == ',') ? end + 1 : end;
            }
        }
        return out;
    }();
    if (marks.empty() || !rdram)
    {
        return;
    }
    for (const uint64_t mark : marks)
    {
        if (mark != vsync)
        {
            continue;
        }
        const char *dir = std::getenv("PS2X_RAM_DUMP_DIR");
        const std::string path = std::string(dir && *dir ? dir : ".") + "/ram_v" + std::to_string(vsync) + ".bin";
        if (FILE *f = std::fopen(path.c_str(), "wb"))
        {
            std::fwrite(rdram, 1, PS2_RAM_SIZE, f);
            std::fclose(f);
        }
        g_ps2xDebugGsDumpVsync.store(vsync, std::memory_order_release);
    }
}

// PS2X_GUEST_WATCH=<addr>[:<len>] (hex, physical; len defaults to 16): every guest store that
// overlaps the range, on stderr: vsync, address, size, value, guest ra and the host symbol of the
// generated function doing the store (its name carries the guest address). PS2X_GUEST_WATCH_MAX
// caps the lines (default 100000). PS2X_GUEST_WATCH_HI=<hex> keeps only 16-byte stores whose upper
// qword equals it, or 8-byte stores of it (e.g. 41: the register qword of an A+D SCISSOR_2 write),
// for finding a packet builder anywhere.
namespace
{
    std::atomic<long> s_watchLines{0};
    long s_watchMax = 100000;
    bool s_watchHiFilter = false;
    uint64_t s_watchHi = 0;
    struct GuestWatchInit
    {
        GuestWatchInit()
        {
            const char *spec = std::getenv("PS2X_GUEST_WATCH");
            if (!spec || !*spec)
                return;
            char *end = nullptr;
            const unsigned long lo = std::strtoul(spec, &end, 16);
            unsigned long len = 16;
            if (end && *end == ':')
                len = std::strtoul(end + 1, nullptr, 16);
            if (const char *h = std::getenv("PS2X_GUEST_WATCH_HI"))
            {
                s_watchHiFilter = true;
                s_watchHi = std::strtoull(h, nullptr, 16);
            }
            if (const char *m = std::getenv("PS2X_GUEST_WATCH_MAX"))
                s_watchMax = std::atol(m);
            // Physical for RAM; a scratchpad range (0x70000000+) is kept as is for special stores.
            g_ps2GuestWatchLo = lo >= 0x70000000ul ? static_cast<uint32_t>(lo) : static_cast<uint32_t>(lo) & 0x1FFFFFFFu;
            g_ps2GuestWatchSpan = static_cast<uint32_t>(len ? len : 1u);
            std::fprintf(stderr, "[guest-watch] 0x%x+0x%x\n", g_ps2GuestWatchLo, g_ps2GuestWatchSpan);
        }
    } s_guestWatchInit;
}

#if defined(_MSC_VER)
#include <intrin.h>
#define PS2X_RETURN_ADDRESS() _ReturnAddress()
#else
#define PS2X_RETURN_ADDRESS() __builtin_return_address(0)
#endif

// Not inlined: the return address is in the generated function that did the store.
#if defined(_MSC_VER)
__declspec(noinline)
#else
__attribute__((noinline))
#endif
static void guestWatchLog(uint32_t guestAddr, uint32_t size, uint64_t valueLo, uint64_t valueHi, const R5900Context *ctx,
                          const void *hostPc);

void ps2GuestWatchHit(uint32_t guestAddr, uint32_t size, uint64_t valueLo, uint64_t valueHi, const R5900Context *ctx)
{
    guestWatchLog(guestAddr, size, valueLo, valueHi, ctx, PS2X_RETURN_ADDRESS());
}

// A store through ps2x_abi::Store64/128 (scratchpad or hardware): watched by its full address when the
// range covers it (PS2X_GUEST_WATCH=70000000:4000 for the scratchpad).
void ps2GuestWatchSpecial(uint32_t vaddr, uint32_t size, uint64_t valueLo, uint64_t valueHi, const R5900Context *ctx,
                          const void *hostPc)
{
    // RAM through the uncached mirrors (0x2xxxxxxx, 0x3xxxxxxx) is watched by its physical address.
    const uint32_t top = vaddr >> 28;
    const uint32_t addr = (top == 2u || top == 3u) ? (vaddr & 0x01FFFFFFu) : vaddr;
    if (addr + size - 1u - g_ps2GuestWatchLo < g_ps2GuestWatchSpan + size - 1u)
        guestWatchLog(addr, size, valueLo, valueHi, ctx, hostPc);
}

static void guestWatchLog(uint32_t guestAddr, uint32_t size, uint64_t valueLo, uint64_t valueHi, const R5900Context *ctx,
                          const void *hostPc)
{
    if (s_watchHiFilter && !(size == 16u ? valueHi == s_watchHi : size <= 8u && valueLo == s_watchHi))
        return;
    if (s_watchLines.fetch_add(1, std::memory_order_relaxed) >= s_watchMax)
        return;
    const char *symbol = "?";
#if !defined(_WIN32)
    Dl_info info{};
    if (hostPc && dladdr(hostPc, &info) && info.dli_sname)
        symbol = info.dli_sname;
#endif
    const uint32_t ra = ctx ? static_cast<uint32_t>(_mm_cvtsi128_si32(ctx->r[31])) : 0u;
    std::fprintf(stderr, "[guest-watch] v=%llu addr=0x%x size=%u value=%016llx%s%016llx ra=0x%x fn=%s\n",
                 static_cast<unsigned long long>(s_vsync.load(std::memory_order_relaxed)), guestAddr, size,
                 static_cast<unsigned long long>(valueHi), size > 8u ? ":" : "", static_cast<unsigned long long>(valueLo),
                 ra, symbol);
}
