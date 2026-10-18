#pragma once
// VIF1 worker thread (T-0122). The EE queues VIF1 work (a DMA kick's data, the rest of a stalled
// transfer); the worker runs it: VIF1 commands, the VU1 microprograms they start, the GIF packets
// those send. Results stay deterministic because everything the EE can observe of it is either
//  - a timed device event (DMA completion, i-bit interrupt) applied at its EE cycle, and the EE
//    never passes a scheduler checkpoint at or beyond the worker's clock (gate()), so every event
//    up to the EE's cycle is queued before the scheduler sorts and applies due events; or
//  - state the worker owns (VU1 memory, VIF1 registers, the GIF arbiter and GS), which the EE
//    reads or writes only after waiting for the worker to go idle (PS2Memory::syncVif1Worker).

#include "runtime/ps2_memory.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

struct Vif1JobSegment
{
    std::vector<uint8_t> data;
    Vif1ChainSources sources; // guest addresses of `data` (a chain), for re-reading after a stall
    bool tracked = false;
};

struct Vif1Job
{
    enum class Kind
    {
        Dma,    // a VIF1 DMA kick: completes the transfer unless VIF1 stalls in it
        Resume, // the rest of a stalled transfer after FBRST.STC
    };
    Kind kind = Kind::Dma;
    std::vector<Vif1JobSegment> segments;
    uint64_t startCycle = 0u; // EE cycle the work was queued
    uint32_t fbrst = 0u;      // VU0 FBRST at that time (DE1/TE1 for the VU1 callbacks)
};

struct Vif1Worker
{
    static constexpr uint64_t kNoCycle = std::numeric_limits<uint64_t>::max();

    // EE thread: runs the job now (no worker) or queues it.
    static void submit(PS2Memory &memory, Vif1Job &&job);
    // Runs a job on the calling thread (the worker, or the EE without one).
    static void run(PS2Memory &memory, Vif1Job &job);
    // The worker's VIF1 clock moved (m_vif1Clock); wakes an EE gate() waiting for it.
    static void publish(PS2Memory &memory);
    // EE thread, scheduler: waits until the worker has no work or its clock is past eeCycle (plus
    // the quantum); then arms the next gate deadline at the worker's clock while it has work.
    static void gate(PS2Memory &memory, uint64_t eeCycle);

    void loop(PS2Memory &memory);

    std::thread thread;
    std::thread::id id;
    std::mutex mutex;
    std::condition_variable workCv; // jobs queued / stop
    std::condition_variable doneCv; // worker idle / clock moved past eeWaitingAt
    std::deque<Vif1Job> jobs;
    bool busy = false;
    bool stop = false;
    std::atomic<bool> hasWork{false};          // a job is queued or running
    std::atomic<uint64_t> clock{0u};           // no event the worker queues later is before this
    std::atomic<uint64_t> eeWaitingAt{kNoCycle}; // the EE waits in gate() for clock > this
    uint64_t gateArmedAt = 0u;                 // EE: cycle of the pending gate deadline (0: none)
    uint64_t quantum = 20000u;                 // EE: gate() waits for clock > eeCycle + quantum
};
