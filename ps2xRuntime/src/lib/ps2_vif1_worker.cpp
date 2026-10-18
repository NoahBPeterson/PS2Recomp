// VIF1 worker thread (T-0122); see ps2_vif1_worker.h.
#include "ps2_vif1_worker.h"

#include <algorithm>
#include <cstdlib>

#if defined(__APPLE__) || defined(__linux__)
#include <pthread.h>
#endif
#if defined(__APPLE__)
#include <pthread/qos.h>
#endif
namespace
{
    void nameVif1Thread()
    {
#if defined(__APPLE__)
        pthread_setname_np("VIF1-VU1");
        // Without a QoS class macOS may run it on the efficiency cores; it is half the frame's work.
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#elif defined(__linux__)
        pthread_setname_np(pthread_self(), "VIF1-VU1");
#endif
    }
}

void Vif1Worker::submit(PS2Memory &memory, Vif1Job &&job)
{
    job.startCycle = memory.m_eeClock ? memory.m_eeClock() : 0u;
    job.fbrst = memory.m_vu0FbrstSource ? memory.m_vu0FbrstSource() : 0u;
    Vif1Worker *worker = memory.m_vif1Worker.get();
    if (!worker)
    {
        run(memory, job);
        return;
    }
    uint64_t gateAt = 0u;
    {
        std::lock_guard lock(worker->mutex);
        // An idle worker starts the job no earlier than now, so nothing it queues is before now;
        // a busy one may still queue events from earlier work at its own clock.
        if (!worker->hasWork.load())
            worker->clock.store(std::max(worker->clock.load(), job.startCycle));
        worker->jobs.push_back(std::move(job));
        worker->hasWork.store(true);
        gateAt = worker->clock.load();
    }
    worker->workCv.notify_one();
    if (worker->gateArmedAt == 0u || worker->gateArmedAt > gateAt)
    {
        worker->gateArmedAt = gateAt;
        memory.m_scheduleDeviceDeadline(gateAt);
    }
}

void Vif1Worker::run(PS2Memory &memory, Vif1Job &job)
{
    memory.m_vu1RunFbrst = job.fbrst;
    if (memory.deviceTimingEnabled())
        memory.m_vif1Clock = std::max(memory.m_vif1Clock, job.startCycle);
    publish(memory);
    for (Vif1JobSegment &segment : job.segments)
    {
        // The chain's source map, so a stall inside it can read the held-back part again.
        memory.m_vif1Sources.buffer = segment.tracked ? segment.data.data() : nullptr;
        memory.m_vif1Sources.current = std::move(segment.sources);
        memory.processVIF1Data(segment.data.data(), static_cast<uint32_t>(segment.data.size()));
        memory.m_vif1Sources.buffer = nullptr;
        memory.m_vif1Sources.current = Vif1ChainSources{};
    }
    if (memory.m_gifArbiter)
        memory.m_gifArbiter->drain();
    if (job.kind == Vif1Job::Kind::Dma)
    {
        // A transfer stalled on a VIFcode i-bit isn't finished until the CPU releases the stall
        // (FBRST.STC).
        if (memory.m_vif1Stalled)
            memory.m_vif1DmaHeld = true;
        else
            memory.finishVif1Dma();
    }
    else if (!memory.m_vif1Stalled && memory.m_vif1DmaHeld)
    {
        memory.m_vif1DmaHeld = false;
        memory.finishVif1Dma();
    }
}

void Vif1Worker::publish(PS2Memory &memory)
{
    Vif1Worker *worker = memory.m_vif1Worker.get();
    if (!worker || std::this_thread::get_id() != worker->id)
        return;
    const uint64_t now = memory.m_vif1Clock;
    worker->clock.store(now);
    if (now > worker->eeWaitingAt.load())
    {
        std::lock_guard lock(worker->mutex);
        worker->doneCv.notify_all();
    }
}

void Vif1Worker::gate(PS2Memory &memory, uint64_t eeCycle)
{
    Vif1Worker *worker = memory.m_vif1Worker.get();
    if (!worker)
        return;
    if (worker->gateArmedAt != 0u && worker->gateArmedAt <= eeCycle)
        worker->gateArmedAt = 0u; // that deadline is due now
    if (!worker->hasWork.load())
        return;
    uint64_t clock = worker->clock.load();
    if (clock <= eeCycle)
    {
        // Every event up to eeCycle must be queued before the scheduler collects due deadlines.
        // Waiting for a quantum more than that costs nothing in exactness and saves handoffs.
        const uint64_t target = eeCycle + worker->quantum;
        std::unique_lock lock(worker->mutex);
        worker->eeWaitingAt.store(target);
        worker->doneCv.wait(lock, [&] { return !worker->hasWork.load() || worker->clock.load() > target; });
        worker->eeWaitingAt.store(kNoCycle);
        clock = worker->clock.load();
        if (!worker->hasWork.load())
            return;
    }
    if (worker->gateArmedAt == 0u)
    {
        worker->gateArmedAt = clock;
        memory.m_scheduleDeviceDeadline(clock);
    }
}

void Vif1Worker::loop(PS2Memory &memory)
{
    nameVif1Thread();
    for (;;)
    {
        Vif1Job job;
        {
            std::unique_lock lock(mutex);
            workCv.wait(lock, [&] { return stop || !jobs.empty(); });
            if (jobs.empty())
                return; // stop
            job = std::move(jobs.front());
            jobs.pop_front();
            busy = true;
        }
        run(memory, job);
        {
            std::lock_guard lock(mutex);
            busy = false;
            clock.store(std::max(clock.load(), memory.m_vif1Clock));
            if (jobs.empty())
                hasWork.store(false);
        }
        doneCv.notify_all();
    }
}

void PS2Memory::startVif1Worker()
{
    if (m_vif1Worker || !deviceTimingEnabled())
        return;
    static const bool enabled = []
    {
        const char *value = std::getenv("PS2X_VIF1_THREAD");
        return !(value && value[0] == '0');
    }();
    if (!enabled)
        return;
    auto worker = std::make_unique<Vif1Worker>();
    if (const char *quantum = std::getenv("PS2X_VIF1_QUANTUM"))
        worker->quantum = std::strtoull(quantum, nullptr, 10);
    worker->clock.store(m_vif1Clock);
    Vif1Worker *raw = worker.get();
    m_vif1Worker = std::move(worker);
    raw->thread = std::thread([this, raw] { raw->loop(*this); });
    raw->id = raw->thread.get_id();
}

void PS2Memory::stopVif1Worker()
{
    if (!m_vif1Worker)
        return;
    {
        std::lock_guard lock(m_vif1Worker->mutex);
        m_vif1Worker->stop = true;
    }
    m_vif1Worker->workCv.notify_all();
    if (m_vif1Worker->thread.joinable())
        m_vif1Worker->thread.join();
    m_vif1Worker.reset();
}

bool PS2Memory::onVif1Worker() const
{
    return m_vif1Worker && std::this_thread::get_id() == m_vif1Worker->id;
}

void PS2Memory::syncVif1Worker()
{
    Vif1Worker *worker = m_vif1Worker.get();
    if (!worker || !worker->hasWork.load() || std::this_thread::get_id() == worker->id)
        return;
    std::unique_lock lock(worker->mutex);
    worker->doneCv.wait(lock, [&] { return !worker->hasWork.load(); });
}

void PS2Memory::gateVif1Worker(uint64_t eeCycle)
{
    Vif1Worker::gate(*this, eeCycle);
}

uint64_t PS2Memory::lastDeviceEventCycle() const
{
    std::lock_guard lock(m_deviceEventMutex);
    return m_deviceEvents.empty() ? 0u : m_deviceEvents.back().cycle;
}
