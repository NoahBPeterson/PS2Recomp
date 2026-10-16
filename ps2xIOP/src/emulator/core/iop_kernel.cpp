#include "iop_kernel.h"

#include "iop_memory.h"
#include "../iop_emulator_const.h"

#include <algorithm>
#include <cstdio>
#include <iterator>

namespace ps2x::iop::detail
{
    namespace
    {
        uint32_t alignUp(uint32_t value, uint32_t alignment)
        {
            if (alignment <= 1u)
                return value;
            const uint32_t mask = alignment - 1u;
            return (value + mask) & ~mask;
        }
    }

    IopKernel::IopKernel(IopMemory &memory) noexcept
        : m_memory(memory)
    {
    }

    void IopKernel::reset()
    {
        m_threads.clear();
        m_semaphores.clear();
        m_eventFlags.clear();
        m_alarms.clear();
        m_nextThreadId = 1;
        m_nextSemaphoreId = 1;
        m_nextEventFlagId = 1;
        m_currentThread = nullptr;
    }

    // thbase: the error paths and pure functions are checked against the IOPRP300 THREADMAN by
    // ps2_iop_ground_truth stage 2; scheduling (preemption, blocking, time) is not probed yet.
    namespace
    {
        constexpr int32_t KE_ERROR = -1;
        constexpr int32_t KE_ILLEGAL_ATTR = -401;
        constexpr int32_t KE_ILLEGAL_ENTRY = -402;
        constexpr int32_t KE_ILLEGAL_PRIORITY = -403;
        constexpr int32_t KE_ILLEGAL_STACK_SIZE = -404;
        constexpr int32_t KE_ILLEGAL_THID = -406;
        constexpr int32_t KE_UNKNOWN_THID = -407;
        constexpr int32_t KE_DORMANT = -413;
        constexpr int32_t KE_NOT_DORMANT = -414;
        constexpr int32_t KE_NOT_WAIT = -416;
        constexpr int32_t KE_ILLEGAL_MODE = -405;
        constexpr int32_t KE_UNKNOWN_SEMID = -408;
        constexpr int32_t KE_UNKNOWN_EVFID = -409;
        constexpr int32_t KE_SEMA_ZERO = -419;
        constexpr int32_t KE_SEMA_OVF = -420;
        constexpr int32_t KE_EVF_COND = -421;
        constexpr int32_t KE_EVF_MULTI = -422;
        constexpr int32_t KE_EVF_ILPAT = -423;
        constexpr int32_t KE_WAIT_DELETE = -425;
        constexpr uint32_t SA_ATTR_MASK = 0x101u; // SA_THPRI | SA_IHTHPRI
        constexpr uint32_t EA_MULTI = 0x2u;
        constexpr uint32_t WEF_OR = 0x1u, WEF_CLEAR = 0x10u;
        // CreateThread attribute bits the firmware accepts (TH_ASM, TH_C, TH_UMODE,
        // TH_NO_FILLSTACK, TH_CLEAR_STACK and three high bits it lets through).
        constexpr uint32_t TH_ATTR_MASK = 0xE3300008u;
    }

    bool IopKernel::dispatchThreadImport(uint16_t ordinal, IopCpuState &cpu, uint64_t currentCycle)
    {
        const auto setV0 = [&](int32_t value)
        {
            cpu.gpr[2] = static_cast<uint32_t>(value);
        };

        switch (ordinal)
        {
        case 4: // CreateThread(&{attr, option, entry, stacksize, priority})
        {
            const uint32_t descriptor = cpu.gpr[4];
            const uint32_t attr = m_memory.read32(descriptor + 0u);
            const uint32_t priority = m_memory.read32(descriptor + 16u);
            const uint32_t entry = m_memory.read32(descriptor + 8u);
            const uint32_t stackSize = m_memory.read32(descriptor + 12u);
            if ((attr & ~TH_ATTR_MASK) != 0u)
            {
                setV0(KE_ILLEGAL_ATTR);
                return true;
            }
            if (priority - 1u >= 126u)
            {
                setV0(KE_ILLEGAL_PRIORITY);
                return true;
            }
            if ((entry & 3u) != 0u)
            {
                setV0(KE_ILLEGAL_ENTRY);
                return true;
            }
            if (stackSize < 0x130u)
            {
                setV0(KE_ILLEGAL_STACK_SIZE);
                return true;
            }
            IopThread thread;
            thread.id = static_cast<int>(m_nextThreadId++);
            thread.attr = m_memory.read32(descriptor + 0u);
            thread.option = m_memory.read32(descriptor + 4u);
            thread.entry = m_memory.read32(descriptor + 8u);
            thread.stackSize = alignUp(stackSize, 256u); // the firmware rounds the stack to 256 bytes
            thread.priority = 0u; // the firmware reports 0 until the thread is first started
            thread.initialPriority = priority;
            thread.stackBase = m_memory.allocate(thread.stackSize + kStackGuardBytes, 16u);
            if (thread.stackBase == 0u)
            {
                setV0(-400);
                return true;
            }
            m_memory.write32(descriptor + 12u, thread.stackSize); // written back, as the firmware does
            const int id = thread.id;
            m_threads.emplace(id, std::move(thread));
            setV0(id);
            return true;
        }
        case 5: // DeleteThread
        {
            const int id = static_cast<int>(cpu.gpr[4]);
            if (id == 0)
            {
                setV0(KE_ILLEGAL_THID);
                return true;
            }
            const auto it = m_threads.find(id);
            if (it == m_threads.end())
            {
                setV0(KE_UNKNOWN_THID);
                return true;
            }
            if (it->second.state != IopThreadState::Dormant)
            {
                setV0(KE_NOT_DORMANT);
                return true;
            }
            if (it->second.stackBase != 0u)
                (void)m_memory.freeAllocation(it->second.stackBase);
            m_threads.erase(it);
            setV0(0);
            return true;
        }
        case 6: // StartThread
        case 7: // StartThreadArgs
        {
            const int id = static_cast<int>(cpu.gpr[4]);
            if (id == 0)
            {
                setV0(KE_ILLEGAL_THID);
                return true;
            }
            const auto it = m_threads.find(id);
            if (it == m_threads.end())
            {
                setV0(KE_UNKNOWN_THID);
                return true;
            }
            IopThread &thread = it->second;
            if (thread.state != IopThreadState::Dormant)
            {
                setV0(KE_NOT_DORMANT);
                return true;
            }
            thread.priority = thread.initialPriority; // every start resets the priority...
            thread.wakeupCount = 0;                   // ...and the pending wakeups
            thread.cpu = {};
            thread.cpu.pc = thread.entry;
            thread.cpu.gpr[4] = cpu.gpr[5];
            thread.cpu.gpr[5] = ordinal == 7 ? cpu.gpr[6] : 0u;
            thread.cpu.gpr[28] = cpu.gpr[28];
            thread.cpu.gpr[29] = alignUp(thread.stackBase + thread.stackSize, 16u) - 16u;
            thread.cpu.gpr[31] = kThreadReturnSentinel;
            thread.state = IopThreadState::Ready;
            setV0(0);
            // A thread started with a higher priority than the caller runs at once (checked
            // against the firmware: the caller only continues after it has run).
            if (m_currentThread != nullptr && thread.priority < m_currentThread->priority)
                cpu.yielded = true;
            return true;
        }
        case 8: // ExitThread
        case 9: // ExitDeleteThread
            if (m_currentThread != nullptr)
            {
                m_currentThread->state = ordinal == 9 ? IopThreadState::Dead : IopThreadState::Dormant;
                cpu.stopped = true;
                cpu.yielded = true;
            }
            setV0(0);
            return true;
        case 10:
        case 11: // TerminateThread
        {
            const int id = static_cast<int>(cpu.gpr[4]);
            const auto it = m_threads.find(id);
            if (id == 0 || (it != m_threads.end() && &it->second == m_currentThread))
            {
                setV0(KE_ILLEGAL_THID);
                return true;
            }
            if (it == m_threads.end())
            {
                setV0(KE_UNKNOWN_THID);
                return true;
            }
            if (it->second.state == IopThreadState::Dormant)
            {
                setV0(KE_DORMANT);
                return true;
            }
            it->second.state = IopThreadState::Dormant;
            it->second.waitId = 0;
            setV0(0);
            return true;
        }
        case 12:
        case 13:
            setV0(0);
            return true;
        case 14:
        case 15: // ChangeThreadPriority(thid or 0 = self, priority or 0 = the caller's)
        {
            int id = static_cast<int>(cpu.gpr[4]);
            if (id == 0 && m_currentThread != nullptr)
                id = m_currentThread->id;
            const auto it = m_threads.find(id);
            if (it == m_threads.end())
            {
                setV0(KE_UNKNOWN_THID);
                return true;
            }
            if (it->second.state == IopThreadState::Dormant)
            {
                setV0(KE_DORMANT);
                return true;
            }
            uint32_t priority = cpu.gpr[5];
            if (priority == 0u)
                priority = m_currentThread != nullptr ? m_currentThread->priority : it->second.priority;
            else if (priority - 1u >= 126u)
            {
                setV0(KE_ILLEGAL_PRIORITY);
                return true;
            }
            it->second.priority = priority;
            setV0(0);
            // Rescheduling, as the firmware does: lowering the caller below (or to) a ready
            // thread's priority, or raising a ready thread above the caller, switches threads.
            if (m_currentThread != nullptr)
            {
                bool yield = false;
                if (&it->second == m_currentThread)
                {
                    for (const auto &[otherId, other] : m_threads)
                        if (other.state == IopThreadState::Ready && other.priority <= priority)
                            yield = true;
                }
                else if (it->second.state == IopThreadState::Ready && priority < m_currentThread->priority)
                    yield = true;
                if (yield)
                    cpu.yielded = true;
            }
            return true;
        }
        case 16:
        case 17:
            setV0(0);
            cpu.yielded = true;
            return true;
        case 18:
        case 19: // ReleaseWaitThread
        {
            const int id = static_cast<int>(cpu.gpr[4]);
            const auto it = m_threads.find(id);
            if (id == 0 || (it != m_threads.end() && &it->second == m_currentThread))
            {
                setV0(KE_ILLEGAL_THID);
                return true;
            }
            if (it == m_threads.end())
            {
                setV0(KE_UNKNOWN_THID);
                return true;
            }
            if (it->second.state == IopThreadState::Sleep ||
                it->second.state == IopThreadState::Delay ||
                it->second.state == IopThreadState::Semaphore ||
                it->second.state == IopThreadState::EventFlag)
            {
                it->second.state = IopThreadState::Ready;
                it->second.waitId = 0;
                it->second.cpu.gpr[2] = static_cast<uint32_t>(-418); // KE_RELEASE_WAIT
                setV0(0);
            }
            else
                setV0(KE_NOT_WAIT);
            return true;
        }
        case 20:
            setV0(m_currentThread != nullptr ? m_currentThread->id : 0);
            return true;
        case 21:
            setV0(0x1000);
            return true;
        case 22:
        case 23:
            setV0(referThreadStatus(static_cast<int>(cpu.gpr[4]), cpu.gpr[5]) ? 0 : KE_UNKNOWN_THID);
            return true;
        case 24: // SleepThread
            if (m_currentThread != nullptr)
            {
                if (m_currentThread->wakeupCount > 0)
                    --m_currentThread->wakeupCount;
                else
                    sleepCurrent(cpu);
            }
            setV0(0);
            return true;
        case 25:
        case 26: // WakeupThread
        {
            const int id = static_cast<int>(cpu.gpr[4]);
            const auto it = m_threads.find(id);
            if (id == 0 || (it != m_threads.end() && &it->second == m_currentThread))
            {
                setV0(KE_ILLEGAL_THID);
                return true;
            }
            if (it == m_threads.end())
            {
                setV0(KE_UNKNOWN_THID);
                return true;
            }
            if (it->second.state == IopThreadState::Dormant)
            {
                setV0(KE_DORMANT);
                return true;
            }
            setV0(0);
            if (it->second.state == IopThreadState::Sleep)
            {
                it->second.state = IopThreadState::Ready;
                if (m_currentThread != nullptr && it->second.priority < m_currentThread->priority)
                    cpu.yielded = true; // preempted by the woken thread, as on the firmware
            }
            else
                ++it->second.wakeupCount;
            return true;
        }
        case 27:
        case 28: // CancelWakeupThread(thid or 0 = self): returns the pending wakeup count
        {
            int id = static_cast<int>(cpu.gpr[4]);
            if (id == 0 && m_currentThread != nullptr)
                id = m_currentThread->id;
            const auto it = m_threads.find(id);
            if (it == m_threads.end())
            {
                setV0(KE_UNKNOWN_THID);
                return true;
            }
            const int old = it->second.wakeupCount;
            it->second.wakeupCount = 0;
            setV0(old);
            return true;
        }
        case 29: // SuspendThread
        case 30: // iSuspendThread
        case 31: // ResumeThread
        case 32: // iResumeThread
            // Not implemented by the IOPRP300 THREADMAN: all four just return KE_ERROR.
            setV0(KE_ERROR);
            return true;
        case 33: // DelayThread
            if (m_currentThread != nullptr)
            {
                const uint64_t delayCycles = (static_cast<uint64_t>(cpu.gpr[4]) * kIopClockHz + 999'999ull) / 1'000'000ull;
                delayCurrentUntil(currentCycle + std::max<uint64_t>(delayCycles, 1u), cpu);
            }
            setV0(0);
            return true;
        case 34: // GetSystemTime
            m_memory.write32(cpu.gpr[4], static_cast<uint32_t>(currentCycle));
            m_memory.write32(cpu.gpr[4] + 4u, static_cast<uint32_t>(currentCycle >> 32u));
            setV0(0);
            return true;
        case 35: // SetAlarm
        case 36: // iSetAlarm
        {
            constexpr int32_t kErrorFoundHandler = -104;
            const uint32_t clockAddress = cpu.gpr[4];
            const uint32_t handler = cpu.gpr[5];
            const uint32_t argument = cpu.gpr[6];
            const uint64_t clock = static_cast<uint64_t>(m_memory.read32(clockAddress)) |
                                   (static_cast<uint64_t>(m_memory.read32(clockAddress + 4u)) << 32u);
            const bool duplicate = std::any_of(m_alarms.begin(), m_alarms.end(), [&](const IopAlarm &alarm)
                                               { return alarm.handler == handler && alarm.argument == argument; });
            if (duplicate)
            {
                setV0(kErrorFoundHandler);
                return true;
            }
            m_alarms.push_back({currentCycle + std::max<uint64_t>(clock, 1u), handler, argument, cpu.gpr[28]});
            setV0(0);
            return true;
        }
        case 37: // CancelAlarm
        case 38: // iCancelAlarm
        {
            constexpr int32_t kErrorNotFoundHandler = -105;
            const uint32_t handler = cpu.gpr[4];
            const uint32_t argument = cpu.gpr[5];
            const auto it = std::find_if(m_alarms.begin(), m_alarms.end(), [&](const IopAlarm &alarm)
                                         { return alarm.handler == handler && alarm.argument == argument; });
            if (it == m_alarms.end())
            {
                setV0(kErrorNotFoundHandler);
                return true;
            }
            m_alarms.erase(it);
            setV0(0);
            return true;
        }
        case 39: // USec2SysClock
        {
            const uint64_t cycles = (static_cast<uint64_t>(cpu.gpr[4]) * kIopClockHz) / 1'000'000ull;
            m_memory.write32(cpu.gpr[5], static_cast<uint32_t>(cycles));
            m_memory.write32(cpu.gpr[5] + 4u, static_cast<uint32_t>(cycles >> 32u));
            setV0(0);
            return true;
        }
        case 40:
        {
            const uint64_t cycles = static_cast<uint64_t>(m_memory.read32(cpu.gpr[4])) | (static_cast<uint64_t>(m_memory.read32(cpu.gpr[4] + 4u)) << 32u);
            const uint64_t usec = (cycles * 1'000'000ull) / kIopClockHz;
            if (cpu.gpr[5] != 0u)
                m_memory.write32(cpu.gpr[5], static_cast<uint32_t>(usec / 1'000'000ull));
            if (cpu.gpr[6] != 0u)
                m_memory.write32(cpu.gpr[6], static_cast<uint32_t>(usec % 1'000'000ull));
            setV0(0);
            return true;
        }
        case 42: // GetThreadCurrentPriority
            setV0(m_currentThread != nullptr ? static_cast<int32_t>(m_currentThread->priority) : 0);
            return true;
        case 41:
            setV0(0);
            return true;
        default:
            return false;
        }
    }

    bool IopKernel::referThreadStatus(int id, uint32_t outputAddress)
    {
        if (id == 0 && m_currentThread != nullptr)
            id = m_currentThread->id;
        const auto it = m_threads.find(id);
        if (it == m_threads.end() || outputAddress == 0u)
            return false;

        const IopThread &thread = it->second;
        uint32_t status = 0x10u;
        switch (thread.state)
        {
        case IopThreadState::Running:
            status = 0x01u;
            break;
        case IopThreadState::Ready:
            status = 0x02u;
            break;
        case IopThreadState::Sleep:
        case IopThreadState::Delay:
        case IopThreadState::Semaphore:
        case IopThreadState::EventFlag:
            status = 0x04u;
            break;
        case IopThreadState::Suspended:
            status = 0x08u;
            break;
        default:
            status = 0x10u;
            break;
        }

        m_memory.write32(outputAddress + 0u, thread.attr);
        m_memory.write32(outputAddress + 4u, thread.option);
        m_memory.write32(outputAddress + 8u, status);
        m_memory.write32(outputAddress + 12u, thread.entry);
        m_memory.write32(outputAddress + 16u, thread.stackBase);
        m_memory.write32(outputAddress + 20u, thread.stackSize);
        m_memory.write32(outputAddress + 24u, thread.cpu.gpr[28]);
        m_memory.write32(outputAddress + 28u, thread.initialPriority);
        m_memory.write32(outputAddress + 32u, thread.priority);
        m_memory.write32(outputAddress + 36u, thread.state == IopThreadState::Sleep ? 1u : thread.state == IopThreadState::Delay   ? 2u
                                                                                       : thread.state == IopThreadState::Semaphore ? 3u
                                                                                       : thread.state == IopThreadState::EventFlag ? 4u
                                                                                                                                   : 0u);
        m_memory.write32(outputAddress + 40u, static_cast<uint32_t>(thread.waitId));
        m_memory.write32(outputAddress + 44u, static_cast<uint32_t>(thread.wakeupCount));
        for (uint32_t word = 12; word < 17; ++word) // regContext and reserved[4]
            m_memory.write32(outputAddress + 4u * word, 0u);
        return true;
    }

    // thsemap as the IOPRP300 THREADMAN implements it; the non-blocking paths are checked against
    // the original module by ps2_iop_ground_truth stage 2 (tools/conformance/iop.sh). Known gap: the
    // i-variants are accepted from thread context (the firmware returns KE_ILLEGAL_CONTEXT there);
    // the HLE does not track interrupt context yet.

    bool IopKernel::dispatchSemaphoreImport(uint16_t ordinal, IopCpuState &cpu)
    {
        const auto setV0 = [&](int32_t value)
        {
            cpu.gpr[2] = static_cast<uint32_t>(value);
        };
        const auto waiters = [&](int id)
        {
            uint32_t n = 0u;
            for (const auto &[threadId, thread] : m_threads)
                if (thread.state == IopThreadState::Semaphore && thread.waitId == id)
                    ++n;
            return n;
        };

        if (ordinal == 4) // CreateSema(&{attr, option, initial, max})
        {
            const uint32_t descriptor = cpu.gpr[4];
            Semaphore semaphore;
            semaphore.attr = m_memory.read32(descriptor + 0u);
            if ((semaphore.attr & ~SA_ATTR_MASK) != 0u)
            {
                setV0(KE_ILLEGAL_ATTR);
                return true;
            }
            semaphore.id = static_cast<int>(m_nextSemaphoreId++);
            semaphore.option = m_memory.read32(descriptor + 4u);
            semaphore.current = static_cast<int>(m_memory.read32(descriptor + 8u));
            semaphore.initial = semaphore.current;
            semaphore.maximum = static_cast<int>(m_memory.read32(descriptor + 12u));
            m_semaphores.emplace(semaphore.id, semaphore);
            setV0(semaphore.id);
            return true;
        }
        if (ordinal < 5 || ordinal > 12 || ordinal == 10)
            return false;

        const int id = static_cast<int>(cpu.gpr[4]);
        const auto it = m_semaphores.find(id);
        if (it == m_semaphores.end())
        {
            setV0(KE_UNKNOWN_SEMID);
            return true;
        }
        Semaphore &sema = it->second;
        switch (ordinal)
        {
        case 5: // DeleteSema: waiters wake up with KE_WAIT_DELETE
            for (auto &[threadId, thread] : m_threads)
                if (thread.state == IopThreadState::Semaphore && thread.waitId == id)
                {
                    thread.state = IopThreadState::Ready;
                    thread.waitId = 0;
                    thread.cpu.gpr[2] = static_cast<uint32_t>(KE_WAIT_DELETE);
                }
            m_semaphores.erase(it);
            setV0(0);
            return true;
        case 6: // SignalSema
        case 7: // iSignalSema
            if (waiters(id) != 0u)
            {
                ++sema.current; // handed straight to the woken waiter
                wakeOneSemaphore(id);
                setV0(0);
            }
            else if (sema.current >= sema.maximum)
                setV0(KE_SEMA_OVF);
            else
            {
                ++sema.current;
                setV0(0);
            }
            return true;
        case 8: // WaitSema
        case 9: // PollSema
            if (sema.current > 0)
            {
                --sema.current;
                setV0(0);
            }
            else if (ordinal == 9)
                setV0(KE_SEMA_ZERO);
            else if (m_currentThread != nullptr)
            {
                m_currentThread->state = IopThreadState::Semaphore;
                m_currentThread->waitId = id;
                cpu.yielded = true;
                setV0(0);
            }
            return true;
        default: // 11 ReferSemaStatus, 12 iReferSemaStatus
        {
            const uint32_t info = cpu.gpr[5];
            if (info != 0u)
            {
                m_memory.write32(info + 0u, sema.attr);
                m_memory.write32(info + 4u, sema.option);
                m_memory.write32(info + 8u, static_cast<uint32_t>(sema.initial));
                m_memory.write32(info + 12u, static_cast<uint32_t>(sema.maximum));
                m_memory.write32(info + 16u, static_cast<uint32_t>(sema.current));
                m_memory.write32(info + 20u, waiters(id));
            }
            setV0(0);
            return true;
        }
        }
    }

    void IopKernel::wakeOneSemaphore(int id)
    {
        IopThread *best = nullptr;
        for (auto &[threadId, thread] : m_threads)
        {
            if (thread.state != IopThreadState::Semaphore || thread.waitId != id)
                continue;
            if (best == nullptr || thread.priority < best->priority)
                best = &thread;
        }

        const auto semaphore = m_semaphores.find(id);
        if (best != nullptr && semaphore != m_semaphores.end() && semaphore->second.current > 0)
        {
            --semaphore->second.current;
            best->state = IopThreadState::Ready;
            best->waitId = 0;
        }
    }

    bool IopKernel::eventSatisfied(const EventFlag &event, uint32_t bits, uint32_t mode)
    {
        if (bits == 0u)
            return false;
        return (mode & 1u) != 0u ? (event.bits & bits) != 0u : (event.bits & bits) == bits;
    }

    void IopKernel::wakeEventWaiters(EventFlag &event)
    {
        for (auto &[threadId, thread] : m_threads)
        {
            if (thread.state != IopThreadState::EventFlag || thread.waitId != event.id)
                continue;
            if (!eventSatisfied(event, thread.waitBits, thread.waitMode))
                continue;

            if (thread.waitResultAddress != 0u)
                m_memory.write32(thread.waitResultAddress, event.bits);
            thread.cpu.gpr[2] = 0u;
            if ((thread.waitMode & 0x10u) != 0u)
                event.bits = 0u;
            thread.state = IopThreadState::Ready;
            thread.waitId = 0;
            thread.waitBits = 0;
            thread.waitMode = 0;
            thread.waitResultAddress = 0;
            if (event.bits == 0u)
                break;
        }
    }

    int IopKernel::createInternalEventFlag(uint32_t attr, uint32_t option, uint32_t bits)
    {
        EventFlag event;
        event.id = static_cast<int>(m_nextEventFlagId++);
        event.attr = attr;
        event.option = option;
        event.bits = bits;
        event.initBits = bits;
        const int id = event.id;
        m_eventFlags.emplace(id, event);
        return id;
    }

    bool IopKernel::setInternalEventFlag(int id, uint32_t bits)
    {
        const auto event = m_eventFlags.find(id);
        if (event == m_eventFlags.end())
            return false;
        event->second.bits |= bits;
        wakeEventWaiters(event->second);
        return true;
    }

    // thevent as the IOPRP300 THREADMAN implements it (see the note above dispatchSemaphoreImport).
    bool IopKernel::dispatchEventImport(uint16_t ordinal, IopCpuState &cpu)
    {
        const auto setV0 = [&](int32_t value)
        {
            cpu.gpr[2] = static_cast<uint32_t>(value);
        };
        const auto waiters = [&](int id)
        {
            uint32_t n = 0u;
            for (const auto &[threadId, thread] : m_threads)
                if (thread.state == IopThreadState::EventFlag && thread.waitId == id)
                    ++n;
            return n;
        };

        if (ordinal == 4) // CreateEventFlag(&{attr, option, bits})
        {
            const uint32_t descriptor = cpu.gpr[4];
            const uint32_t attr = m_memory.read32(descriptor + 0u);
            if ((attr & ~EA_MULTI) != 0u)
            {
                setV0(KE_ILLEGAL_ATTR);
                return true;
            }
            setV0(createInternalEventFlag(attr, m_memory.read32(descriptor + 4u), m_memory.read32(descriptor + 8u)));
            return true;
        }
        if (ordinal < 5 || ordinal > 14 || ordinal == 12)
            return false;

        const uint32_t bits = cpu.gpr[5];
        const uint32_t mode = cpu.gpr[6];
        if (ordinal == 10 || ordinal == 11) // argument checks come before the ID lookup
        {
            if ((mode & ~(WEF_OR | WEF_CLEAR)) != 0u)
            {
                setV0(KE_ILLEGAL_MODE);
                return true;
            }
            if (bits == 0u)
            {
                setV0(KE_EVF_ILPAT);
                return true;
            }
        }
        const int id = static_cast<int>(cpu.gpr[4]);
        const auto it = m_eventFlags.find(id);
        if (it == m_eventFlags.end())
        {
            setV0(KE_UNKNOWN_EVFID);
            return true;
        }
        EventFlag &event = it->second;
        switch (ordinal)
        {
        case 5: // DeleteEventFlag: waiters wake up with KE_WAIT_DELETE
            for (auto &[threadId, thread] : m_threads)
                if (thread.state == IopThreadState::EventFlag && thread.waitId == id)
                {
                    thread.state = IopThreadState::Ready;
                    thread.waitId = 0;
                    thread.cpu.gpr[2] = static_cast<uint32_t>(KE_WAIT_DELETE);
                }
            m_eventFlags.erase(it);
            setV0(0);
            return true;
        case 6: // SetEventFlag
        case 7: // iSetEventFlag
            if (bits != 0u)
            {
                event.bits |= bits;
                wakeEventWaiters(event);
            }
            setV0(0);
            return true;
        case 8: // ClearEventFlag: the argument is the mask of bits to keep
        case 9:
            event.bits &= bits;
            setV0(0);
            return true;
        case 10: // WaitEventFlag(id, bits, mode, &result)
        case 11: // PollEventFlag
        {
            if ((event.attr & EA_MULTI) == 0u && waiters(id) != 0u)
            {
                setV0(KE_EVF_MULTI);
                return true;
            }
            if (eventSatisfied(event, bits, mode))
            {
                if (cpu.gpr[7] != 0u)
                    m_memory.write32(cpu.gpr[7], event.bits);
                if ((mode & WEF_CLEAR) != 0u)
                    event.bits = 0u; // WEF_CLEAR clears every bit, not just the matched ones
                setV0(0);
            }
            else if (ordinal == 11)
                setV0(KE_EVF_COND);
            else if (m_currentThread != nullptr)
            {
                m_currentThread->state = IopThreadState::EventFlag;
                m_currentThread->waitId = event.id;
                m_currentThread->waitBits = bits;
                m_currentThread->waitMode = mode;
                m_currentThread->waitResultAddress = cpu.gpr[7];
                setV0(0);
                cpu.yielded = true;
            }
            else
                setV0(KE_EVF_COND);
            return true;
        }
        default: // 13 ReferEventFlagStatus, 14 iReferEventFlagStatus
            if (bits != 0u)
            {
                m_memory.write32(bits + 0u, event.attr);
                m_memory.write32(bits + 4u, event.option);
                m_memory.write32(bits + 8u, event.initBits);
                m_memory.write32(bits + 12u, event.bits);
                m_memory.write32(bits + 16u, waiters(id));
            }
            setV0(0);
            return true;
        }
    }

    void IopKernel::sleepCurrent(IopCpuState &cpu)
    {
        if (m_currentThread == nullptr)
            return;
        m_currentThread->state = IopThreadState::Sleep;
        cpu.yielded = true;
    }

    void IopKernel::delayCurrentUntil(uint64_t wakeCycle, IopCpuState &cpu)
    {
        if (m_currentThread == nullptr)
            return;
        m_currentThread->wakeCycle = wakeCycle;
        m_currentThread->state = IopThreadState::Delay;
        cpu.yielded = true;
    }

    IopThread *IopKernel::beginNextReady(uint64_t currentCycle)
    {
        for (auto &[id, thread] : m_threads)
        {
            if (thread.state == IopThreadState::Delay && thread.wakeCycle <= currentCycle)
                thread.state = IopThreadState::Ready;
        }

        IopThread *next = nullptr;
        for (auto &[id, thread] : m_threads)
        {
            if (thread.state != IopThreadState::Ready)
                continue;
            if (next == nullptr || thread.priority < next->priority ||
                (thread.priority == next->priority && thread.id < next->id))
                next = &thread;
        }
        if (next == nullptr)
            return nullptr;

        m_currentThread = next;
        next->state = IopThreadState::Running;
        next->cpu.stopped = false;
        next->cpu.yielded = false;
        return next;
    }

    uint64_t IopKernel::nextWakeCycle(uint64_t fallback) const
    {
        uint64_t nextWake = fallback;
        for (const auto &[id, thread] : m_threads)
        {
            if (thread.state == IopThreadState::Delay)
                nextWake = std::min(nextWake, thread.wakeCycle);
        }
        return nextWake;
    }

    std::vector<IopAlarm> IopKernel::takeDueAlarms(uint64_t currentCycle)
    {
        std::vector<IopAlarm> due;
        for (auto it = m_alarms.begin(); it != m_alarms.end();)
        {
            if (it->dueCycle <= currentCycle)
            {
                due.push_back(*it);
                it = m_alarms.erase(it);
            }
            else
                ++it;
        }
        std::sort(due.begin(), due.end(), [](const IopAlarm &a, const IopAlarm &b)
                  { return a.dueCycle < b.dueCycle; });
        return due;
    }

    void IopKernel::rearmAlarm(const IopAlarm &alarm)
    {
        m_alarms.push_back(alarm);
    }

    uint64_t IopKernel::nextAlarmCycle(uint64_t fallback) const
    {
        uint64_t next = fallback;
        for (const IopAlarm &alarm : m_alarms)
            next = std::min(next, alarm.dueCycle);
        return next;
    }

    void IopKernel::dumpThreads(uint64_t currentCycle) const
    {
        static constexpr const char *kStateNames[] = {
            "Dormant", "Ready", "Running", "Sleep", "Delay", "Semaphore", "EventFlag", "Suspended", "Dead"};
        std::fprintf(stderr, "[IOP threads] cycle=%llu count=%zu\n",
                     static_cast<unsigned long long>(currentCycle), m_threads.size());
        for (const auto &[id, thread] : m_threads)
        {
            const auto stateIndex = static_cast<size_t>(thread.state);
            const char *stateName = stateIndex < std::size(kStateNames) ? kStateNames[stateIndex] : "?";
            std::fprintf(stderr,
                         "  tid=%d state=%s waitId=%d waitBits=0x%x waitMode=0x%x wakeups=%d prio=0x%x entry=0x%x pc=0x%x ra=0x%x sp=0x%x\n",
                         id, stateName, thread.waitId, thread.waitBits, thread.waitMode, thread.wakeupCount,
                         thread.priority, thread.entry, thread.cpu.pc, thread.cpu.gpr[31], thread.cpu.gpr[29]);
        }
        for (const auto &[id, sema] : m_semaphores)
            std::fprintf(stderr, "  sema=%d count=%d/%d\n", id, sema.current, sema.maximum);
        for (const auto &[id, event] : m_eventFlags)
            std::fprintf(stderr, "  evf=%d bits=0x%x\n", id, event.bits);
    }

    void IopKernel::endTimeslice(IopThread &thread, uint32_t returnSentinel)
    {
        if (thread.cpu.pc == returnSentinel || thread.cpu.stopped)
            thread.state = IopThreadState::Dormant;
        else if (thread.state == IopThreadState::Running)
            thread.state = IopThreadState::Ready;
        m_currentThread = nullptr;
        cleanupDeadThreads();
    }

    void IopKernel::cleanupDeadThreads()
    {
        for (auto thread = m_threads.begin(); thread != m_threads.end();)
        {
            if (thread->second.state != IopThreadState::Dead)
            {
                ++thread;
                continue;
            }
            if (thread->second.stackBase != 0u)
                (void)m_memory.freeAllocation(thread->second.stackBase);
            thread = m_threads.erase(thread);
        }
    }

    void IopKernel::terminateThreadsInRange(uint32_t base, uint32_t size)
    {
        for (auto &[id, thread] : m_threads)
        {
            const uint32_t pc = IopMemory::physicalAddress(thread.cpu.pc);
            if (pc >= base && pc < base + size)
                thread.state = IopThreadState::Dead;
        }
    }
}
