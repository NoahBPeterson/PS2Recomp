#include "iop_devices.h"

#include <algorithm>

namespace ps2x::iop::detail
{
    namespace
    {
        constexpr uint32_t kGateEnable = 1u << 0;
        constexpr uint32_t kZeroReturn = 1u << 3;
        constexpr uint32_t kTargetIntr = 1u << 4;
        constexpr uint32_t kOverflowIntr = 1u << 5;
        constexpr uint32_t kRepeatIntr = 1u << 6;
        constexpr uint32_t kToggleIntr = 1u << 7;
        constexpr uint32_t kExtSignal = 1u << 8;
        constexpr uint32_t kT2Prescale = 1u << 9;
        constexpr uint32_t kIntrEnable = 1u << 10; // active low: 0 = interrupt requested
        constexpr uint32_t kTargetFlag = 1u << 11;
        constexpr uint32_t kOverflowFlag = 1u << 12;
        constexpr uint32_t kStopped = 1u << 15;
        constexpr uint32_t kModeWriteMask = 0x63FFu;
        constexpr uint32_t kModeFlagMask = 0x1C00u;
        constexpr uint32_t kRatePixel = 3u;

        constexpr unsigned gateMode(uint32_t mode) { return (mode >> 1) & 3u; }
        constexpr unsigned t45Prescale(uint32_t mode) { return (mode >> 13) & 3u; }
        constexpr uint64_t maxCount(int i) { return i < 3 ? 0xFFFFull : 0xFFFFFFFFull; }

        constexpr uint32_t kIntcStat = 0x1F801070u;
        constexpr uint32_t kIntcMask = 0x1F801074u;
        constexpr uint32_t kIntcCtrl = 0x1F801078u;

        // Counter register decode: index and offset within the counter's 16-byte block, or -1.
        int counterIndex(uint32_t phys, uint32_t &offset)
        {
            if (phys >= 0x1F801100u && phys < 0x1F801130u)
            {
                offset = phys & 0xFu;
                return static_cast<int>((phys >> 4) & 0xFu);
            }
            if (phys >= 0x1F801480u && phys < 0x1F8014B0u)
            {
                offset = phys & 0xFu;
                return static_cast<int>((phys >> 4) & 0xFu) - 5;
            }
            return -1;
        }
    }

    IopDevices::IopDevices() { reset(); }

    void IopDevices::reset()
    {
        m_cycle = 0;
        m_lastUpdate = 0;
        m_usbStart = 0;
        m_usbDelta = kUsbPeriod;
        m_hblanking = m_vblanking = false;
        m_iStat = m_iMask = m_iCtrl = 0;
        static constexpr uint32_t kIrqBits[6] = {1u << IrqTimer0, 1u << IrqTimer1, 1u << IrqTimer2,
                                                  1u << IrqTimer3, 1u << IrqTimer4, 1u << IrqTimer5};
        for (int i = 0; i < 6; ++i)
        {
            m_counters[i] = Counter{};
            m_counters[i].rate = 1;
            m_counters[i].mode = kIntrEnable;
            m_counters[i].target = kFutureTarget;
            m_counters[i].irqBit = kIrqBits[i];
            m_counters[i].startCycle = m_cycle;
        }
        m_nextEvent = m_cycle + 1; // first evaluation right away, as PCSX2's psxRcntInit asks
    }

    void IopDevices::advanceTo(uint64_t cycle)
    {
        while (m_nextEvent <= cycle)
        {
            m_cycle = std::max(m_cycle, m_nextEvent);
            update();
        }
        m_cycle = std::max(m_cycle, cycle);
    }

    bool IopDevices::canCount(int i) const noexcept
    {
        const Counter &c = m_counters[i];
        if (c.mode & kStopped)
            return false;
        if (!(c.mode & kGateEnable))
            return true;
        const unsigned gm = gateMode(c.mode);
        if (i == 2 || i == 4 || i == 5)
            return (gm & 1u) != 0u; // these gates only switch the counter on or off
        const bool blanking = i == 0 ? m_hblanking : m_vblanking;
        if ((gm == 0u && blanking) || (gm == 2u && !blanking))
            return false;
        return true;
    }

    void IopDevices::sync(int i)
    {
        Counter &c = m_counters[i];
        if (c.repeatIrq && !c.toggleIrq)
            c.mode |= kIntrEnable; // pulsed: the request line returns high shortly after
        if (canCount(i) && c.rate != kRateHblank)
        {
            const uint64_t change = (m_cycle - c.startCycle) / c.rate;
            if (change > 0)
            {
                c.count += change;
                c.startCycle += change * c.rate;
                c.startCycle &= ~(static_cast<uint64_t>(c.rate) - 1u);
            }
        }
        else
        {
            if ((c.mode & kGateEnable) && gateMode(c.mode) == 2u)
                c.count = 0;
            c.startCycle = m_cycle;
        }
    }

    void IopDevices::scheduleNext()
    {
        for (int i = 0; i < 6; ++i)
            scheduleCounter(i);
    }

    // PCSX2's _rcntSet(i): only counter i's events are (re)considered.
    void IopDevices::scheduleCounter(int i)
    {
        {
            const Counter &c = m_counters[i];
            if (c.rate == kRateHblank || !canCount(i))
                return;
            const uint64_t overflowCap = maxCount(i) + 1u;
            if (c.count > overflowCap || c.count > c.target)
            {
                // Just passed (missed between evaluations): PCSX2 re-evaluates 4 cycles after its
                // last evaluation, replacing whatever was scheduled.
                m_nextEvent = m_lastUpdate + 4u;
                m_nextEvent = std::max(m_nextEvent, m_lastUpdate + 1u);
                return;
            }
            // PCSX2 measures the time to an event as an unsigned offset from its last evaluation:
            // an event already in the past wraps to a huge offset and is not scheduled.
            const auto consider = [&](uint64_t at) {
                if (at >= m_lastUpdate)
                    m_nextEvent = std::min(m_nextEvent, at);
            };
            consider(c.startCycle + (overflowCap - c.count) * c.rate);
            if (c.target & kFutureTarget)
                return;
            consider(c.startCycle + (c.target - c.count) * c.rate);
        }
        // At most one evaluation per cycle; one already due runs at the current cycle.
        m_nextEvent = std::max(m_nextEvent, m_lastUpdate + 1u);
    }

    void IopDevices::update()
    {
        m_lastUpdate = m_cycle;
        for (int i = 0; i < 6; ++i)
        {
            sync(i);
            if (m_counters[i].rate == kRateHblank || !canCount(i))
                continue;
            testOverflow(i);
            testTarget(i);
        }
        m_nextEvent = m_cycle + kUpdatePeriod;
        const uint64_t sinceUsb = m_cycle - m_usbStart;
        uint64_t usbRemaining = m_usbDelta;
        if (sinceUsb >= m_usbDelta)
        {
            m_usbStart += kUsbPeriod * (sinceUsb / kUsbPeriod);
            m_usbDelta = kUsbPeriod;
        }
        else
            usbRemaining -= sinceUsb;
        m_nextEvent = std::min(m_nextEvent, m_cycle + usbRemaining);
        scheduleNext();
    }

    void IopDevices::fireInterrupt(int i, bool overflow)
    {
        Counter &c = m_counters[i];
        bool updateIntr = c.repeatIrq;
        if (c.mode & kIntrEnable)
        {
            const bool alreadySet = (c.mode & (overflow ? kOverflowFlag : kTargetFlag)) != 0u;
            if (updateIntr || !alreadySet)
                m_iStat |= c.irqBit;
            updateIntr |= c.toggleIrq;
        }
        if (updateIntr)
        {
            if (c.toggleIrq)
                c.mode ^= kIntrEnable;
            else
                c.mode &= ~kIntrEnable;
        }
    }

    void IopDevices::testTarget(int i)
    {
        Counter &c = m_counters[i];
        if (c.count < c.target)
            return;
        if (c.mode & kTargetIntr)
            fireInterrupt(i, false);
        c.mode |= kTargetFlag;
        if (c.mode & kZeroReturn)
            c.count -= c.target;
        else
            c.target |= kFutureTarget;
    }

    void IopDevices::testOverflow(int i)
    {
        Counter &c = m_counters[i];
        if (c.count <= maxCount(i))
            return;
        if (c.mode & kOverflowIntr)
            fireInterrupt(i, true);
        c.mode |= kOverflowFlag;
        c.count -= maxCount(i) + 1u;
        c.target &= maxCount(i);
    }

    void IopDevices::startGate(int i)
    {
        Counter &c = m_counters[i];
        if (!(c.mode & kGateEnable))
            return;
        switch (gateMode(c.mode))
        {
        case 0: // counts while the gate signal is low: freeze the count now
            c.count = readCount(i);
            c.startCycle = m_cycle & ~(static_cast<uint64_t>(c.rate) - 1u);
            break;
        case 2: // counts only during blanking, cleared at both ends
            sync(i);
            c.mode &= ~kStopped;
            c.count = 0;
            c.target &= ~kFutureTarget;
            break;
        default:
            break;
        }
    }

    void IopDevices::endGate(int i)
    {
        Counter &c = m_counters[i];
        if (!(c.mode & kGateEnable))
            return;
        switch (gateMode(c.mode))
        {
        case 0:
            c.startCycle = m_cycle & ~(static_cast<uint64_t>(c.rate) - 1u);
            break;
        case 1: // counts continuously, cleared at the end of blanking
        case 2:
            sync(i);
            c.count = 0;
            c.target &= ~kFutureTarget;
            break;
        case 3: // starts at the end of the next blanking
            if (c.mode & kStopped)
            {
                c.startCycle = m_cycle & ~(static_cast<uint64_t>(c.rate) - 1u);
                c.mode &= ~kStopped;
            }
            break;
        }
    }

    void IopDevices::hblankStart()
    {
        for (int i : {1, 3})
        {
            if (m_counters[i].rate == kRateHblank && canCount(i))
            {
                ++m_counters[i].count;
                testOverflow(i);
                testTarget(i);
            }
        }
        startGate(0);
        m_hblanking = true;
        scheduleCounter(0);
    }

    void IopDevices::hblankEnd()
    {
        endGate(0);
        m_hblanking = false;
        scheduleCounter(0);
    }

    void IopDevices::vblankStart()
    {
        raiseIrq(IrqVblankStart);
        startGate(1);
        startGate(3);
        m_vblanking = true;
        scheduleCounter(1);
        scheduleCounter(3);
    }

    void IopDevices::vblankEnd()
    {
        raiseIrq(IrqVblankEnd);
        endGate(1);
        endGate(3);
        m_vblanking = false;
        scheduleCounter(1);
        scheduleCounter(3);
    }

    void IopDevices::setNewIntrMode(int i)
    {
        Counter &c = m_counters[i];
        c.mode &= ~(kTargetFlag | kOverflowFlag);
        c.mode |= kIntrEnable;
        c.repeatIrq = (c.mode & kRepeatIntr) != 0u;
        c.toggleIrq = (c.mode & kToggleIntr) != 0u;
    }

    uint64_t IopDevices::readCount(int i)
    {
        sync(i);
        return m_counters[i].count;
    }

    void IopDevices::writeCount(int i, uint32_t value)
    {
        Counter &c = m_counters[i];
        sync(i);
        c.count = value & maxCount(i);
        c.target &= maxCount(i);
        if (c.count > c.target)
            c.target |= kFutureTarget;
        scheduleCounter(i);
    }

    void IopDevices::writeMode(int i, uint32_t value)
    {
        Counter &c = m_counters[i];
        const uint32_t old = c.mode;
        c.mode = (value & kModeWriteMask) | (c.mode & kModeFlagMask);
        if (!((old & (kTargetFlag | kOverflowFlag)) && (old & (kTargetIntr | kOverflowIntr))))
            setNewIntrMode(i);
        if (i == 2)
        {
            c.rate = (c.mode & kT2Prescale) ? 8u : 1u;
        }
        else if (i < 3 || i == 3)
        {
            c.rate = 1u;
            if (c.mode & kExtSignal)
                c.rate = i == 0 ? kRatePixel : kRateHblank;
            if ((c.mode & kGateEnable) && gateMode(c.mode) >= 2u)
                c.mode |= kStopped;
        }
        else
        {
            static constexpr uint32_t kPrescale[4] = {1u, 8u, 16u, 256u};
            c.rate = kPrescale[t45Prescale(c.mode)];
        }
        c.count = 0;
        c.startCycle = m_cycle & ~(static_cast<uint64_t>(c.rate) - 1u);
        c.target &= maxCount(i);
        scheduleCounter(i);
    }

    void IopDevices::writeTarget(int i, uint32_t value)
    {
        Counter &c = m_counters[i];
        c.target = value & maxCount(i);
        sync(i);
        if (c.target <= c.count)
            c.target |= kFutureTarget;
        scheduleCounter(i);
    }

    bool IopDevices::handles(uint32_t phys) noexcept
    {
        uint32_t offset = 0;
        if (counterIndex(phys, offset) >= 0)
        {
            if (phys < 0x1F801480u)
                return offset == 0x0u || offset == 0x4u || offset == 0x8u;
            return offset == 0x0u || offset == 0x2u || offset == 0x4u || offset == 0x8u || offset == 0xAu;
        }
        return phys >= kIntcStat && phys < kIntcCtrl + 4u;
    }

    uint32_t IopDevices::read(uint32_t phys, unsigned bytes)
    {
        uint32_t offset = 0;
        const int i = counterIndex(phys, offset);
        if (i >= 0)
        {
            switch (offset)
            {
            case 0x0:
                return static_cast<uint32_t>(readCount(i));
            case 0x2:
                return static_cast<uint32_t>(readCount(i) >> 16);
            case 0x4:
            {
                const uint32_t mode = m_counters[i].mode;
                setNewIntrMode(i); // reading the mode clears the target/overflow flags
                return mode;
            }
            case 0x8:
                return static_cast<uint32_t>(m_counters[i].target);
            case 0xA:
                return static_cast<uint32_t>(m_counters[i].target >> 16);
            default:
                return 0u;
            }
        }
        const uint32_t base = phys & ~3u;
        const unsigned shift = (phys & 3u) * 8u;
        uint32_t value = 0u;
        if (base == kIntcStat)
            value = m_iStat;
        else if (base == kIntcMask)
            value = m_iMask;
        else if (base == kIntcCtrl)
        {
            value = m_iCtrl;
            m_iCtrl = 0u; // reading I_CTRL disables interrupts (INTRMAN's CpuSuspendIntr relies on it)
        }
        (void)bytes;
        return value >> shift;
    }

    void IopDevices::write(uint32_t phys, unsigned bytes, uint32_t value)
    {
        uint32_t offset = 0;
        const int i = counterIndex(phys, offset);
        if (i >= 0)
        {
            switch (offset)
            {
            case 0x0:
                writeCount(i, value);
                break;
            case 0x2:
                writeCount(i, value << 16);
                break;
            case 0x4:
                writeMode(i, value);
                break;
            case 0x8:
                writeTarget(i, value);
                break;
            case 0xA:
                writeTarget(i, value << 16);
                break;
            default:
                break;
            }
            return;
        }
        const uint32_t base = phys & ~3u;
        const unsigned shift = (phys & 3u) * 8u;
        const uint32_t laneMask = (bytes >= 4u ? 0xFFFFFFFFu : ((1u << (bytes * 8u)) - 1u)) << shift;
        const uint32_t lanes = (value << shift) & laneMask;
        if (base == kIntcStat)
            m_iStat &= lanes | ~laneMask; // write 0 to acknowledge
        else if (base == kIntcMask)
            m_iMask = (m_iMask & ~laneMask) | lanes;
        else if (base == kIntcCtrl)
            m_iCtrl = (m_iCtrl & ~laneMask) | lanes;
    }
}
