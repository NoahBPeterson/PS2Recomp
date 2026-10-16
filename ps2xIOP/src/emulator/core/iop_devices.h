#pragma once

#include <cstdint>

namespace ps2x::iop::detail
{
    // IOP hardware below the CPU, for running Sony's own kernel and drivers (LLE): the IOP clock,
    // the interrupt controller and the six root counters. Register semantics follow PCSX2 2.8.2
    // (IopCounters.cpp, IopHwRead/Write.cpp), which tools/conformance checks differentially.
    // Time is in IOP cycles (36.864 MHz); the owner calls advanceTo() as the CPU runs.
    class IopDevices
    {
    public:
        static constexpr uint32_t kClockHz = 36'864'000u;
        // PCSX2 re-evaluates the counters at least every 768 cycles (its SPU2 tick); the pulsed
        // interrupt flag of a repeating counter returns high at that evaluation, so the cadence is
        // observable and part of the model.
        static constexpr uint64_t kUpdatePeriod = 768u;

        // Interrupt lines (I_STAT bits).
        enum Irq : unsigned
        {
            IrqVblankStart = 0,
            IrqCdvd = 2,
            IrqDma = 3,
            IrqTimer0 = 4,
            IrqTimer1 = 5,
            IrqTimer2 = 6,
            IrqSpu2 = 9,
            IrqVblankEnd = 11,
            IrqTimer3 = 14,
            IrqTimer4 = 15,
            IrqTimer5 = 16,
        };

        struct Counter
        {
            uint64_t count = 0;
            uint64_t target = 0; // bit 36: target disabled until the next overflow
            uint32_t rate = 1;   // IOP cycles per tick; kRateHblank counts hblanks
            uint32_t irqBit = 0;
            uint64_t startCycle = 0;
            uint32_t mode = 0;
            bool repeatIrq = false; // interrupt behaviour latched when the flags were last cleared
            bool toggleIrq = false;
        };
        static constexpr uint32_t kRateHblank = 0x2001u;
        static constexpr uint64_t kFutureTarget = 0x1000000000ull;

        IopDevices();
        void reset();

        [[nodiscard]] uint64_t cycle() const noexcept { return m_cycle; }
        // Runs every counter event up to and including `cycle`, then sets the clock to it.
        void advanceTo(uint64_t cycle);
        // Cycle of the next counter event (target/overflow/update), for idle time skipping.
        [[nodiscard]] uint64_t nextEventCycle() const noexcept { return m_nextEvent; }

        // Interrupt controller.
        void raiseIrq(unsigned line) noexcept { m_iStat |= 1u << line; }
        [[nodiscard]] uint32_t iStat() const noexcept { return m_iStat; }
        [[nodiscard]] uint32_t iMask() const noexcept { return m_iMask; }
        [[nodiscard]] uint32_t iCtrl() const noexcept { return m_iCtrl; }
        // The CPU's interrupt line (COP0 Cause.IP2).
        [[nodiscard]] bool interruptLine() const noexcept { return m_iCtrl != 0u && (m_iStat & m_iMask) != 0u; }

        // Video timing, driven by the EE side (or a test). VblankStart/End also raise IRQ 0/11.
        void hblankStart();
        void hblankEnd();
        void vblankStart();
        void vblankEnd();

        // Hardware register access (physical addresses). handles() says whether a device owns it.
        [[nodiscard]] static bool handles(uint32_t phys) noexcept;
        uint32_t read(uint32_t phys, unsigned bytes);
        void write(uint32_t phys, unsigned bytes, uint32_t value);

        [[nodiscard]] const Counter &counter(int i) const noexcept { return m_counters[i]; }

    private:
        bool canCount(int i) const noexcept;
        void sync(int i);
        void update();
        void scheduleNext();
        void scheduleCounter(int i);
        void fireInterrupt(int i, bool overflow);
        void testTarget(int i);
        void testOverflow(int i);
        void startGate(int i);
        void endGate(int i);
        void setNewIntrMode(int i);
        uint64_t readCount(int i);
        void writeCount(int i, uint32_t value);
        void writeMode(int i, uint32_t value);
        void writeTarget(int i, uint32_t value);

        uint64_t m_cycle = 0;
        uint64_t m_lastUpdate = 0;
        // PCSX2 also re-evaluates at its USB tick (PSXCLK/1000), with this bookkeeping.
        static constexpr uint64_t kUsbPeriod = kClockHz / 1000u;
        uint64_t m_usbStart = 0;
        uint64_t m_usbDelta = kUsbPeriod;
        uint64_t m_nextEvent = 0;
        bool m_hblanking = false;
        bool m_vblanking = false;
        uint32_t m_iStat = 0, m_iMask = 0, m_iCtrl = 0;
        Counter m_counters[6];
    };
}
