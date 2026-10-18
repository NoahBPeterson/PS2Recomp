#ifndef PS2_MEMORY_H
#define PS2_MEMORY_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <array>
#include <functional>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <iostream>
#include <mutex>

#include "gs/ps2_gif_arbiter.h"
#include "ps2_vif1_source_map.h"
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(USE_SSE2NEON)
#include "sse2neon.h"
#else
#include <immintrin.h> // For SSE/AVX instructions
#include <smmintrin.h> // For SSE4.1 instructions
#endif

class GS;

#include "runtime/ps2_memory_map.h"

struct GSRegisters
{
    uint64_t pmode;    // Pixel mode
    uint64_t smode1;   // Sync mode 1
    uint64_t smode2;   // Sync mode 2
    uint64_t srfsh;    // Refresh control
    uint64_t synch1;   // Synchronization control 1
    uint64_t synch2;   // Synchronization control 2
    uint64_t syncv;    // Synchronization control V
    uint64_t dispfb1;  // Display buffer 1
    uint64_t display1; // Display area 1
    uint64_t dispfb2;  // Display buffer 2
    uint64_t display2; // Display area 2
    uint64_t extbuf;   // External buffer
    uint64_t extdata;  // External data
    uint64_t extwrite; // External write
    uint64_t bgcolor;  // Background color
    std::atomic<uint64_t> csr;
    std::atomic<uint64_t> vsyncTick;
    uint64_t imr;      // Interrupt mask
    uint64_t busdir;   // Bus direction
    uint64_t siglblid; // Signal label ID
};
static_assert(sizeof(GSRegisters) == (20u * sizeof(uint64_t)), "GSRegisters layout changed unexpectedly");
static_assert(alignof(GSRegisters) == alignof(uint64_t), "GSRegisters alignment must remain 64-bit");
static_assert(std::atomic<uint64_t>::is_always_lock_free, "GS CSR atomic must be lock-free on all supported targets");

// PS2 VIF (VPU Interface) registers
struct VIFRegisters
{
    uint32_t stat;   // Status
    uint32_t fbrst;  // VIF Force Break
    uint32_t err;    // Error status
    uint32_t mark;   // Interrupt control
    uint32_t cycle;  // Transfer mode
    uint32_t mode;   // Mode control
    uint32_t num;    // Data amount counter
    uint32_t mask;   // Data mask
    uint32_t code;   // VIFcode
    uint32_t itops;  // ITOP save
    uint32_t base;   // Base address
    uint32_t ofst;   // Offset
    uint32_t tops;   // TOPS
    uint32_t itop;   // ITOP
    uint32_t top;    // TOP
    uint32_t row[4]; // Transfer row data
    uint32_t col[4]; // Transfer column data
};
static_assert(sizeof(VIFRegisters) == (23u * sizeof(uint32_t)), "VIFRegisters layout changed unexpectedly");

// PS2 DMA registers
struct DMARegisters
{
    uint32_t chcr; // Channel control
    uint32_t madr; // Memory address
    uint32_t qwc;  // Quadword count
    uint32_t tadr; // Tag address
    uint32_t asr0; // Address stack 0
    uint32_t asr1; // Address stack 1
    uint32_t sadr; // Source address
};
static_assert(sizeof(DMARegisters) == (7u * sizeof(uint32_t)), "DMARegisters layout changed unexpectedly");

struct JumpTable
{
    uint32_t address = 0;          // Base address of the jump table
    uint32_t baseRegister = 0;     // Register used for index
    std::vector<uint32_t> targets; // Jump targets
};

class PS2Memory
{
public:
    PS2Memory();
    ~PS2Memory();

    PS2Memory(const PS2Memory &) = delete;
    PS2Memory &operator=(const PS2Memory &) = delete;
    PS2Memory(PS2Memory &&) = delete;
    PS2Memory &operator=(PS2Memory &&) = delete;

    // Initialize memory
    bool initialize(size_t ramSize = PS2_RAM_SIZE);

    // Memory access methods
    uint8_t *getRDRAM() { return m_rdram; }
    uint8_t *getScratchpad() { return m_scratchpad; }
    uint8_t *getIOPRAM() { return iop_ram; }
    uint64_t dmaStartCount() const { return m_dmaStartCount.load(std::memory_order_relaxed); }
    uint64_t gifCopyCount() const { return m_gifCopyCount.load(std::memory_order_relaxed); }
    uint64_t gsWriteCount() const { return m_gsWriteCount.load(std::memory_order_relaxed); }
    uint64_t vifWriteCount() const { return m_vifWriteCount.load(std::memory_order_relaxed); }
    uint64_t getVU0CodeGeneration() const { return m_vu0CodeGeneration.load(std::memory_order_relaxed); }
    uint64_t getVU1CodeGeneration() const { return m_vu1CodeGeneration.load(std::memory_order_relaxed); }

    // Read/write memory
    uint8_t read8(uint32_t address);
    uint16_t read16(uint32_t address);
    uint32_t read32(uint32_t address);
    uint64_t read64(uint32_t address);
    __m128i read128(uint32_t address);

    void write8(uint32_t address, uint8_t value);
    void write16(uint32_t address, uint16_t value);
    void write32(uint32_t address, uint32_t value);
    void write64(uint32_t address, uint64_t value);
    void write128(uint32_t address, __m128i value);

    // TLB handling
    uint32_t translateAddress(uint32_t virtualAddress);
    bool tlbRead(uint32_t index, uint32_t &vpn, uint32_t &pfn, uint32_t &mask, bool &valid) const;
    bool tlbWrite(uint32_t index, uint32_t vpn, uint32_t pfn, uint32_t mask, bool valid);
    int32_t tlbProbe(uint32_t vpn) const;
    size_t tlbEntryCount() const { return m_tlbEntries.size(); }

    // Hardware register interface
    bool writeIORegister(uint32_t address, uint32_t value);
    uint32_t readIORegister(uint32_t address);

    // EE timers advance from the scheduler's emulated EE-cycle clock. The
    // returned mask uses bits 0..3 for newly raised TIM0..TIM3 interrupts.
    uint32_t advanceEeTimers(uint64_t eeCycles) noexcept;
    [[nodiscard]] uint64_t cyclesUntilNextEeTimerInterrupt() const noexcept;
    void resetEeTimers() noexcept;

    using GifPacketCallback = std::function<void(const uint8_t *, uint32_t)>;
    void setGifPacketCallback(GifPacketCallback cb) { m_gifPacketCallback = std::move(cb); }
    void setGifArbiter(GifArbiter *arbiter) { m_gifArbiter = arbiter; }

    using Vu1MscalCallback = std::function<void(uint32_t startPC, uint32_t top, uint32_t itop)>;
    void setVu1MscalCallback(Vu1MscalCallback cb) { m_vu1MscalCallback = std::move(cb); }
    using Vu1MscntCallback = std::function<void(uint32_t top, uint32_t itop)>;
    void setVu1MscntCallback(Vu1MscntCallback cb) { m_vu1MscntCallback = std::move(cb); }

    // VIF1 timing (T-0122). With an EE clock and a deadline scheduler set (the runtime sets both),
    // a VIF1 DMA's completion (D_STAT, CHCR.STR, the DMAC cause) and its VIFcode i-bit interrupts
    // become visible at the EE cycle the VIF1/VU1 timing model reaches: transfer time per qword,
    // MSCAL/MSCNT/MPG/FLUSH* waiting for the running microprogram, which takes the VU1 cycles the
    // callbacks report through noteVu1RunCycles(). Without them (tests, tools) both are immediate.
    // VU memory, VIF1 registers and GS packets still change when the data is processed.
    using EeClockFn = std::function<uint64_t()>;
    using DeviceDeadlineFn = std::function<void(uint64_t eeCycle)>;
    void setDeviceTiming(EeClockFn clock, DeviceDeadlineFn scheduleDeadline);
    void noteVu1RunCycles(uint64_t cycles) { m_lastVu1RunCycles = cycles; }
    // Applies the timed device events due at eeCycle; the scheduler's deadline calls it.
    void applyDueDeviceEvents(uint64_t eeCycle);
    // EE cycle of the last queued device event, 0 if none is queued.
    uint64_t lastDeviceEventCycle() const { return m_deviceEvents.empty() ? 0u : m_deviceEvents.back().cycle; }

    void markVU0CodeModified() { m_vu0CodeGeneration.fetch_add(1, std::memory_order_relaxed); }
    void markVU1CodeModified() { m_vu1CodeGeneration.fetch_add(1, std::memory_order_relaxed); }
    uint8_t *getVU1Code() { return m_vu1Code; }
    const uint8_t *getVU1Code() const { return m_vu1Code; }
    uint8_t *getVU1Data() { return m_vu1Data; }
    const uint8_t *getVU1Data() const { return m_vu1Data; }
    uint8_t *getVU0Code() { return m_vu0Code; }
    const uint8_t *getVU0Code() const { return m_vu0Code; }
    uint8_t *getVU0Data() { return m_vu0Data; }
    const uint8_t *getVU0Data() const { return m_vu0Data; }

    bool isPath3Masked() const { return m_path3Masked; }
    void flushMaskedPath3Packets(bool drainImmediately = true);

    void submitGifPacket(GifPathId pathId, const uint8_t *data, uint32_t sizeBytes, bool drainImmediately = true, bool path2DirectHl = false);
    void forwardVif1DirectData(const uint8_t *data, uint32_t sizeBytes, bool directHl);
    void processGIFPacket(uint32_t srcPhysAddr, uint32_t qwCount);
    void processGIFPacket(const uint8_t *data, uint32_t sizeBytes);
    bool tryProcessNativeGifImageUploadChain(GS &gs, uint32_t tadr, uint32_t chcr);
    bool tryProcessNativeGifPackedChain(GS &gs, uint32_t tadr, uint32_t chcr);
    void processVIF0Data(uint32_t srcPhysAddr, uint32_t sizeBytes);
    void processVIF0Data(const uint8_t *data, uint32_t sizeBytes);
    void processVIF1Data(uint32_t srcPhysAddr, uint32_t sizeBytes);
    void processVIF1Data(const uint8_t *data, uint32_t sizeBytes);
    void processPendingTransfers();
    std::vector<uint32_t> consumeCompletedDmacCauses();
    std::vector<uint32_t> consumeCompletedIntcCauses();

    int pollDmaRegisters();

    // Track code modifications for self-modifying code
    void registerCodeRegion(uint32_t start, uint32_t end);
    bool isCodeAddress(uint32_t address) const;
    bool isCodeModified(uint32_t address, uint32_t size);
    void clearModifiedFlag(uint32_t address, uint32_t size);

    // GS register accessors
    GSRegisters &gs() { return gs_regs; }
    const GSRegisters &gs() const { return gs_regs; }
    uint8_t *getGSVRAM() { return m_gsVRAM; }
    const uint8_t *getGSVRAM() const { return m_gsVRAM; }
    bool hasSeenGifCopy() const { return m_seenGifCopy; }
    // Main RAM (32MB)
    uint8_t *m_rdram;

    // Scratchpad memory (16KB)
    uint8_t *m_scratchpad;

    // IOP RAM (2MB)
    uint8_t *iop_ram;

    bool m_seenGifCopy;
    std::atomic<uint64_t> m_dmaStartCount{0};
    std::atomic<uint64_t> m_gifCopyCount{0};
    std::atomic<uint64_t> m_gsWriteCount{0};
    std::atomic<uint64_t> m_vifWriteCount{0};
    std::atomic<uint64_t> m_vu0CodeGeneration{0};
    std::atomic<uint64_t> m_vu1CodeGeneration{0};
    // I/O registers
    std::unordered_map<uint32_t, uint32_t> m_ioRegisters;

    // Registers
    GSRegisters gs_regs;
    uint8_t *m_gsVRAM;
    VIFRegisters vif0_regs;
    VIFRegisters vif1_regs;
    DMARegisters dma_regs[10]; // 10 DMA channels

    // TLB entries
    struct TLBEntry
    {
        uint32_t vpn;
        uint32_t pfn;
        uint32_t mask;
        bool valid;
    };

    std::vector<TLBEntry> m_tlbEntries;

    GifPacketCallback m_gifPacketCallback;
    GifArbiter *m_gifArbiter = nullptr;
    Vu1MscalCallback m_vu1MscalCallback;
    Vu1MscntCallback m_vu1MscntCallback;

    // VIF1 timing (setDeviceTiming). m_vif1Clock: EE cycle the VIF1 model has reached; m_vu1BusyUntil:
    // EE cycle the last microprogram ends; m_vif1ClockBytes: transferred bytes not yet a whole cycle.
    struct TimedDeviceEvent
    {
        uint64_t cycle = 0u;
        bool vif1DmaComplete = false; // else INTC_VIF1 (a VIFcode i-bit)
    };
    static constexpr uint32_t kVif1BytesPerEeCycle = 8u; // one qword per bus cycle (EE clock / 2)
    EeClockFn m_eeClock;
    DeviceDeadlineFn m_scheduleDeviceDeadline;
    std::vector<TimedDeviceEvent> m_deviceEvents; // in cycle order
    uint64_t m_vif1Clock = 0u;
    uint64_t m_vu1BusyUntil = 0u;
    uint64_t m_lastVu1RunCycles = 0u;
    uint32_t m_vif1ClockBytes = 0u;
    uint32_t m_vif1CompletionsPending = 0u;
    bool deviceTimingEnabled() const { return m_eeClock && m_scheduleDeviceDeadline; }
    void beginVif1Work();
    void advanceVif1Clock(uint32_t bytes);
    void waitVif1ForVu1() { m_vif1Clock = std::max(m_vif1Clock, m_vu1BusyUntil); }
    void runVu1Program(bool resume, uint32_t startPC, uint32_t top, uint32_t itop);
    void finishVif1Dma();
    void raiseVif1Intc();
    void queueDeviceEvent(bool vif1DmaComplete);

    uint8_t *m_vu0Code = nullptr;
    uint8_t *m_vu0Data = nullptr;
    uint8_t *m_vu1Code = nullptr;
    uint8_t *m_vu1Data = nullptr;
    bool m_path3Masked = false;
    uint32_t m_vif1PendingPath2ImageQwc = 0u;
    bool m_vif1PendingPath2DirectHl = false;
    // Remaining data qwords of a VIF1 DIRECT/DIRECTHL cut off by the end of a processVIF1Data() buffer; the next
    // buffer starts with that raw data (no VIFcodes).
    uint32_t m_vif1PendingDirectQwc = 0u;
    bool m_vif1PendingDirectHl = false;
    std::vector<std::vector<uint8_t>> m_path3MaskedFifo;

    struct PendingTransfer
    {
        bool fromScratchpad = false;
        uint32_t srcAddr = 0;
        uint32_t qwc = 0;
        std::vector<uint8_t> chainData;
    };
    std::vector<PendingTransfer> m_pendingGifTransfers;
    std::vector<PendingTransfer> m_pendingVif0Transfers;
    std::vector<PendingTransfer> m_pendingVif1Transfers;
    std::mutex m_completedDmacMutex;
    std::vector<uint32_t> m_completedDmacCauses;
    std::vector<uint32_t> m_completedIntcCauses;
    void queueIntcCause(uint32_t cause);
    void completeVif1Dma();
    void resumeVIF1AfterStall();
    void walkDmaChain(uint32_t channelBase, uint32_t chcr, DmaChainCursor &cursor, std::vector<uint8_t> &chainBuf,
                      Vif1ChainSources *sources);
    // Bytes the last chain on each channel gathered (VIF0, VIF1, GIF): the next chain buffer
    // reserves that much instead of growing by repeated reallocation.
    std::array<size_t, 3> m_chainBytesHint{};
    // VIF1 i-bit stall: data behind the stall, and whether the VIF1 DMA
    // completion is held until the stall is released.
    bool m_vif1IrqStallPending = false;
    bool m_vif1Stalled = false;
    bool m_vif1DmaHeld = false;
    std::vector<uint8_t> m_vif1StalledData;
    Vif1SourceTracking m_vif1Sources; // guest addresses of m_vif1StalledData, re-read on resume

    struct CodeRegion
    {
        uint32_t start;
        uint32_t end;
        std::vector<bool> modified; // Bitmap of modified 4-byte blocks
    };
    std::vector<CodeRegion> m_codeRegions;

    bool isAddressInRegion(uint32_t address, const CodeRegion &region);
    void markModified(uint32_t address, uint32_t size);
    bool isScratchpad(uint32_t address) const;
    uint8_t *mapVuMemory(uint32_t physAddr, uint32_t size, uint32_t &offset, uint32_t &limit);
    const uint8_t *mapVuMemory(uint32_t physAddr, uint32_t size, uint32_t &offset, uint32_t &limit) const;
    struct EeTimer
    {
        uint32_t count = 0;
        uint32_t mode = 0;
        uint32_t compare = 0;
        uint32_t hold = 0;
        uint64_t clockRemainder = 0;
    };

    std::array<EeTimer, 4> m_eeTimers{};
    bool tryProcessScratchpadDma(uint32_t channelBase, uint32_t chcr);
    void completeDmacChannel(uint32_t channelBase, uint32_t cause);
    void queueCompletedDmacCause(uint32_t cause);
};

#endif // PS2_MEMORY_H
