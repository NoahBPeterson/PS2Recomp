#ifndef PS2_VU1_H
#define PS2_VU1_H

#include <array>
#include <cstdint>
#include <cstring>

class GS;
class PS2Memory;

struct VU1State
{
    float vf[32][4];
    int32_t vi[16];
    float acc[4];
    float q;
    float p;
    float i;
    uint32_t r;
    uint32_t pc;
    uint32_t mac;
    uint32_t clip;
    uint32_t status;
    uint64_t cycles;
    bool ebit;
    bool haltAfterDelaySlot;
    bool dBitEnabled;
    bool tBitEnabled;
    bool stoppedByD;
    bool stoppedByT;
    uint32_t top;  // VIF TOP visible to XTOP
    uint32_t itop; // VIF ITOP visible to XITOP

    bool branchPending;
    uint32_t branchTarget;
    uint32_t branchDelay;
};

class VU1Interpreter
{
public:
    enum class Unit : uint8_t
    {
        VU0,
        VU1
    };

    explicit VU1Interpreter(Unit unit = Unit::VU1);

    void reset();

    void execute(uint8_t *vuCode, uint32_t codeSize,
                 uint8_t *vuData, uint32_t dataSize,
                 GS &gs, PS2Memory *memory = nullptr,
                 uint32_t startPC = 0, uint32_t top = 0, uint32_t itop = 0,
                 uint32_t maxCycles = 65536);

    void resume(uint8_t *vuCode, uint32_t codeSize,
                uint8_t *vuData, uint32_t dataSize,
                GS &gs, PS2Memory *memory = nullptr,
                uint32_t top = 0, uint32_t itop = 0, uint32_t maxCycles = 65536);

    // Replays (ps2x_vu1_replay): the state and cycle counter a recorded run started from.
    void restoreState(const VU1State &state)
    {
        m_state = state;
        m_cycle = state.cycles;
    }

    VU1State &state() { return m_state; }
    const VU1State &state() const { return m_state; }

private:
    enum Pipeline : uint8_t
    {
        PipelineNone = 0,
        PipelineFmac,
        PipelineLsu,
        PipelineFdiv,
        PipelineEfu,
        PipelineIalu,
        PipelineBranch,
        PipelineXgkick
    };

    struct VfAccess
    {
        uint8_t reg = 0;
        uint8_t lanes = 0;
    };

    struct InstructionUsage
    {
        std::array<VfAccess, 2> vfRead{};
        VfAccess vfWrite{};
        uint8_t vfReadCount = 0;
        uint16_t viRead = 0;
        uint16_t viWrite = 0;
        uint8_t accRead = 0;
        uint8_t accWrite = 0;
        uint8_t latency = 0;
        uint8_t vfLatency = 0;
        uint8_t viLatency = 0;
        Pipeline pipeline = PipelineNone;
        bool waitQ = false;
        bool waitP = false;
        bool readsClip = false;
        bool writesClip = false;
        bool delaysNextBranchRead = false;
        bool reserved = false;
    };

    struct DecodedInstructionPair
    {
        uint32_t lower = 0;
        uint32_t upper = 0;
        InstructionUsage lowerUsage{};
        InstructionUsage upperUsage{};
        bool iBit = false;
        bool eBit = false;
        bool mBit = false;
        bool dBit = false;
        bool tBit = false;
        uint8_t upperVfShadowReg = 0;
        uint8_t suppressedLowerVf = 0;

        // Flattened from the usages at decode time for the per-issue scoreboard work:
        // VF lanes read (reg * 4 + component), VI registers read, ACC components read.
        uint8_t vfReadLaneCount = 0;
        uint8_t viReadCount = 0;
        uint8_t accReadCount = 0;
        uint8_t vfReadLanes[16]{};
        uint8_t viReads[15]{};
        uint8_t accReads[4]{};
        // Lanes written (reg * 4 + component) and the latencies markPairWrites applies.
        uint8_t lowerVfWriteCount = 0;
        uint8_t upperVfWriteCount = 0;
        uint8_t viWriteCount = 0;
        uint8_t accWriteCount = 0;
        uint8_t lowerVfWriteLanes[4]{};
        uint8_t upperVfWriteLanes[4]{};
        uint8_t viWrites[15]{};
        uint8_t accWrites[4]{};
        uint8_t lowerVfWriteLatency = 0;
        uint8_t upperVfWriteLatency = 0;
        uint8_t viWriteLatency = 0;
    };

    struct FlagPipelineEntry
    {
        uint64_t readyCycle = 0;
        uint64_t issueCycle = 0;
        uint32_t mac = 0;
        uint32_t status = 0;
        uint32_t extraSticky = 0;
        uint32_t clip = 0;
        bool valid = false;
        bool writesMac = false;
        bool writesStatus = false;
        bool writesSticky = false;
        bool writesClip = false;
    };

    struct ScalarPipelineEntry
    {
        uint64_t readyCycle = 0;
        float value = 0.0f;
        uint32_t statusDi = 0;
        bool valid = false;
    };

    struct PendingStore
    {
        uint64_t readyCycle = 0;
        uint32_t address = 0;
        std::array<uint32_t, 4> words{};
        uint8_t laneMask = 0;
        bool valid = false;
    };

    struct PendingVfWrite
    {
        uint64_t readyCycle = 0;
        uint64_t sequence = 0;
        std::array<float, 4> value{};
        uint8_t reg = 0;
        uint8_t laneMask = 0;
        bool valid = false;
    };

    struct PendingViWrite
    {
        uint64_t readyCycle = 0;
        uint64_t sequence = 0;
        int32_t value = 0;
        uint8_t reg = 0;
        bool valid = false;
    };

    struct PendingAccWrite
    {
        uint64_t readyCycle = 0;
        uint64_t sequence = 0;
        std::array<float, 4> value{};
        uint8_t laneMask = 0;
        bool valid = false;
    };

    struct XgkickPipeline
    {
        static constexpr uint32_t kBufferSize = 0x10000u;
        std::array<uint8_t, kBufferSize> packet{};
        uint32_t sourceAddress = 0;
        uint32_t totalBytes = 0;
        uint32_t copiedBytes = 0;
        uint32_t currentTagEnd = 0;
        uint32_t cycleCredit = 0;
        uint64_t issueCycle = 0;
        bool active = false;
        bool currentTagEop = false;
    };

    static constexpr uint32_t kFmacLatency = 4u;
    // Cycles before the run budget in which VU1 queues VF/ACC writes instead of landing them at
    // issue (see run()); comfortably above twice the longest VF/ACC write latency (4).
    static constexpr uint32_t kImmediateWriteMargin = 16u;
    static constexpr uint32_t kAccForwardLatency = 1u;
    static constexpr uint32_t kMaxFlagEntries = 8u;
    static constexpr uint32_t kMaxPendingStores = 8u;
    static constexpr uint32_t kMaxPendingVfWrites = 16u;
    static constexpr uint32_t kMaxPendingViWrites = 8u;
    static constexpr uint32_t kMaxPendingAccWrites = 8u;
    static constexpr uint32_t kMaxDecodedPairs = 0x4000u / 8u;

    Unit m_unit;
    VU1State m_state;
    std::array<DecodedInstructionPair, kMaxDecodedPairs> m_decodedCodeCache{};
    const uint8_t *m_cachedVuCode = nullptr;
    const PS2Memory *m_cachedMemory = nullptr;
    uint32_t m_cachedCodeSize = 0;
    uint64_t m_cachedCodeGeneration = 0;
    bool m_decodedCodeCacheValid = false;

    // Flag writes (FMAC MAC/STATUS, CLIP, FSSET, FCSET) all become visible kFmacLatency cycles after
    // issue, at most one upper and one lower per cycle: slot (issue cycle % 8) holds them in issue
    // order, and the commit of cycle c takes slot (c - kFmacLatency).
    struct FlagSlot
    {
        uint64_t issueCycle = UINT64_MAX;
        uint8_t count = 0;
        std::array<FlagPipelineEntry, 2> entries{};
    };
    static constexpr uint32_t kFlagRingSize = 8u;
    std::array<FlagSlot, kFlagRingSize> m_flagRing{};
    uint64_t m_flagPendingUntil = 0;
    FlagPipelineEntry *allocFlagEntry(bool upper, uint32_t overflowCode);
    FlagSlot *currentFlagSlot();
    ScalarPipelineEntry m_fdiv{};
    std::array<ScalarPipelineEntry, 2> m_efu{};
    std::array<PendingStore, kMaxPendingStores> m_storePipeline{};
    std::array<PendingVfWrite, kMaxPendingVfWrites> m_vfWritePipeline{};
    std::array<PendingViWrite, kMaxPendingViWrites> m_viWritePipeline{};
    std::array<PendingAccWrite, kMaxPendingAccWrites> m_accWritePipeline{};
    // Bit i set <=> slot i of the matching pipeline array is valid (kept in step with .valid), so
    // the per-cycle commit and the slot searches touch only live entries, in slot order.
    uint32_t m_storeMask = 0;
    uint32_t m_vfWriteMask = 0;
    uint32_t m_viWriteMask = 0;
    uint32_t m_accWriteMask = 0;
    DecodedInstructionPair m_uncachedDecoded{};
    // Commit wheel: bucket (readyCycle % kWheelSize) lists the pipeline-array entries that become
    // visible at that cycle, as (kind << 5) | slot in allocation order. Every cycle passes through
    // commitReadyPipelines(), and array latencies are far below kWheelSize, so a bucket only ever
    // holds entries due at the current cycle.
    enum WheelKind : uint8_t
    {
        WheelStore = 1,
        WheelVf,
        WheelVi,
        WheelAcc
    };
    static constexpr uint32_t kWheelSize = 64u;
    struct WheelBucket
    {
        uint8_t count = 0;
        uint8_t items[kMaxFlagEntries + kMaxPendingStores + kMaxPendingVfWrites + kMaxPendingViWrites +
                      kMaxPendingAccWrites]{};
    };
    std::array<WheelBucket, kWheelSize> m_wheel{};
    // VU1 writes VF registers and ACC at issue instead of queueing them (see run()); these keep
    // pipelinesPending() true until the cycle the queued write would have committed.
    uint64_t m_vfPendingUntil = 0;
    uint64_t m_accPendingUntil = 0;
    void wheelPush(WheelKind kind, uint32_t slot, uint64_t readyCycle);
    XgkickPipeline m_xgkick{};
    std::array<std::array<uint64_t, 4>, 32> m_vfReady{};
    std::array<uint64_t, 16> m_viReady{};
    std::array<uint64_t, 4> m_accReady{};
    std::array<std::array<uint64_t, 4>, 32> m_vfLatestWrite{};
    std::array<uint64_t, 16> m_viLatestWrite{};
    std::array<uint64_t, 4> m_accLatestWrite{};

    uint64_t m_cycle = 0;
    uint64_t m_nextWriteSequence = 0;
    uint64_t m_efuResourceReady = 0;
    uint32_t m_workingClip = 0;
    uint32_t m_currentUpperInstruction = 0;
    int32_t m_viBranchBackupValue = 0;
    uint8_t m_viBranchBackupReg = 0;
    bool m_viBranchBackupValid = false;
    uint8_t *m_activeVuData = nullptr;
    uint32_t m_activeVuDataSize = 0;
    GS *m_activeGs = nullptr;
    PS2Memory *m_activeMemory = nullptr;
    bool m_stopRequested = false;
    bool m_pendingHaltD = false;
    bool m_pendingHaltT = false;

    void run(uint8_t *vuCode, uint32_t codeSize,
             uint8_t *vuData, uint32_t dataSize,
             GS &gs, PS2Memory *memory, uint32_t maxCycles);

    InstructionUsage decodeUpperUsage(uint32_t upper) const;
    InstructionUsage decodeLowerUsage(uint32_t lower) const;
    static void addVfRead(InstructionUsage &usage, uint8_t reg, uint8_t lanes);
    static void addVfWrite(InstructionUsage &usage, uint8_t reg, uint8_t lanes);
    static uint8_t vfReadLanes(const InstructionUsage &usage, uint8_t reg);
    DecodedInstructionPair decodeInstructionPair(const uint8_t *vuCode, uint32_t pc) const;
    const DecodedInstructionPair &getDecodedInstructionPairForPc(const uint8_t *vuCode, uint32_t codeSize, PS2Memory *memory, uint32_t pc);
    void rebuildDecodedCodeCache(const uint8_t *vuCode, uint32_t codeSize, const PS2Memory *memory, uint64_t generation);

    void execUpper(uint32_t instr);
    void execLower(uint32_t instr, uint8_t *vuData, uint32_t dataSize, GS &gs, PS2Memory *memory, uint32_t upperInstr);

    enum class FmacKind : uint8_t
    {
        Add,    // s + r
        Sub,    // s - r
        Madd,   // acc + s * r
        Msub,   // acc - s * r
        Mul,    // s * r
        Opmsub, // acc - cross(s, r), w = 0
        Opmula  // cross(s, r), w = 0
    };
    enum class FmacOperand : uint8_t
    {
        Broadcast, // vt[bc]
        Q,
        I,
        Vector // vt[c] (crossed for OPMSUB/OPMULA)
    };
    // One FMAC instruction: result, flags (from the exact double-precision value) and the
    // product-sum sticky flags in one pass; same results as computing the float result and
    // then applyFmacDest/applyFmacDestAcc.
    void execFmac(FmacKind kind, FmacOperand operand, uint8_t bcLane, bool toAcc, float *vd, uint8_t dest,
                  const float *vs, const float *vt, const float *acc, float q, float i);

    void applyDest(float *dst, const float *result, uint8_t dest);
    void applyDestAcc(const float *result, uint8_t dest);
    uint8_t normalizeFmacExactResult(float &value, long double exactResult) const;
    void updateFmacFlags(const uint8_t laneFlags[4], uint8_t dest, uint32_t extraSticky);
    void queueFsset(uint16_t immediate);
    void queueClip(uint32_t clip);
    void queueFcset(uint32_t clip);
    void queueQ(float value, uint32_t latency, uint32_t statusDi);
    void queueP(float value, uint32_t latency);
    void queueStore(uint32_t address, const uint32_t words[4], uint8_t laneMask);
    void queueVfWrite(uint8_t reg, uint8_t laneMask, const float value[4], uint32_t latency);
    void queueViWrite(uint8_t reg, int32_t value, uint32_t latency);
    void queueAccWrite(uint8_t laneMask, const float value[4], uint32_t latency);
    void startXgkick(uint32_t qwordAddress);

    void resetScheduler();
    void commitReadyPipelines();
    void advanceOneCycle();
    void advanceTo(uint64_t targetCycle);
    void flushPipelines();
    void progressXgkick();
    void finishXgkick();
    uint64_t calculatePairReadyCycle(const DecodedInstructionPair &decoded) const;
    void markPairWrites(const DecodedInstructionPair &decoded);
    bool pipelinesPending() const;

    // Denormals read as signed zero, Inf/NaN as signed max (VU operands have no IEEE specials).
    float normalizeOperand(float value) const
    {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        const uint32_t sign = bits & 0x80000000u;
        const uint32_t exponent = bits & 0x7F800000u;
        bits = exponent == 0u ? sign : (exponent == 0x7F800000u ? (sign | 0x7F7FFFFFu) : bits);
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }
    float normalizeResult(float value, uint32_t &laneFlags) const;
    uint32_t microAddressMask() const;
    int32_t readBranchVi(uint8_t reg) const;
    void recordViWriteForBranch(uint8_t reg, int32_t oldValue);
    void reportReservedInstruction(bool upper, uint32_t instruction);
    float broadcast(const float *vf, uint8_t bc);
};

#endif
