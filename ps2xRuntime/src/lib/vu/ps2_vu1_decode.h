#ifndef PS2_VU1_DECODE_H
#define PS2_VU1_DECODE_H
// VU instruction decoding (scoreboard usage, pair flags, FMAC handler selection), constexpr so
// translated VU1 blocks (T-0121) decode their instruction words at compile time with exactly the
// interpreter's decoder.

#include "ps2_vu1_detail.h"

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define PS2X_VU_FMAC_NEON 1
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <emmintrin.h>
#define PS2X_VU_FMAC_SSE2 1
#endif

// FMAC MADD/MSUB/OPMSUB round twice on every platform, like the VU (and PCSX2): the product is
// chopped (and flushed/clamped like any result), then the add. Every path writes the two steps
// out; the build turns FP contraction off so the compiler does not fuse the scalar ones.

constexpr void VU1Interpreter::addVfRead(InstructionUsage &usage, uint8_t reg, uint8_t lanes)
{
    if (lanes == 0u)
        return;
    for (uint32_t index = 0; index < usage.vfReadCount; ++index)
    {
        if (usage.vfRead[index].reg == reg)
        {
            usage.vfRead[index].lanes |= lanes;
            return;
        }
    }
    if (usage.vfReadCount < usage.vfRead.size())
        usage.vfRead[usage.vfReadCount++] = {reg, lanes};
}

constexpr void VU1Interpreter::addVfWrite(InstructionUsage &usage, uint8_t reg, uint8_t lanes)
{
    if (reg == 0u || lanes == 0u)
        return;
    if (usage.vfWrite.reg == 0u)
        usage.vfWrite = {reg, lanes};
    else if (usage.vfWrite.reg == reg)
        usage.vfWrite.lanes |= lanes;
}

constexpr uint8_t VU1Interpreter::vfReadLanes(const InstructionUsage &usage, uint8_t reg)
{
    for (uint32_t index = 0; index < usage.vfReadCount; ++index)
    {
        if (usage.vfRead[index].reg == reg)
            return usage.vfRead[index].lanes;
    }
    return 0u;
}

constexpr VU1Interpreter::InstructionUsage VU1Interpreter::decodeUpperUsage(uint32_t upper)
{
    InstructionUsage usage;
    usage.pipeline = PipelineFmac;
    usage.latency = kFmacLatency;

    const uint8_t op = static_cast<uint8_t>(upper & 0x3Fu);
    const uint8_t dest = DEST(upper);
    const uint8_t fs = FS(upper);
    const uint8_t ft = FT(upper);
    const uint8_t fd = FD(upper);

    if (op <= 0x2Fu)
    {
        addVfRead(usage, fs, dest);
        addVfWrite(usage, fd, dest);
        if (op <= 0x1Bu)
            addVfRead(usage, ft, laneForComponent(op & 3u));
        else if (op >= 0x28u)
            addVfRead(usage, ft, op == 0x2Eu ? 0xEu : dest);
        if (op == 0x08u || op == 0x09u || op == 0x0Au || op == 0x0Bu ||
            op == 0x0Cu || op == 0x0Du || op == 0x0Eu || op == 0x0Fu ||
            op == 0x21u || op == 0x23u || op == 0x25u || op == 0x27u ||
            op == 0x29u || op == 0x2Du || op == 0x2Eu)
        {
            usage.accRead = dest;
        }
        return usage;
    }

    if (op >= 0x3Cu)
    {
        const uint8_t special = static_cast<uint8_t>((upper & 3u) | ((upper >> 4) & 0x7Cu));
        const bool writesAcc =
            special <= 0x0Fu ||
            (special >= 0x18u && special <= 0x1Cu) ||
            special == 0x1Eu ||
            (special >= 0x20u && special <= 0x2Au) ||
            (special >= 0x2Cu && special <= 0x2Eu);
        if (writesAcc)
        {
            addVfRead(usage, fs, dest);
            if (special <= 0x1Bu)
                addVfRead(usage, ft, laneForComponent(special & 3u));
            else if ((special >= 0x28u && special <= 0x2Eu))
                addVfRead(usage, ft, special == 0x2Eu ? 0xEu : dest);
            usage.accWrite = dest;
            if ((special >= 0x08u && special <= 0x0Fu) ||
                special == 0x21u || special == 0x23u || special == 0x25u ||
                special == 0x27u || special == 0x29u || special == 0x2Du)
            {
                usage.accRead = dest;
            }
        }
        else if (special >= 0x10u && special <= 0x17u)
        {
            addVfRead(usage, fs, dest);
            addVfWrite(usage, ft, dest);
        }
        else if (special == 0x1Du)
        {
            addVfRead(usage, fs, dest);
            addVfWrite(usage, ft, dest);
        }
        else if (special == 0x1Fu)
        {
            addVfRead(usage, fs, 0xEu);
            addVfRead(usage, ft, 0x1u);
            usage.writesClip = true;
        }
        else if (special != 0x2Fu && special != 0x30u)
        {
            usage.reserved = true;
        }
        return usage;
    }

    usage.reserved = true;
    return usage;
}

constexpr VU1Interpreter::InstructionUsage VU1Interpreter::decodeLowerUsage(uint32_t lower, Unit unit)
{
    InstructionUsage usage;
    if (lower == 0u || lower == 0x8000033Cu)
        return usage;

    const uint8_t opHi = static_cast<uint8_t>((lower >> 25) & 0x7Fu);
    const uint8_t vfT = FT(lower);
    const uint8_t vfS = FS(lower);
    const uint8_t viT = VIT(lower);
    const uint8_t viS = VIS(lower);
    const uint8_t viD = VID(lower);
    const uint8_t dest = DEST(lower);
    auto readVi = [&](uint8_t reg)
    {
        if (reg != 0u)
            usage.viRead |= static_cast<uint16_t>(1u << reg);
    };
    auto writeVi = [&](uint8_t reg)
    {
        if (reg != 0u)
            usage.viWrite |= static_cast<uint16_t>(1u << reg);
    };

    switch (opHi)
    {
    case 0x00:
        usage.pipeline = PipelineLsu;
        usage.latency = 4u;
        readVi(viS);
        addVfWrite(usage, vfT, dest);
        return usage;
    case 0x01:
        usage.pipeline = PipelineLsu;
        usage.latency = 1u;
        readVi(viT);
        addVfRead(usage, vfS, dest);
        return usage;
    case 0x04:
        usage.pipeline = PipelineLsu;
        usage.latency = 4u;
        readVi(viS);
        writeVi(viT);
        return usage;
    case 0x05:
        usage.pipeline = PipelineLsu;
        usage.latency = 1u;
        readVi(viS);
        readVi(viT);
        return usage;
    case 0x08:
    case 0x09:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viS);
        writeVi(viT);
        return usage;
    case 0x10:
    case 0x12:
    case 0x13:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.readsClip = true;
        writeVi(1u);
        return usage;
    case 0x11:
        usage.pipeline = PipelineFmac;
        usage.latency = kFmacLatency;
        usage.writesClip = true;
        return usage;
    case 0x14:
    case 0x16:
    case 0x17:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        writeVi(viT);
        return usage;
    case 0x15:
        usage.pipeline = PipelineFmac;
        usage.latency = kFmacLatency;
        return usage;
    case 0x18:
    case 0x1A:
    case 0x1B:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        readVi(viS);
        writeVi(viT);
        return usage;
    case 0x1C:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.readsClip = true;
        writeVi(viT);
        return usage;
    case 0x20:
        usage.pipeline = PipelineBranch;
        return usage;
    case 0x21:
        usage.pipeline = PipelineBranch;
        usage.latency = 1u;
        writeVi(viT);
        return usage;
    case 0x24:
        usage.pipeline = PipelineBranch;
        readVi(viS);
        return usage;
    case 0x25:
        usage.pipeline = PipelineBranch;
        usage.latency = 1u;
        readVi(viS);
        writeVi(viT);
        return usage;
    case 0x28:
    case 0x29:
        usage.pipeline = PipelineBranch;
        readVi(viS);
        readVi(viT);
        return usage;
    case 0x2C:
    case 0x2D:
    case 0x2E:
    case 0x2F:
        usage.pipeline = PipelineBranch;
        readVi(viS);
        return usage;
    case 0x40:
        break;
    default:
        usage.reserved = true;
        return usage;
    }

    const uint8_t direct = static_cast<uint8_t>(lower & 0x3Fu);
    if (direct == 0x30u || direct == 0x31u || direct == 0x34u || direct == 0x35u)
    {
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viS);
        readVi(viT);
        writeVi(viD);
        return usage;
    }
    if (direct == 0x32u)
    {
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viS);
        writeVi(viT);
        return usage;
    }
    if (direct < 0x3Cu)
    {
        usage.reserved = true;
        return usage;
    }

    const uint8_t special = static_cast<uint8_t>((lower & 3u) | ((lower >> 4) & 0x7Cu));
    switch (special)
    {
    case 0x30:
    case 0x31:
        usage.pipeline = PipelineFmac;
        usage.latency = 4u;
        addVfRead(usage, vfS, special == 0x31u ? 0xFu : dest);
        addVfWrite(usage, vfT, dest);
        break;
    case 0x34:
    case 0x36:
        usage.pipeline = PipelineLsu;
        usage.latency = 4u;
        usage.viLatency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viS);
        writeVi(viS);
        addVfWrite(usage, vfT, dest);
        break;
    case 0x35:
    case 0x37:
        usage.pipeline = PipelineLsu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viT);
        writeVi(viT);
        addVfRead(usage, vfS, dest);
        break;
    case 0x38:
        usage.pipeline = PipelineFdiv;
        usage.latency = 7u;
        addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        addVfRead(usage, vfT, laneForComponent((lower >> 23) & 3u));
        break;
    case 0x39:
        usage.pipeline = PipelineFdiv;
        usage.latency = 7u;
        addVfRead(usage, vfT, laneForComponent((lower >> 23) & 3u));
        break;
    case 0x3A:
        usage.pipeline = PipelineFdiv;
        usage.latency = 13u;
        addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        addVfRead(usage, vfT, laneForComponent((lower >> 23) & 3u));
        break;
    case 0x3B:
        usage.pipeline = PipelineFdiv;
        usage.waitQ = true;
        break;
    case 0x3C:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        writeVi(viT);
        break;
    case 0x3D:
        usage.pipeline = PipelineFmac;
        usage.latency = 4u;
        readVi(viS);
        addVfWrite(usage, vfT, dest);
        break;
    case 0x3E:
        usage.pipeline = PipelineLsu;
        usage.latency = 4u;
        readVi(viS);
        writeVi(viT);
        break;
    case 0x3F:
        usage.pipeline = PipelineLsu;
        usage.latency = 1u;
        readVi(viS);
        readVi(viT);
        break;
    case 0x40:
    case 0x41:
        usage.pipeline = PipelineFmac;
        usage.latency = 4u;
        addVfWrite(usage, vfT, dest);
        break;
    case 0x42:
    case 0x43:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        break;
    case 0x64:
        if (unit == Unit::VU0)
        {
            usage.reserved = true;
            break;
        }
        usage.pipeline = PipelineFmac;
        usage.latency = 4u;
        addVfWrite(usage, vfT, dest);
        break;
    case 0x68:
    case 0x69:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        writeVi(viT);
        break;
    case 0x6C:
        if (unit == Unit::VU0)
        {
            usage.reserved = true;
            break;
        }
        usage.pipeline = PipelineXgkick;
        usage.latency = 2u;
        readVi(viS);
        break;
    case 0x70:
    case 0x71:
    case 0x72:
    case 0x73:
    case 0x74:
    case 0x75:
    case 0x76:
    case 0x78:
    case 0x79:
    case 0x7A:
    case 0x7C:
    case 0x7D:
    case 0x7E:
        // EFU (PCSX2 LowerOP_T3_xx): ESADD 70, ERSADD 71, ELENG 72, ERLENG 73, EATANxy 74, EATANxz 75, ESUM 76, ESQRT 78,
        // ERSQRT 79, ERCPR 7A, ESIN 7C, EATAN 7D, EEXP 7E (77 is invalid)
        if (unit == Unit::VU0)
        {
            usage.reserved = true;
            break;
        }
        usage.pipeline = PipelineEfu;
        switch (special)
        {
        case 0x70:
            usage.latency = 11u;
            break;
        case 0x71:
        case 0x72:
        case 0x79:
            usage.latency = 18u;
            break;
        case 0x73:
            usage.latency = 24u;
            break;
        case 0x74:
        case 0x75:
        case 0x7D:
            usage.latency = 54u;
            break;
        case 0x76:
        case 0x78:
        case 0x7A:
            usage.latency = 12u;
            break;
        case 0x7C:
            usage.latency = 29u;
            break;
        case 0x7E:
            usage.latency = 44u;
            break;
        default:
            break;
        }
        if (special >= 0x70u && special <= 0x73u)
            addVfRead(usage, vfS, 0xEu);
        else if (special == 0x74u)
            addVfRead(usage, vfS, 0xCu);
        else if (special == 0x75u)
            addVfRead(usage, vfS, 0xAu);
        else if (special == 0x76u)
            addVfRead(usage, vfS, 0xFu);
        else
            addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        break;
    case 0x7B:
        if (unit == Unit::VU0)
        {
            usage.reserved = true;
            break;
        }
        usage.pipeline = PipelineEfu;
        usage.waitP = true;
        break;
    default:
        usage.reserved = true;
        break;
    }
    return usage;
}

template <VU1Interpreter::FmacKind Kind, VU1Interpreter::FmacOperand Operand>
[[gnu::always_inline]] inline void VU1Interpreter::computeFmacOpScalar(uint32_t instr, uint8_t dest, float result[4],
                                                                      uint8_t laneFlags[4], uint32_t &extraSticky)
{
    // Same computation as execFmac (see there), with the opcode family fixed at compile time.
    static constexpr uint8_t kCrossLeft[4] = {1u, 2u, 0u, 3u};
    static constexpr uint8_t kCrossRight[4] = {2u, 0u, 1u, 3u};
    constexpr bool cross = Kind == FmacKind::Opmsub || Kind == FmacKind::Opmula;
    constexpr bool productSum = Kind == FmacKind::Madd || Kind == FmacKind::Msub || Kind == FmacKind::Opmsub;
    const float *vs = m_state.vf[FS(instr)];
    const float *vt = m_state.vf[FT(instr)];
    for (uint32_t component = 0; component < 4u; ++component)
    {
        if ((dest & (1u << (3u - component))) == 0u)
            continue;
        const float s = normalizeOperand(vs[cross ? kCrossLeft[component] : component]);
        float r;
        if constexpr (Operand == FmacOperand::Broadcast)
            r = normalizeOperand(vt[instr & 3u]);
        else if constexpr (Operand == FmacOperand::Q)
            r = normalizeOperand(m_state.q);
        else if constexpr (Operand == FmacOperand::I)
            r = normalizeOperand(m_state.i);
        else
            r = normalizeOperand(vt[cross ? kCrossRight[component] : component]);
        const float a = normalizeOperand(m_state.acc[component]);
        const double ls = static_cast<double>(s);
        const double lr = static_cast<double>(r);
        const double la = static_cast<double>(a);
        float value;
        double exact;
        if constexpr (productSum)
        {
            // Two roundings: the product chopped (and clamped / flushed like any result), then
            // the add chopped. The product's Z/S/U/O accumulate into the sticky flags.
            float product = s * r;
            extraSticky |= normalizeFmacExactResult(product, ls * lr) & 0xFu;
            const double lp = static_cast<double>(product);
            if constexpr (Kind == FmacKind::Madd)
            {
                value = a + product;
                exact = la + lp;
            }
            else
            {
                value = a - product;
                exact = la - lp;
            }
            if constexpr (Kind == FmacKind::Opmsub)
            {
                if (component == 3u)
                {
                    value = 0.0f;
                    exact = 0.0;
                }
            }
        }
        else if constexpr (Kind == FmacKind::Add)
        {
            value = s + r;
            exact = ls + lr;
        }
        else if constexpr (Kind == FmacKind::Sub)
        {
            value = s - r;
            exact = ls - lr;
        }
        else if constexpr (Kind == FmacKind::Mul)
        {
            value = s * r;
            exact = ls * lr;
        }
        else
        {
            value = component == 3u ? 0.0f : s * r;
            exact = component == 3u ? 0.0 : ls * lr;
        }
        laneFlags[component] = normalizeFmacExactResult(value, exact);
        result[component] = value;
    }
}


template <VU1Interpreter::FmacKind Kind, VU1Interpreter::FmacOperand Operand>
[[gnu::always_inline]] inline VU1Interpreter::FmacFlagBits VU1Interpreter::computeFmacOp(uint32_t instr, uint8_t dest,
                                                                                        float result[4])
{
    constexpr bool cross = Kind == FmacKind::Opmsub || Kind == FmacKind::Opmula;
    constexpr bool productSum = Kind == FmacKind::Madd || Kind == FmacKind::Msub || Kind == FmacKind::Opmsub;
    constexpr bool usesAcc = productSum;
    constexpr bool wZero = cross; // OPMSUB/OPMULA: w = 0
    FmacFlagBits bits;
#if defined(PS2X_VU_FMAC_NEON)
    // Lanes are components x y z w; MAC/dest bits weigh them 8 4 2 1.
    const auto normalize = [](uint32x4_t v)
    {
        const uint32x4_t sign = vandq_u32(v, vdupq_n_u32(0x80000000u));
        const uint32x4_t exponent = vandq_u32(v, vdupq_n_u32(0x7F800000u));
        uint32x4_t out = vbslq_u32(vceqzq_u32(exponent), sign, v);
        return vbslq_u32(vceqq_u32(exponent, vdupq_n_u32(0x7F800000u)), vorrq_u32(sign, vdupq_n_u32(0x7F7FFFFFu)), out);
    };
    static constexpr uint8_t kCrossLeft[16] = {4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 12, 13, 14, 15};  // y z x w
    static constexpr uint8_t kCrossRight[16] = {8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15}; // z x y w
    uint32x4_t sBits = vreinterpretq_u32_f32(vld1q_f32(m_state.vf[FS(instr)]));
    if constexpr (cross)
        sBits = vreinterpretq_u32_u8(vqtbl1q_u8(vreinterpretq_u8_u32(sBits), vld1q_u8(kCrossLeft)));
    uint32x4_t rBits;
    if constexpr (Operand == FmacOperand::Broadcast)
    {
        uint32_t lane;
        std::memcpy(&lane, &m_state.vf[FT(instr)][instr & 3u], 4);
        rBits = vdupq_n_u32(lane);
    }
    else if constexpr (Operand == FmacOperand::Q || Operand == FmacOperand::I)
    {
        uint32_t lane;
        std::memcpy(&lane, Operand == FmacOperand::Q ? &m_state.q : &m_state.i, 4);
        rBits = vdupq_n_u32(lane);
    }
    else
    {
        rBits = vreinterpretq_u32_f32(vld1q_f32(m_state.vf[FT(instr)]));
        if constexpr (cross)
            rBits = vreinterpretq_u32_u8(vqtbl1q_u8(vreinterpretq_u8_u32(rBits), vld1q_u8(kCrossRight)));
    }
    const float32x4_t s = vreinterpretq_f32_u32(normalize(sBits));
    const float32x4_t r = vreinterpretq_f32_u32(normalize(rBits));
    float32x4_t a = vdupq_n_f32(0.0f);
    if constexpr (usesAcc)
        a = vreinterpretq_f32_u32(normalize(vreinterpretq_u32_f32(vld1q_f32(m_state.acc))));

    // Product-sum: two roundings, the product chopped and flushed/clamped like any result, then
    // the add chopped.
    float32x4_t product = vdupq_n_f32(0.0f);
    if constexpr (productSum)
        product = vreinterpretq_f32_u32(normalize(vreinterpretq_u32_f32(vmulq_f32(s, r))));
    float32x4_t value;
    if constexpr (Kind == FmacKind::Add)
        value = vaddq_f32(s, r);
    else if constexpr (Kind == FmacKind::Sub)
        value = vsubq_f32(s, r);
    else if constexpr (Kind == FmacKind::Madd)
        value = vaddq_f32(a, product);
    else if constexpr (Kind == FmacKind::Msub || Kind == FmacKind::Opmsub)
        value = vsubq_f32(a, product);
    else
        value = vmulq_f32(s, r);

    static constexpr uint32_t kWeights[4] = {8u, 4u, 2u, 1u};
    const uint32x4_t weights = vld1q_u32(kWeights);
    const uint32x4_t destLanes = vtstq_u32(vdupq_n_u32(dest), weights);
    const auto laneBits = [&](uint32x4_t mask) { return vaddvq_u32(vandq_u32(vandq_u32(mask, destLanes), weights)); };
    if constexpr (wZero)
        value = vsetq_lane_f32(0.0f, value, 3);
    {
        // Fast path. Each float result is the exact result of its operation rounded toward zero
        // (run() executes microcode in FE_TOWARDZERO), so a normal result below the largest
        // float means no Z/U/O and the float's sign, with nothing to fix up. A zero result is an
        // exact zero for an add or subtract of normal (or zero) operands, and for MUL when an
        // operand is zero. Anything else (and every lane of a product-sum's products the same
        // way) takes the full classification below.
        const auto finite = [](uint32x4_t bits)
        {
            const uint32x4_t magnitude = vandq_u32(bits, vdupq_n_u32(0x7FFFFFFFu));
            return vcltq_u32(vsubq_u32(magnitude, vdupq_n_u32(0x00800000u)), vdupq_n_u32(0x7F7FFFFFu - 0x00800000u));
        };
        const uint32x4_t sZero = vceqzq_u32(vandq_u32(vreinterpretq_u32_f32(s), vdupq_n_u32(0x7FFFFFFFu)));
        const uint32x4_t rZero = vceqzq_u32(vandq_u32(vreinterpretq_u32_f32(r), vdupq_n_u32(0x7FFFFFFFu)));
        const uint32x4_t valueBits = vreinterpretq_u32_f32(value);
        const uint32x4_t valueZero = vceqzq_u32(vandq_u32(valueBits, vdupq_n_u32(0x7FFFFFFFu)));
        uint32x4_t ok = finite(valueBits);
        if constexpr (Kind == FmacKind::Mul || Kind == FmacKind::Opmula)
            ok = vorrq_u32(ok, vandq_u32(valueZero, vorrq_u32(sZero, rZero)));
        else
            ok = vorrq_u32(ok, valueZero);
        if constexpr (wZero)
            ok = vsetq_lane_u32(0xFFFFFFFFu, ok, 3); // w = exact +0
        const uint32x4_t productBits = vreinterpretq_u32_f32(product);
        if constexpr (productSum)
        {
            // (the normalized product: a flushed or clamped one is not "finite" or an exact zero)
            const uint32x4_t productZero = vceqzq_u32(vandq_u32(productBits, vdupq_n_u32(0x7FFFFFFFu)));
            ok = vandq_u32(ok, vorrq_u32(finite(productBits), vandq_u32(productZero, vorrq_u32(sZero, rZero))));
        }
        if (vminvq_u32(vornq_u32(ok, destLanes)) == 0xFFFFFFFFu)
        {
            vst1q_f32(result, value);
            const uint32_t z = laneBits(valueZero), sg = laneBits(vcltzq_s32(vreinterpretq_s32_u32(valueBits)));
            bits.mac = z | (sg << 4);
            bits.status = (z ? 1u : 0u) | (sg ? 2u : 0u);
            if constexpr (productSum)
            {
                const uint32_t pz = laneBits(vceqzq_u32(vandq_u32(productBits, vdupq_n_u32(0x7FFFFFFFu))));
                const uint32_t ps = laneBits(vcltzq_s32(vreinterpretq_s32_u32(productBits)));
                bits.productSticky = (pz ? 1u : 0u) | (ps ? 2u : 0u);
            }
            return bits;
        }
    }

    // The exact result in double (a product of two floats is exact there; for a product-sum, the
    // exact sum of the accumulator and the chopped product).
    const float64x2_t sLo = vcvt_f64_f32(vget_low_f32(s)), sHi = vcvt_high_f64_f32(s);
    const float64x2_t rLo = vcvt_f64_f32(vget_low_f32(r)), rHi = vcvt_high_f64_f32(r);
    float64x2_t eLo, eHi;
    if constexpr (Kind == FmacKind::Add)
        eLo = vaddq_f64(sLo, rLo), eHi = vaddq_f64(sHi, rHi);
    else if constexpr (Kind == FmacKind::Sub)
        eLo = vsubq_f64(sLo, rLo), eHi = vsubq_f64(sHi, rHi);
    else if constexpr (Kind == FmacKind::Mul || Kind == FmacKind::Opmula)
        eLo = vmulq_f64(sLo, rLo), eHi = vmulq_f64(sHi, rHi);
    else
    {
        const float64x2_t aLo = vcvt_f64_f32(vget_low_f32(a)), aHi = vcvt_high_f64_f32(a);
        const float64x2_t pLo = vcvt_f64_f32(vget_low_f32(product)), pHi = vcvt_high_f64_f32(product);
        if constexpr (Kind == FmacKind::Madd)
            eLo = vaddq_f64(aLo, pLo), eHi = vaddq_f64(aHi, pHi);
        else
            eLo = vsubq_f64(aLo, pLo), eHi = vsubq_f64(aHi, pHi);
    }
    if constexpr (wZero)
    {
        eHi = vsetq_lane_f64(0.0, eHi, 1);
        value = vsetq_lane_f32(0.0f, value, 3);
    }

    struct Classes
    {
        uint32x4_t zero, sign, under, over;
    };
    const auto classify = [](float64x2_t lo, float64x2_t hi)
    {
        const auto narrow = [](uint64x2_t l, uint64x2_t h) { return vcombine_u32(vmovn_u64(l), vmovn_u64(h)); };
        const float64x2_t maximum = vdupq_n_f64(static_cast<double>(std::numeric_limits<float>::max()));
        const float64x2_t minimum = vdupq_n_f64(static_cast<double>(std::numeric_limits<float>::min()));
        const uint64x2_t zLo = vceqzq_f64(lo), zHi = vceqzq_f64(hi);
        Classes c;
        c.zero = narrow(zLo, zHi);
        c.sign = narrow(vcltzq_s64(vreinterpretq_s64_f64(lo)), vcltzq_s64(vreinterpretq_s64_f64(hi)));
        c.over = narrow(vcgtq_f64(vabsq_f64(lo), maximum), vcgtq_f64(vabsq_f64(hi), maximum));
        c.under = narrow(vbicq_u64(vcltq_f64(vabsq_f64(lo), minimum), zLo), vbicq_u64(vcltq_f64(vabsq_f64(hi), minimum), zHi));
        return c;
    };
    const Classes c = classify(eLo, eHi);
    // Z for zero and underflow (result +-0), O clamps to +-max, otherwise the float result.
    const uint32x4_t signBits = vandq_u32(c.sign, vdupq_n_u32(0x80000000u));
    const uint32x4_t zeroed = vorrq_u32(c.zero, c.under);
    uint32x4_t out = vbslq_u32(zeroed, signBits, vreinterpretq_u32_f32(value));
    out = vbslq_u32(c.over, vorrq_u32(signBits, vdupq_n_u32(0x7F7FFFFFu)), out);
    vst1q_f32(result, vreinterpretq_f32_u32(out));

    const uint32_t z = laneBits(zeroed), sg = laneBits(c.sign), u = laneBits(c.under), o = laneBits(c.over);
    bits.mac = z | (sg << 4) | (u << 8) | (o << 12);
    bits.status = (z ? 1u : 0u) | (sg ? 2u : 0u) | (u ? 4u : 0u) | (o ? 8u : 0u);
    if constexpr (productSum)
    {
        const Classes p = classify(vmulq_f64(sLo, rLo), vmulq_f64(sHi, rHi));
        bits.productSticky = (laneBits(vorrq_u32(p.zero, p.under)) ? 1u : 0u) | (laneBits(p.sign) ? 2u : 0u) |
                             (laneBits(p.under) ? 4u : 0u) | (laneBits(p.over) ? 8u : 0u);
    }
#elif defined(PS2X_VU_FMAC_SSE2)
    // Lanes are components x y z w; MAC/dest bits weigh them 8 4 2 1 (reversed movemask order).
    static constexpr uint8_t kReverse[16] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};
    const auto normalize = [](__m128 v)
    {
        const __m128i bitsIn = _mm_castps_si128(v);
        const __m128i sign = _mm_and_si128(bitsIn, _mm_set1_epi32(static_cast<int>(0x80000000u)));
        const __m128i exponent = _mm_and_si128(bitsIn, _mm_set1_epi32(0x7F800000));
        const __m128i isZero = _mm_cmpeq_epi32(exponent, _mm_setzero_si128());
        const __m128i isMax = _mm_cmpeq_epi32(exponent, _mm_set1_epi32(0x7F800000));
        __m128i out = _mm_or_si128(_mm_and_si128(isZero, sign), _mm_andnot_si128(isZero, bitsIn));
        const __m128i clamped = _mm_or_si128(sign, _mm_set1_epi32(0x7F7FFFFF));
        out = _mm_or_si128(_mm_and_si128(isMax, clamped), _mm_andnot_si128(isMax, out));
        return _mm_castsi128_ps(out);
    };
    __m128 sIn = _mm_loadu_ps(m_state.vf[FS(instr)]);
    if constexpr (cross)
        sIn = _mm_shuffle_ps(sIn, sIn, _MM_SHUFFLE(3, 0, 2, 1)); // y z x w
    __m128 rIn;
    if constexpr (Operand == FmacOperand::Broadcast)
        rIn = _mm_set1_ps(m_state.vf[FT(instr)][instr & 3u]);
    else if constexpr (Operand == FmacOperand::Q)
        rIn = _mm_set1_ps(m_state.q);
    else if constexpr (Operand == FmacOperand::I)
        rIn = _mm_set1_ps(m_state.i);
    else
    {
        rIn = _mm_loadu_ps(m_state.vf[FT(instr)]);
        if constexpr (cross)
            rIn = _mm_shuffle_ps(rIn, rIn, _MM_SHUFFLE(3, 1, 0, 2)); // z x y w
    }
    const __m128 s = normalize(sIn);
    const __m128 r = normalize(rIn);
    __m128 a = _mm_setzero_ps();
    if constexpr (usesAcc)
        a = normalize(_mm_loadu_ps(m_state.acc));

    // Product-sum: two roundings, the product chopped and flushed/clamped like any result, then
    // the add chopped.
    __m128 product = _mm_setzero_ps();
    if constexpr (productSum)
        product = normalize(_mm_mul_ps(s, r));
    __m128 value;
    if constexpr (Kind == FmacKind::Add)
        value = _mm_add_ps(s, r);
    else if constexpr (Kind == FmacKind::Sub)
        value = _mm_sub_ps(s, r);
    else if constexpr (Kind == FmacKind::Madd)
        value = _mm_add_ps(a, product);
    else if constexpr (Kind == FmacKind::Msub || Kind == FmacKind::Opmsub)
        value = _mm_sub_ps(a, product);
    else
        value = _mm_mul_ps(s, r);

    const __m128d sLo = _mm_cvtps_pd(s), sHi = _mm_cvtps_pd(_mm_movehl_ps(s, s));
    const __m128d rLo = _mm_cvtps_pd(r), rHi = _mm_cvtps_pd(_mm_movehl_ps(r, r));
    __m128d eLo, eHi;
    if constexpr (Kind == FmacKind::Add)
        eLo = _mm_add_pd(sLo, rLo), eHi = _mm_add_pd(sHi, rHi);
    else if constexpr (Kind == FmacKind::Sub)
        eLo = _mm_sub_pd(sLo, rLo), eHi = _mm_sub_pd(sHi, rHi);
    else if constexpr (Kind == FmacKind::Mul || Kind == FmacKind::Opmula)
        eLo = _mm_mul_pd(sLo, rLo), eHi = _mm_mul_pd(sHi, rHi);
    else
    {
        const __m128d aLo = _mm_cvtps_pd(a), aHi = _mm_cvtps_pd(_mm_movehl_ps(a, a));
        const __m128d pLo = _mm_cvtps_pd(product), pHi = _mm_cvtps_pd(_mm_movehl_ps(product, product));
        if constexpr (Kind == FmacKind::Madd)
            eLo = _mm_add_pd(aLo, pLo), eHi = _mm_add_pd(aHi, pHi);
        else
            eLo = _mm_sub_pd(aLo, pLo), eHi = _mm_sub_pd(aHi, pHi);
    }
    if constexpr (wZero)
        eHi = _mm_move_sd(_mm_setzero_pd(), eHi); // w = +0

    struct Classes
    {
        uint32_t zero, sign, under, over; // bit = component
    };
    const auto classify = [](__m128d lo, __m128d hi)
    {
        const __m128d signMask = _mm_set1_pd(-0.0);
        const __m128d maximum = _mm_set1_pd(static_cast<double>(std::numeric_limits<float>::max()));
        const __m128d minimum = _mm_set1_pd(static_cast<double>(std::numeric_limits<float>::min()));
        const auto mask = [](__m128d l, __m128d h) { return static_cast<uint32_t>(_mm_movemask_pd(l) | (_mm_movemask_pd(h) << 2)); };
        const __m128d magLo = _mm_andnot_pd(signMask, lo), magHi = _mm_andnot_pd(signMask, hi);
        Classes c;
        c.zero = mask(_mm_cmpeq_pd(lo, _mm_setzero_pd()), _mm_cmpeq_pd(hi, _mm_setzero_pd()));
        c.sign = mask(lo, hi);
        c.over = mask(_mm_cmpgt_pd(magLo, maximum), _mm_cmpgt_pd(magHi, maximum));
        c.under = mask(_mm_cmplt_pd(magLo, minimum), _mm_cmplt_pd(magHi, minimum)) & ~c.zero;
        return c;
    };
    const uint32_t destComponents = kReverse[dest & 0xFu];
    const auto laneBits = [&](uint32_t components) { return static_cast<uint32_t>(kReverse[components & destComponents]); };

    const Classes c = classify(eLo, eHi);
    const uint32_t zeroed = c.zero | c.under;
    alignas(16) uint32_t out[4];
    _mm_store_si128(reinterpret_cast<__m128i *>(out), _mm_castps_si128(value));
    if constexpr (wZero)
        out[3] = 0u;
    for (uint32_t component = 0; component < 4u; ++component)
    {
        const uint32_t sign = (c.sign >> component & 1u) ? 0x80000000u : 0u;
        if (zeroed >> component & 1u)
            out[component] = sign;
        else if (c.over >> component & 1u)
            out[component] = sign | 0x7F7FFFFFu;
    }
    std::memcpy(result, out, sizeof(out));

    const uint32_t z = laneBits(zeroed), sg = laneBits(c.sign), u = laneBits(c.under), o = laneBits(c.over);
    bits.mac = z | (sg << 4) | (u << 8) | (o << 12);
    bits.status = (z ? 1u : 0u) | (sg ? 2u : 0u) | (u ? 4u : 0u) | (o ? 8u : 0u);
    if constexpr (productSum)
    {
        const Classes p = classify(_mm_mul_pd(sLo, rLo), _mm_mul_pd(sHi, rHi));
        bits.productSticky = (laneBits(p.zero | p.under) ? 1u : 0u) | (laneBits(p.sign) ? 2u : 0u) |
                             (laneBits(p.under) ? 4u : 0u) | (laneBits(p.over) ? 8u : 0u);
    }
#else
    uint8_t laneFlags[4]{};
    computeFmacOpScalar<Kind, Operand>(instr, dest, result, laneFlags, bits.productSticky);
    for (uint32_t component = 0; component < 4u; ++component)
    {
        const uint8_t lane = laneForComponent(component);
        if ((dest & lane) == 0u)
            continue;
        const uint32_t flags = laneFlags[component];
        bits.mac |= ((flags & 1u) ? lane : 0u) | ((flags & 2u) ? lane << 4 : 0u) | ((flags & 4u) ? lane << 8 : 0u) |
                    ((flags & 8u) ? lane << 12 : 0u);
        bits.status |= flags;
    }
#endif
    return bits;
}

inline void VU1Interpreter::queueFmacFlags(uint32_t mac, uint32_t status, uint32_t extraSticky)
{
    FlagPipelineEntry *entry = allocFlagEntry(true, 0xFFFFFFFFu);
    if (!entry)
        return;
    entry->mac = mac;
    entry->status = status;
    entry->extraSticky = extraSticky;
    entry->writesMac = true;
    entry->writesStatus = true;
}

template <VU1Interpreter::FmacKind Kind, VU1Interpreter::FmacOperand Operand, bool ToAcc, uint8_t Dest>
void VU1Interpreter::execFmacOp(uint32_t instr)
{
    const uint8_t dest = Dest != 0u ? Dest : DEST(instr);
    float result[4];
    const FmacFlagBits bits = computeFmacOp<Kind, Operand>(instr, dest, result);
    if (dest != 0u)
        queueFmacFlags(bits.mac, bits.status, bits.productSticky);
    if constexpr (ToAcc)
        applyDestAcc(result, dest);
    else
        applyDest(m_state.vf[FD(instr)], result, dest);
}

template <VU1Interpreter::FmacKind Kind, VU1Interpreter::FmacOperand Operand, bool ToAcc, uint8_t Dest, bool Live>
[[gnu::always_inline]] inline void VU1Interpreter::execFmacRec(uint32_t instr, uint32_t &sticky)
{
    static_assert(Dest != 0u, "an FMAC with an empty dest mask writes nothing, flags included");
    float result[4];
    const FmacFlagBits bits = computeFmacOp<Kind, Operand>(instr, Dest, result);
    if constexpr (Live)
    {
        queueFmacFlags(bits.mac, bits.status, bits.productSticky | sticky);
        sticky = 0u;
    }
    else
        sticky |= (bits.status | bits.productSticky) & 0xFu;
    if constexpr (ToAcc)
        applyDestAcc(result, Dest);
    else
        applyDest(m_state.vf[FD(instr)], result, Dest);
}

template <VU1Interpreter::FmacKind Kind, VU1Interpreter::FmacOperand Operand, bool ToAcc, uint8_t Dest>
[[gnu::always_inline]] inline VU1Interpreter::FmacFlagBits VU1Interpreter::execFmacDirect(uint32_t instr)
{
    static_assert(Dest != 0u, "an FMAC with an empty dest mask writes nothing, flags included");
    float result[4];
    const FmacFlagBits bits = computeFmacOp<Kind, Operand>(instr, Dest, result);
    if constexpr (ToAcc)
        applyDestAcc(result, Dest);
    else
        applyDest(m_state.vf[FD(instr)], result, Dest);
    return bits;
}

constexpr VU1Interpreter::FmacSpec VU1Interpreter::fmacSpecFor(uint32_t upper)
{
    using K = FmacKind;
    using O = FmacOperand;
    const uint8_t op = static_cast<uint8_t>(upper & 0x3Fu);
    const bool toAcc = op >= 0x3Cu;
    if (op >= 0x30u && !toAcc)
        return {};
    const uint8_t code = toAcc ? static_cast<uint8_t>((upper & 3u) | ((upper >> 4) & 0x7Cu)) : op;
    if (code <= 0x03u)
        return {true, K::Add, O::Broadcast, toAcc};
    if (code <= 0x07u)
        return {true, K::Sub, O::Broadcast, toAcc};
    if (code <= 0x0Bu)
        return {true, K::Madd, O::Broadcast, toAcc};
    if (code <= 0x0Fu)
        return {true, K::Msub, O::Broadcast, toAcc};
    if (code >= 0x18u && code <= 0x1Bu)
        return {true, K::Mul, O::Broadcast, toAcc};
    switch (code)
    {
    case 0x1Cu: return {true, K::Mul, O::Q, toAcc};
    case 0x1Eu: return {true, K::Mul, O::I, toAcc};
    case 0x20u: return {true, K::Add, O::Q, toAcc};
    case 0x21u: return {true, K::Madd, O::Q, toAcc};
    case 0x22u: return {true, K::Add, O::I, toAcc};
    case 0x23u: return {true, K::Madd, O::I, toAcc};
    case 0x24u: return {true, K::Sub, O::Q, toAcc};
    case 0x25u: return {true, K::Msub, O::Q, toAcc};
    case 0x26u: return {true, K::Sub, O::I, toAcc};
    case 0x27u: return {true, K::Msub, O::I, toAcc};
    case 0x28u: return {true, K::Add, O::Vector, toAcc};
    case 0x29u: return {true, K::Madd, O::Vector, toAcc};
    case 0x2Au: return {true, K::Mul, O::Vector, toAcc};
    case 0x2Cu: return {true, K::Sub, O::Vector, toAcc};
    case 0x2Du: return {true, K::Msub, O::Vector, toAcc};
    case 0x2Eu: return {true, toAcc ? K::Opmula : K::Opmsub, O::Vector, toAcc};
    default: return {};
    }
}

constexpr VU1Interpreter::UpperHandler VU1Interpreter::fmacHandlerFor(uint32_t upper)
{
    const FmacSpec spec = fmacSpecFor(upper);
    if (!spec.valid)
        return nullptr;
    const uint8_t dest = static_cast<uint8_t>((upper >> 21) & 0xFu);
    // Every (kind, operand, acc, dest) combination the generic path below can produce.
#define PS2X_FMAC_D(K_, O_, A_)                                                                  \
    if (spec.kind == FmacKind::K_ && spec.operand == FmacOperand::O_ && spec.toAcc == A_)        \
        return dest == 0xFu   ? &VU1Interpreter::execFmacOp<FmacKind::K_, FmacOperand::O_, A_, 0xFu> \
               : dest == 0xEu ? &VU1Interpreter::execFmacOp<FmacKind::K_, FmacOperand::O_, A_, 0xEu> \
                              : &VU1Interpreter::execFmacOp<FmacKind::K_, FmacOperand::O_, A_, 0u>;
#define PS2X_FMAC(K_, O_) PS2X_FMAC_D(K_, O_, true) PS2X_FMAC_D(K_, O_, false)
    PS2X_FMAC(Add, Broadcast)
    PS2X_FMAC(Sub, Broadcast)
    PS2X_FMAC(Madd, Broadcast)
    PS2X_FMAC(Msub, Broadcast)
    PS2X_FMAC(Mul, Broadcast)
    PS2X_FMAC(Mul, Q)
    PS2X_FMAC(Mul, I)
    PS2X_FMAC(Add, Q)
    PS2X_FMAC(Madd, Q)
    PS2X_FMAC(Add, I)
    PS2X_FMAC(Madd, I)
    PS2X_FMAC(Sub, Q)
    PS2X_FMAC(Msub, Q)
    PS2X_FMAC(Sub, I)
    PS2X_FMAC(Msub, I)
    PS2X_FMAC(Add, Vector)
    PS2X_FMAC(Madd, Vector)
    PS2X_FMAC(Mul, Vector)
    PS2X_FMAC(Sub, Vector)
    PS2X_FMAC(Msub, Vector)
    PS2X_FMAC_D(Opmula, Vector, true)
    PS2X_FMAC_D(Opmsub, Vector, false)
#undef PS2X_FMAC
#undef PS2X_FMAC_D
    return nullptr;
}

constexpr VU1Interpreter::DecodedInstructionPair VU1Interpreter::decodePair(uint32_t lower, uint32_t upper, Unit unit)
{
    DecodedInstructionPair decoded;
    decoded.lower = lower;
    decoded.upper = upper;
    decoded.iBit = (decoded.upper & 0x80000000u) != 0u;
    decoded.eBit = (decoded.upper & 0x40000000u) != 0u;
    decoded.mBit = (decoded.upper & 0x20000000u) != 0u;
    decoded.dBit = (decoded.upper & 0x10000000u) != 0u;
    decoded.tBit = (decoded.upper & 0x08000000u) != 0u;
    decoded.upperUsage = decodeUpperUsage(decoded.upper);
    decoded.upperHandler = decoded.upperUsage.reserved ? nullptr : fmacHandlerFor(decoded.upper);
    if (!decoded.iBit)
        decoded.lowerUsage = decodeLowerUsage(decoded.lower, unit);

    const uint8_t upperWriteReg = decoded.upperUsage.vfWrite.reg;
    if (upperWriteReg != 0u && (vfReadLanes(decoded.lowerUsage, upperWriteReg) != 0u || decoded.lowerUsage.vfWrite.reg == upperWriteReg))
    {
        decoded.upperVfShadowReg = upperWriteReg;
        if (decoded.lowerUsage.vfWrite.reg == upperWriteReg)
            decoded.suppressedLowerVf = upperWriteReg;
    }

    for (const InstructionUsage *usage : {&decoded.upperUsage, &decoded.lowerUsage})
    {
        for (uint32_t index = 0; index < usage->vfReadCount; ++index)
        {
            const VfAccess &access = usage->vfRead[index];
            for (uint32_t component = 0; component < 4u; ++component)
                if ((access.lanes & laneForComponent(component)) != 0u && decoded.vfReadLaneCount < 16u)
                    decoded.vfReadLanes[decoded.vfReadLaneCount++] = static_cast<uint8_t>(access.reg * 4u + component);
        }
        for (uint32_t reg = 1; reg < 16u; ++reg)
            if ((usage->viRead & (1u << reg)) != 0u && decoded.viReadCount < 15u)
                decoded.viReads[decoded.viReadCount++] = static_cast<uint8_t>(reg);
        for (uint32_t component = 0; component < 4u; ++component)
            if ((usage->accRead & laneForComponent(component)) != 0u && decoded.accReadCount < 4u)
                decoded.accReads[decoded.accReadCount++] = static_cast<uint8_t>(component);
    }

    const VfAccess lowerWrite = decoded.lowerUsage.vfWrite;
    if (lowerWrite.reg != 0u && decoded.suppressedLowerVf != lowerWrite.reg)
    {
        decoded.lowerVfWriteLatency = decoded.lowerUsage.vfLatency != 0u ? decoded.lowerUsage.vfLatency
                                                                         : decoded.lowerUsage.latency;
        for (uint32_t component = 0; component < 4u; ++component)
            if ((lowerWrite.lanes & laneForComponent(component)) != 0u)
                decoded.lowerVfWriteLanes[decoded.lowerVfWriteCount++] = static_cast<uint8_t>(lowerWrite.reg * 4u + component);
    }
    const VfAccess upperWrite = decoded.upperUsage.vfWrite;
    if (upperWrite.reg != 0u)
    {
        decoded.upperVfWriteLatency = decoded.upperUsage.vfLatency != 0u ? decoded.upperUsage.vfLatency
                                                                         : decoded.upperUsage.latency;
        for (uint32_t component = 0; component < 4u; ++component)
            if ((upperWrite.lanes & laneForComponent(component)) != 0u)
                decoded.upperVfWriteLanes[decoded.upperVfWriteCount++] = static_cast<uint8_t>(upperWrite.reg * 4u + component);
    }
    decoded.viWriteLatency = decoded.lowerUsage.viLatency != 0u ? decoded.lowerUsage.viLatency : decoded.lowerUsage.latency;
    for (uint32_t reg = 1; reg < 16u; ++reg)
        if ((decoded.lowerUsage.viWrite & (1u << reg)) != 0u)
            decoded.viWrites[decoded.viWriteCount++] = static_cast<uint8_t>(reg);
    for (uint32_t component = 0; component < 4u; ++component)
        if ((decoded.upperUsage.accWrite & laneForComponent(component)) != 0u)
            decoded.accWrites[decoded.accWriteCount++] = static_cast<uint8_t>(component);
    return decoded;
}

#endif
