#include "runtime/ps2_vu1.h"
#include "ps2_vu1_detail.h"

#include <cmath>
#include <cstring>
#include <limits>

namespace
{
    int32_t vuFloatToInt(float value, float scale)
    {
        const double scaled = static_cast<double>(value) * static_cast<double>(scale);
        if (scaled >= static_cast<double>(std::numeric_limits<int32_t>::max()))
            return std::numeric_limits<int32_t>::max();
        if (scaled <= static_cast<double>(std::numeric_limits<int32_t>::min()))
            return std::numeric_limits<int32_t>::min();
        return static_cast<int32_t>(scaled);
    }
}

// ============================================================================
// Upper instructions (FMAC pipeline)
// ============================================================================
void VU1Interpreter::execUpper(uint32_t instr)
{
    m_currentUpperInstruction = instr;
    uint8_t dest = DEST(instr);
    uint8_t ft = FT(instr);
    uint8_t fs = FS(instr);
    uint8_t fd = FD(instr);
    uint8_t op = instr & 0x3F;

    float *vd = m_state.vf[fd];
    float normalizedVs[4];
    float normalizedVt[4];
    float normalizedAcc[4];
    for (uint32_t component = 0; component < 4u; ++component)
    {
        normalizedVs[component] = normalizeOperand(m_state.vf[fs][component]);
        normalizedVt[component] = normalizeOperand(m_state.vf[ft][component]);
        normalizedAcc[component] = normalizeOperand(m_state.acc[component]);
    }
    const float *vs = normalizedVs;
    const float *vt = normalizedVt;
    const float *acc = normalizedAcc;
    const float q = normalizeOperand(m_state.q);
    const float i = normalizeOperand(m_state.i);
    float result[4];

    // Upper opcode decoding (bits 5:0 of upper word)
    switch (op)
    {
    case 0x00:
    case 0x01:
    case 0x02:
    case 0x03: // ADDbc
        execFmac(FmacKind::Add, FmacOperand::Broadcast, static_cast<uint8_t>(op & 3), false, vd, dest, vs, vt, acc, q, i);
        return;
    case 0x04:
    case 0x05:
    case 0x06:
    case 0x07: // SUBbc
        execFmac(FmacKind::Sub, FmacOperand::Broadcast, static_cast<uint8_t>(op & 3), false, vd, dest, vs, vt, acc, q, i);
        return;
    case 0x08:
    case 0x09:
    case 0x0A:
    case 0x0B: // MADDbc
        execFmac(FmacKind::Madd, FmacOperand::Broadcast, static_cast<uint8_t>(op & 3), false, vd, dest, vs, vt, acc, q, i);
        return;
    case 0x0C:
    case 0x0D:
    case 0x0E:
    case 0x0F: // MSUBbc
        execFmac(FmacKind::Msub, FmacOperand::Broadcast, static_cast<uint8_t>(op & 3), false, vd, dest, vs, vt, acc, q, i);
        return;
    case 0x10:
    case 0x11:
    case 0x12:
    case 0x13: // MAXbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] > bc) ? vs[c] : bc;
        applyDest(vd, result, dest);
        return;
    }
    case 0x14:
    case 0x15:
    case 0x16:
    case 0x17: // MINIbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] < bc) ? vs[c] : bc;
        applyDest(vd, result, dest);
        return;
    }
    case 0x18:
    case 0x19:
    case 0x1A:
    case 0x1B: // MULbc
        execFmac(FmacKind::Mul, FmacOperand::Broadcast, static_cast<uint8_t>(op & 3), false, vd, dest, vs, vt, acc, q, i);
        return;
    case 0x1C: // MULq
        execFmac(FmacKind::Mul, FmacOperand::Q, 0u, false, vd, dest, vs, vt, acc, q, i);
        return;
    case 0x1D: // MAXi
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] > i) ? vs[c] : i;
        applyDest(vd, result, dest);
        return;
    case 0x1E: // MULi
        execFmac(FmacKind::Mul, FmacOperand::I, 0u, false, vd, dest, vs, vt, acc, q, i);
        return;
    case 0x1F: // MINIi
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] < i) ? vs[c] : i;
        applyDest(vd, result, dest);
        return;
    case 0x20: // ADDq
        execFmac(FmacKind::Add, FmacOperand::Q, 0u, false, vd, dest, vs, vt, acc, q, i);
        return;
    case 0x21: // MADDq
        execFmac(FmacKind::Madd, FmacOperand::Q, 0u, false, vd, dest, vs, vt, acc, q, i);
        return;
    case 0x22: // ADDi
        execFmac(FmacKind::Add, FmacOperand::I, 0u, false, vd, dest, vs, vt, acc, q, i);
        return;
    case 0x23: // MADDi
        execFmac(FmacKind::Madd, FmacOperand::I, 0u, false, vd, dest, vs, vt, acc, q, i);
        return;
    case 0x24: // SUBq
        execFmac(FmacKind::Sub, FmacOperand::Q, 0u, false, vd, dest, vs, vt, acc, q, i);
        return;
    case 0x25: // MSUBq
        execFmac(FmacKind::Msub, FmacOperand::Q, 0u, false, vd, dest, vs, vt, acc, q, i);
        return;
    case 0x26: // SUBi
        execFmac(FmacKind::Sub, FmacOperand::I, 0u, false, vd, dest, vs, vt, acc, q, i);
        return;
    case 0x27: // MSUBi
        execFmac(FmacKind::Msub, FmacOperand::I, 0u, false, vd, dest, vs, vt, acc, q, i);
        return;
    case 0x28: // ADD
        execFmac(FmacKind::Add, FmacOperand::Vector, 0u, false, vd, dest, vs, vt, acc, q, i);
        return;
    case 0x29: // MADD
        execFmac(FmacKind::Madd, FmacOperand::Vector, 0u, false, vd, dest, vs, vt, acc, q, i);
        return;
    case 0x2A: // MUL
        execFmac(FmacKind::Mul, FmacOperand::Vector, 0u, false, vd, dest, vs, vt, acc, q, i);
        return;
    case 0x2B: // MAX
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] > vt[c]) ? vs[c] : vt[c];
        applyDest(vd, result, dest);
        return;
    case 0x2C: // SUB
        execFmac(FmacKind::Sub, FmacOperand::Vector, 0u, false, vd, dest, vs, vt, acc, q, i);
        return;
    case 0x2D: // MSUB
        execFmac(FmacKind::Msub, FmacOperand::Vector, 0u, false, vd, dest, vs, vt, acc, q, i);
        return;
    case 0x2E: // OPMSUB
        execFmac(FmacKind::Opmsub, FmacOperand::Vector, 0u, false, vd, dest, vs, vt, acc, q, i);
        return;
    case 0x2F: // MINI
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] < vt[c]) ? vs[c] : vt[c];
        applyDest(vd, result, dest);
        return;

    // Upper special group (low op 0x3C..0x3F).
    // Like lower1 special, the real selector is not just bits 5:0.  Dobie decodes:
    //   op = (instr & 0x3) | ((instr >> 4) & 0x7C)
    // Several instructions in this group also use FT as the destination, not FD.
    case 0x3C:
    case 0x3D:
    case 0x3E:
    case 0x3F:
    {
        const uint8_t specialOp = static_cast<uint8_t>((instr & 0x3u) | ((instr >> 4) & 0x7Cu));
        float *vtDest = m_state.vf[ft];

        switch (specialOp)
        {
        case 0x00:
        case 0x01:
        case 0x02:
        case 0x03: // ADDAbc
            execFmac(FmacKind::Add, FmacOperand::Broadcast, static_cast<uint8_t>(specialOp & 3), true, nullptr, dest, vs, vt, acc, q, i);
            return;
        case 0x04:
        case 0x05:
        case 0x06:
        case 0x07: // SUBAbc
            execFmac(FmacKind::Sub, FmacOperand::Broadcast, static_cast<uint8_t>(specialOp & 3), true, nullptr, dest, vs, vt, acc, q, i);
            return;
        case 0x08:
        case 0x09:
        case 0x0A:
        case 0x0B: // MADDAbc
            execFmac(FmacKind::Madd, FmacOperand::Broadcast, static_cast<uint8_t>(specialOp & 3), true, nullptr, dest, vs, vt, acc, q, i);
            return;
        case 0x0C:
        case 0x0D:
        case 0x0E:
        case 0x0F: // MSUBAbc
            execFmac(FmacKind::Msub, FmacOperand::Broadcast, static_cast<uint8_t>(specialOp & 3), true, nullptr, dest, vs, vt, acc, q, i);
            return;
        case 0x10: // ITOF0
            for (int c = 0; c < 4; c++)
            {
                int32_t iv;
                std::memcpy(&iv, &m_state.vf[fs][c], 4);
                result[c] = static_cast<float>(iv);
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x11: // ITOF4
            for (int c = 0; c < 4; c++)
            {
                int32_t iv;
                std::memcpy(&iv, &m_state.vf[fs][c], 4);
                result[c] = static_cast<float>(iv) / 16.0f;
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x12: // ITOF12
            for (int c = 0; c < 4; c++)
            {
                int32_t iv;
                std::memcpy(&iv, &m_state.vf[fs][c], 4);
                result[c] = static_cast<float>(iv) / 4096.0f;
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x13: // ITOF15
            for (int c = 0; c < 4; c++)
            {
                int32_t iv;
                std::memcpy(&iv, &m_state.vf[fs][c], 4);
                result[c] = static_cast<float>(iv) / 32768.0f;
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x14: // FTOI0
            for (int c = 0; c < 4; c++)
            {
                int32_t iv = vuFloatToInt(vs[c], 1.0f);
                std::memcpy(&result[c], &iv, 4);
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x15: // FTOI4
            for (int c = 0; c < 4; c++)
            {
                int32_t iv = vuFloatToInt(vs[c], 16.0f);
                std::memcpy(&result[c], &iv, 4);
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x16: // FTOI12
            for (int c = 0; c < 4; c++)
            {
                int32_t iv = vuFloatToInt(vs[c], 4096.0f);
                std::memcpy(&result[c], &iv, 4);
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x17: // FTOI15
            for (int c = 0; c < 4; c++)
            {
                int32_t iv = vuFloatToInt(vs[c], 32768.0f);
                std::memcpy(&result[c], &iv, 4);
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x18:
        case 0x19:
        case 0x1A:
        case 0x1B: // MULAbc
            execFmac(FmacKind::Mul, FmacOperand::Broadcast, static_cast<uint8_t>(specialOp & 3), true, nullptr, dest, vs, vt, acc, q, i);
            return;
        case 0x1C: // MULAq
            execFmac(FmacKind::Mul, FmacOperand::Q, 0u, true, nullptr, dest, vs, vt, acc, q, i);
            return;
        case 0x1D: // ABS
            for (int c = 0; c < 4; c++)
                result[c] = std::fabs(vs[c]);
            applyDest(vtDest, result, dest);
            return;
        case 0x1E: // MULAi
            execFmac(FmacKind::Mul, FmacOperand::I, 0u, true, nullptr, dest, vs, vt, acc, q, i);
            return;
        case 0x1F: // CLIP
        {
            uint32_t wBits = 0u;
            std::memcpy(&wBits, &m_state.vf[ft][3], sizeof(wBits));
            const int32_t limit = (wBits & 0x7F800000u) != 0u ? static_cast<int32_t>(wBits & 0x7FFFFFFFu) : 0x007FFFFF;

            const auto exceedsClipPlane = [limit](float value, uint32_t signMask)
            {
                uint32_t bits = 0u;
                std::memcpy(&bits, &value, sizeof(bits));
                bits ^= signMask;
                int32_t orderedBits = 0;
                std::memcpy(&orderedBits, &bits, sizeof(orderedBits));
                return orderedBits > limit;
            };

            uint32_t flags = 0u;
            if (exceedsClipPlane(m_state.vf[fs][0], 0x00000000u))
                flags |= 0x01u;
            if (exceedsClipPlane(m_state.vf[fs][0], 0x80000000u))
                flags |= 0x02u;
            if (exceedsClipPlane(m_state.vf[fs][1], 0x00000000u))
                flags |= 0x04u;
            if (exceedsClipPlane(m_state.vf[fs][1], 0x80000000u))
                flags |= 0x08u;
            if (exceedsClipPlane(m_state.vf[fs][2], 0x00000000u))
                flags |= 0x10u;
            if (exceedsClipPlane(m_state.vf[fs][2], 0x80000000u))
                flags |= 0x20u;
            queueClip(flags);
            return;
        }
        case 0x20: // ADDAq
            execFmac(FmacKind::Add, FmacOperand::Q, 0u, true, nullptr, dest, vs, vt, acc, q, i);
            return;
        case 0x21: // MADDAq
            execFmac(FmacKind::Madd, FmacOperand::Q, 0u, true, nullptr, dest, vs, vt, acc, q, i);
            return;
        case 0x22: // ADDAi
            execFmac(FmacKind::Add, FmacOperand::I, 0u, true, nullptr, dest, vs, vt, acc, q, i);
            return;
        case 0x23: // MADDAi
            execFmac(FmacKind::Madd, FmacOperand::I, 0u, true, nullptr, dest, vs, vt, acc, q, i);
            return;
        case 0x24: // SUBAq
            execFmac(FmacKind::Sub, FmacOperand::Q, 0u, true, nullptr, dest, vs, vt, acc, q, i);
            return;
        case 0x25: // MSUBAq
            execFmac(FmacKind::Msub, FmacOperand::Q, 0u, true, nullptr, dest, vs, vt, acc, q, i);
            return;
        case 0x26: // SUBAi
            execFmac(FmacKind::Sub, FmacOperand::I, 0u, true, nullptr, dest, vs, vt, acc, q, i);
            return;
        case 0x27: // MSUBAi
            execFmac(FmacKind::Msub, FmacOperand::I, 0u, true, nullptr, dest, vs, vt, acc, q, i);
            return;
        case 0x28: // ADDA
            execFmac(FmacKind::Add, FmacOperand::Vector, 0u, true, nullptr, dest, vs, vt, acc, q, i);
            return;
        case 0x29: // MADDA
            execFmac(FmacKind::Madd, FmacOperand::Vector, 0u, true, nullptr, dest, vs, vt, acc, q, i);
            return;
        case 0x2A: // MULA
            execFmac(FmacKind::Mul, FmacOperand::Vector, 0u, true, nullptr, dest, vs, vt, acc, q, i);
            return;
        case 0x2C: // SUBA
            execFmac(FmacKind::Sub, FmacOperand::Vector, 0u, true, nullptr, dest, vs, vt, acc, q, i);
            return;
        case 0x2D: // MSUBA
            execFmac(FmacKind::Msub, FmacOperand::Vector, 0u, true, nullptr, dest, vs, vt, acc, q, i);
            return;
        case 0x2E: // OPMULA
            execFmac(FmacKind::Opmula, FmacOperand::Vector, 0u, true, nullptr, dest, vs, vt, acc, q, i);
            return;
        case 0x2F:
        case 0x30: // NOP
            return;
        default:
            reportReservedInstruction(true, instr);
            return;
        }
    }

    case 0x30:
    case 0x31:
    case 0x32:
    case 0x33:
    default:
        reportReservedInstruction(true, instr);
        return;
    }
}

void VU1Interpreter::execFmac(FmacKind kind, FmacOperand operand, uint8_t bcLane, bool toAcc, float *vd, uint8_t dest,
                              const float *vs, const float *vt, const float *acc, float q, float i)
{
    // vs/vt/acc/q/i arrive normalized. Expression shapes match the per-opcode code this replaced
    // (the float result in round-to-zero; the exact long double value for the flags), so the
    // compiler contracts them into the same fused operations.
    static constexpr uint8_t kCrossLeft[4] = {1u, 2u, 0u, 3u};
    static constexpr uint8_t kCrossRight[4] = {2u, 0u, 1u, 3u};
    const bool cross = kind == FmacKind::Opmsub || kind == FmacKind::Opmula;
    const bool productSum = kind == FmacKind::Madd || kind == FmacKind::Msub || kind == FmacKind::Opmsub;
    float result[4]{};
    uint8_t laneFlags[4]{};
    uint32_t extraSticky = 0u;
    for (uint32_t component = 0; component < 4u; ++component)
    {
        if ((dest & (1u << (3u - component))) == 0u)
            continue;
        const float s = vs[cross ? kCrossLeft[component] : component];
        float r;
        switch (operand)
        {
        case FmacOperand::Broadcast:
            r = broadcast(vt, bcLane);
            break;
        case FmacOperand::Q:
            r = q;
            break;
        case FmacOperand::I:
            r = i;
            break;
        default:
            r = vt[cross ? kCrossRight[component] : component];
            break;
        }
        const float a = acc[component];
        const long double ls = static_cast<long double>(s);
        const long double lr = static_cast<long double>(r);
        const long double la = static_cast<long double>(a);
        float value;
        long double exact;
        switch (kind)
        {
        case FmacKind::Add:
            value = s + r;
            exact = ls + lr;
            break;
        case FmacKind::Sub:
            value = s - r;
            exact = ls - lr;
            break;
        case FmacKind::Madd:
            value = a + s * r;
            exact = la + ls * lr;
            break;
        case FmacKind::Msub:
            value = a - s * r;
            exact = la - ls * lr;
            break;
        case FmacKind::Mul:
            value = s * r;
            exact = ls * lr;
            break;
        case FmacKind::Opmsub:
            value = component == 3u ? 0.0f : a - s * r;
            exact = component == 3u ? 0.0L : la - ls * lr;
            break;
        default: // Opmula
            value = component == 3u ? 0.0f : s * r;
            exact = component == 3u ? 0.0L : ls * lr;
            break;
        }
        laneFlags[component] = normalizeFmacExactResult(value, exact);
        result[component] = value;
        if (productSum)
        {
            // Product-sum instructions report Z/S/U/O from the add/subtract result as current
            // flags, while every product condition accumulates into the matching sticky flag.
            float product = 0.0f;
            extraSticky |= normalizeFmacExactResult(product, ls * lr) & 0xFu;
        }
    }
    updateFmacFlags(laneFlags, dest, extraSticky);
    if (toAcc)
        applyDestAcc(result, dest);
    else
        applyDest(vd, result, dest);
}
