#ifndef PS2_VU1_UPPER_H
#define PS2_VU1_UPPER_H
// Upper-slot instructions. execUpperInline is the whole decoder; execUpper (ps2_vu1_upper.cpp) is
// its out-of-line copy for the interpreter, while translated VU1 blocks (T-0121) inline it with a
// constant instruction word, so only that instruction's case is left.

#include <bit>
#include "runtime/ps2_vu1.h"
#include "ps2_vu1_detail.h"
#include "ps2_vu1_decode.h"

#include <cmath>
#include <cstring>
#include <limits>

namespace vu1_upper_detail
{
    inline int32_t vuFloatToInt(float value, float scale)
    {
        const double scaled = static_cast<double>(value) * static_cast<double>(scale);
        if (scaled >= static_cast<double>(std::numeric_limits<int32_t>::max()))
            return std::numeric_limits<int32_t>::max();
        if (scaled <= static_cast<double>(std::numeric_limits<int32_t>::min()))
            return std::numeric_limits<int32_t>::min();
        return static_cast<int32_t>(scaled);
    }

    // ITOF's conversion, rounded toward zero like every VU result, without relying on the host's
    // rounding mode: the low bits that do not fit a float's 24-bit significand are dropped first,
    // so the conversion itself is exact. (Inlined after FTOI, the compiler knows a saturated
    // 0x7FFFFFFF and folds a plain cast to 2^31, rounded to nearest at compile time.)
    inline float vuIntToFloat(int32_t value)
    {
        uint32_t magnitude = value < 0 ? 0u - static_cast<uint32_t>(value) : static_cast<uint32_t>(value);
        if (magnitude >= (1u << 24))
        {
            const int shift = 8 - std::countl_zero(magnitude); // bits below the 24 that are kept
            magnitude &= ~((1u << shift) - 1u);
        }
        const float f = static_cast<float>(magnitude);
        return value < 0 ? -f : f;
    }
}

// ============================================================================
// Upper instructions (FMAC pipeline)
// ============================================================================
[[gnu::always_inline]] inline void VU1Interpreter::execUpperInline(uint32_t instr)
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
                result[c] = vu1_upper_detail::vuIntToFloat(iv);
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x11: // ITOF4
            for (int c = 0; c < 4; c++)
            {
                int32_t iv;
                std::memcpy(&iv, &m_state.vf[fs][c], 4);
                result[c] = vu1_upper_detail::vuIntToFloat(iv) / 16.0f;
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x12: // ITOF12
            for (int c = 0; c < 4; c++)
            {
                int32_t iv;
                std::memcpy(&iv, &m_state.vf[fs][c], 4);
                result[c] = vu1_upper_detail::vuIntToFloat(iv) / 4096.0f;
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x13: // ITOF15
            for (int c = 0; c < 4; c++)
            {
                int32_t iv;
                std::memcpy(&iv, &m_state.vf[fs][c], 4);
                result[c] = vu1_upper_detail::vuIntToFloat(iv) / 32768.0f;
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x14: // FTOI0
            for (int c = 0; c < 4; c++)
            {
                int32_t iv = vu1_upper_detail::vuFloatToInt(vs[c], 1.0f);
                std::memcpy(&result[c], &iv, 4);
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x15: // FTOI4
            for (int c = 0; c < 4; c++)
            {
                int32_t iv = vu1_upper_detail::vuFloatToInt(vs[c], 16.0f);
                std::memcpy(&result[c], &iv, 4);
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x16: // FTOI12
            for (int c = 0; c < 4; c++)
            {
                int32_t iv = vu1_upper_detail::vuFloatToInt(vs[c], 4096.0f);
                std::memcpy(&result[c], &iv, 4);
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x17: // FTOI15
            for (int c = 0; c < 4; c++)
            {
                int32_t iv = vu1_upper_detail::vuFloatToInt(vs[c], 32768.0f);
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

#endif
