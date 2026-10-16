#include "ps2_vu1_upper.h"

#include <cmath>
#include <cstring>
#include <limits>

void VU1Interpreter::execUpper(uint32_t instr)
{
    execUpperInline(instr);
}

void VU1Interpreter::execFmac(FmacKind kind, FmacOperand operand, uint8_t bcLane, bool toAcc, float *vd, uint8_t dest,
                              const float *vs, const float *vt, const float *acc, float q, float i)
{
    // vs/vt/acc/q/i arrive normalized. Expression shapes match the per-opcode code this replaced
    // (the float result in round-to-zero; the flags from the result in double, also rounded to
    // zero), so the compiler contracts them into the same fused operations. Double, not long
    // double: that is 80-bit on x86-64 GCC/Clang and gave other flags there (e.g. no overflow for
    // FLT_MAX + 1e20 in double, overflow in 80-bit).
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
        const double ls = static_cast<double>(s);
        const double lr = static_cast<double>(r);
        const double la = static_cast<double>(a);
        float value;
        double exact;
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
            exact = component == 3u ? 0.0 : la - ls * lr;
            break;
        default: // Opmula
            value = component == 3u ? 0.0f : s * r;
            exact = component == 3u ? 0.0 : ls * lr;
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

