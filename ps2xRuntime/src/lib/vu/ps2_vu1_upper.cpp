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
    // vs/vt/acc/q/i arrive normalized. The float result in round-to-zero; the flags from the
    // result in double (exact for a product of two floats). Double, not long
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
        if (productSum)
        {
            // Two roundings, like the hardware: the product chopped (and clamped / flushed like
            // any result), then the add chopped. Current flags come from the add; every product
            // condition accumulates into the matching sticky flag.
            float product = s * r;
            extraSticky |= normalizeFmacExactResult(product, ls * lr) & 0xFu;
            const double lp = static_cast<double>(product);
            if (kind == FmacKind::Madd)
            {
                value = a + product;
                exact = la + lp;
            }
            else
            {
                value = a - product;
                exact = la - lp;
            }
            if (kind == FmacKind::Opmsub && component == 3u)
            {
                value = 0.0f;
                exact = 0.0;
            }
        }
        else
        {
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
            case FmacKind::Mul:
                value = s * r;
                exact = ls * lr;
                break;
            default: // Opmula
                value = component == 3u ? 0.0f : s * r;
                exact = component == 3u ? 0.0 : ls * lr;
                break;
            }
        }
        laneFlags[component] = normalizeFmacExactResult(value, exact);
        result[component] = value;
    }
    updateFmacFlags(laneFlags, dest, extraSticky);
    if (toAcc)
        applyDestAcc(result, dest);
    else
        applyDest(vd, result, dest);
}

