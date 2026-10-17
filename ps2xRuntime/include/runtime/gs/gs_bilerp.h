#pragma once

#include <cstdint>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define PS2X_GS_BILERP_NEON 1
#endif

namespace GSBilerp
{
    // The GS bilinear filter on one 8-bit channel: horizontal then vertical, each step
    // a + ((b - a) * f >> 4) with a 4-bit weight and an arithmetic (flooring) shift, as PCSX2's
    // GSDrawScanline (lerp16_4). The texel coordinate keeps 4 fraction bits (12.4), so these
    // weights are all there is.
    inline uint8_t lerpChannel(uint8_t c00, uint8_t c10, uint8_t c01, uint8_t c11, int fx, int fy)
    {
        auto lerp = [](int a, int b, int f) { return a + (((b - a) * f) >> 4); };
        return static_cast<uint8_t>(lerp(lerp(c00, c10, fx), lerp(c01, c11, fx), fy));
    }

    // lerpChannel on the four channels of a packed RGBA texel at once. Each channel sits in a 16-bit lane of a
    // 64-bit word. a + (((b - a) * f) >> 4) equals (a * (16 - f) + b * f) >> 4 (16a is a multiple of 16), and that
    // sum is at most 255 * 16 = 4080, so no lane carries into the next; the mask after each shift drops the
    // bits the shift moves in from the lane above.
    inline uint32_t bilerpTexelsPortable(uint32_t c00, uint32_t c10, uint32_t c01, uint32_t c11, int fx, int fy)
    {
        constexpr uint64_t kLanes = 0x00FF00FF00FF00FFull;
        auto spread = [](uint32_t c)
        {
            uint64_t x = c;
            x = (x | (x << 16)) & 0x0000FFFF0000FFFFull;
            return (x | (x << 8)) & kLanes;
        };
        const uint64_t ix = static_cast<uint64_t>(16 - fx), wx = static_cast<uint64_t>(fx);
        const uint64_t iy = static_cast<uint64_t>(16 - fy), wy = static_cast<uint64_t>(fy);
        const uint64_t top = ((spread(c00) * ix + spread(c10) * wx) >> 4) & kLanes;
        const uint64_t bottom = ((spread(c01) * ix + spread(c11) * wx) >> 4) & kLanes;
        uint64_t x = ((top * iy + bottom * wy) >> 4) & kLanes;
        x = (x | (x >> 8)) & 0x0000FFFF0000FFFFull;
        return static_cast<uint32_t>(x | (x >> 16));
    }

    // bilerpTexelsPortable, with NEON where there is NEON.
    inline uint32_t bilerpTexels(uint32_t c00, uint32_t c10, uint32_t c01, uint32_t c11, int fx, int fy)
    {
#if defined(PS2X_GS_BILERP_NEON)
        // The same sums in NEON's 16-bit lanes: both rows' horizontal steps in one widening
        // multiply-accumulate (c00|c01 against c10|c11), then the vertical step.
        const uint8x8_t left = vcreate_u8(static_cast<uint64_t>(c00) | (static_cast<uint64_t>(c01) << 32));
        const uint8x8_t right = vcreate_u8(static_cast<uint64_t>(c10) | (static_cast<uint64_t>(c11) << 32));
        const uint8x8_t rows = vshrn_n_u16(vmlal_u8(vmull_u8(left, vdup_n_u8(static_cast<uint8_t>(16 - fx))), right,
                                                    vdup_n_u8(static_cast<uint8_t>(fx))),
                                           4); // top in lanes 0-3, bottom in lanes 4-7
        const uint8x8_t top = rows;
        const uint8x8_t bottom = vext_u8(rows, rows, 4);
        const uint8x8_t out = vshrn_n_u16(vmlal_u8(vmull_u8(top, vdup_n_u8(static_cast<uint8_t>(16 - fy))), bottom,
                                                   vdup_n_u8(static_cast<uint8_t>(fy))),
                                          4);
        return vget_lane_u32(vreinterpret_u32_u8(out), 0);
#else
        return bilerpTexelsPortable(c00, c10, c01, c11, fx, fy);
#endif
    }
}
