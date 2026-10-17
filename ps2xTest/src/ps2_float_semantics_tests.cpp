#include "MiniTest.h"
#include "ps2recomp/code_generator.h"
#include "ps2recomp/instructions.h"
#include "ps2recomp/types.h"
#include "ps2_runtime_macros.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>

using namespace ps2recomp;

namespace
{
    constexpr uint32_t kPlusMax = 0x7FFFFFFFu;
    constexpr uint32_t kMinusMax = 0xFFFFFFFFu;
    // 2^128: the host's +Inf pattern, an ordinary number on the PS2.
    constexpr uint32_t kTwoPow128 = 0x7F800000u;

    uint32_t bitsOf(float f)
    {
        uint32_t b;
        std::memcpy(&b, &f, sizeof(b));
        return b;
    }

    float floatOf(uint32_t b)
    {
        float f;
        std::memcpy(&f, &b, sizeof(f));
        return f;
    }

    Instruction makeCop1S(uint32_t function, uint8_t fd, uint8_t fs, uint8_t ft)
    {
        Instruction inst{};
        inst.opcode = OPCODE_COP1;
        inst.rs = COP1_S;
        inst.rt = ft;
        inst.rd = fs;
        inst.sa = fd;
        inst.function = function;
        return inst;
    }

    uint32_t laneBits(__m128 v, int lane)
    {
        float lanes[4];
        _mm_storeu_ps(lanes, v);
        return bitsOf(lanes[lane]);
    }

    struct HardwareVector
    {
        const char *op;
        uint32_t fs;
        uint32_t ft;
        uint32_t result;
        // 0: must match exactly.
        // 1: one last-bit difference the runtime keeps on purpose, because it follows PCSX2 there: MUL.S chops
        //    the exact product, where the hardware multiplier can come out one lower; RSQRT.S chops its root
        //    and its divide, where the hardware divider rounds by its own rule (1/sqrt(25) rounds up, 3/sqrt(3)
        //    does not).
        int known;
    };

    const HardwareVector kHardwareVectors[] = {
#include "ps2_float_hw_vectors.inc"
    };

    uint32_t runHardwareOp(const HardwareVector &v)
    {
        const float fs = floatOf(v.fs);
        const float ft = floatOf(v.ft);
        const std::string op = v.op;
        if (op == "add")
            return bitsOf(FPU_ADD_S(fs, ft));
        if (op == "sub")
            return bitsOf(FPU_SUB_S(fs, ft));
        if (op == "mul")
            return bitsOf(FPU_MUL_S(fs, ft));
        if (op == "div")
            return bitsOf(FPU_DIV_S(fs, ft));
        if (op == "sqrt")
            return bitsOf(FPU_SQRT_S(fs));
        return bitsOf(FPU_RSQRT_S(fs, ft));
    }

    std::string describe(const HardwareVector &v, uint32_t got)
    {
        char text[160];
        std::snprintf(text, sizeof(text), "%s %08x, %08x: PS2 gives %08x, got %08x", v.op, v.fs, v.ft, v.result, got);
        return text;
    }
}

void register_ps2_float_semantics_tests()
{
    MiniTest::Case("PS2FloatSemantics", [](TestCase &tc)
    {
        tc.Run("FPU results match the PS2 hardware results from ps2autotests", [](TestCase &t)
        {
            for (const HardwareVector &v : kHardwareVectors)
            {
                const uint32_t got = runHardwareOp(v);
                if (v.known == 1)
                {
                    const uint32_t distance = got > v.result ? got - v.result : v.result - got;
                    t.IsTrue(distance == 1u, describe(v, got));
                }
                else
                {
                    t.Equals(got, v.result, describe(v, got));
                }
            }
        });

        tc.Run("FPU ADD.S, SUB.S and MUL.S round toward zero", [](TestCase &t)
        {
            // TC2's DOF blur takes the fraction of its blur amount as x - ((x + 2^23) - 2^23); chopping keeps
            // 0.7 + 2^23 at 2^23, so the fraction stays 0.7 (round to nearest makes it -0.3 and no pass runs).
            const float big = floatOf(0x4B000000u);
            const float whole = FPU_SUB_S(FPU_ADD_S(0.7f, big), big);
            t.Equals(bitsOf(whole), 0x00000000u, "(0.7 + 2^23) - 2^23 should be 0");
            t.Equals(bitsOf(FPU_SUB_S(0.7f, whole)), bitsOf(0.7f), "the fraction of 0.7 should be 0.7");
            t.Equals(bitsOf(FPU_ADD_S(floatOf(0xBFFFFFFFu), floatOf(0xBF800000u))), 0xC03FFFFFu,
                     "-1.9999999 - 1 should chop to -2.9999998, not round to -3");
            t.Equals(bitsOf(FPU_MUL_S(floatOf(0x3FA265B1u), floatOf(0x3F9027C4u))), 0x3FB6E4DAu,
                     "the product should chop (nearest gives 3fb6e4db)");
            t.Equals(bitsOf(FPU_MUL_S(floatOf(0xBFA265B1u), floatOf(0x3F9027C4u))), 0xBFB6E4DAu,
                     "a negative product should chop toward zero too");
            t.Equals(bitsOf(FPU_ADD_S(FPU_MUL_S(floatOf(0x3FA265B1u), floatOf(0x3F9027C4u)), 0.0f)), 0x3FB6E4DAu,
                     "MADD.S is the chopped product, then the chopped add");
        });

        tc.Run("FPU ADD.S keeps one guard bit of the smaller operand", [](TestCase &t)
        {
            // Exponent difference 2: the low bit of 0.25 + 1 ulp is dropped before the subtraction, so
            // 1 - 0x3E800001 is exactly 0.75 (the exact difference would chop to 0x3F3FFFFF).
            t.Equals(bitsOf(FPU_SUB_S(1.0f, floatOf(0x3E800001u))), 0x3F400000u, "1 - (0.25 + ulp) should be 0.75");
            t.Equals(bitsOf(FPU_ADD_S(floatOf(0xBE800001u), 1.0f)), 0x3F400000u, "the alignment does not depend on operand order");
            // Exponent difference 1 keeps every bit: 1 - (0.5 + ulp) is exact.
            t.Equals(bitsOf(FPU_SUB_S(1.0f, floatOf(0x3F000001u))), 0x3EFFFFFEu, "1 - (0.5 + ulp) should be exact");
            // Exponent difference 25 or more leaves only the sign.
            t.Equals(bitsOf(FPU_SUB_S(1.0f, floatOf(0x33000001u))), 0x3F800000u, "1 - (2^-25 + ulp) should be 1");
            t.Equals(bitsOf(FPU_ADD_S(floatOf(0x00400000u), 1.0f)), 0x3F800000u, "an exponent-0 operand reads as zero");
            t.Equals(bitsOf(FPU_MUL_S(floatOf(0x00400000u), -1.0f)), 0x80000000u, "an exponent-0 factor reads as zero");
        });

        tc.Run("FPU CVT.S.W chops integers wider than 24 bits", [](TestCase &t)
        {
            t.Equals(bitsOf(FPU_CVT_S_W(16777217)), 0x4B800000u, "16777217 should become 16777216");
            t.Equals(bitsOf(FPU_CVT_S_W(16777219)), 0x4B800001u, "16777219 should become 16777218, not 16777220");
            t.Equals(bitsOf(FPU_CVT_S_W(-16777219)), 0xCB800001u, "-16777219 should become -16777218");
            t.Equals(bitsOf(FPU_CVT_S_W(0x7FFFFFFF)), 0x4EFFFFFFu, "INT32_MAX should become 2147483520");
            t.Equals(bitsOf(FPU_CVT_S_W(INT32_MIN)), 0xCF000000u, "INT32_MIN is exact");
        });

        tc.Run("exponent 255 is an ordinary exponent, not Inf or NaN", [](TestCase &t)
        {
            const float twoPow128 = floatOf(kTwoPow128);
            t.Equals(bitsOf(FPU_ADD_S(twoPow128, 0.0f)), kTwoPow128, "2^128 + 0 should stay 2^128");
            t.Equals(bitsOf(FPU_MUL_S(twoPow128, 0.5f)), 0x7F000000u, "2^128 * 0.5 should be 2^127");
            t.Equals(bitsOf(FPU_MUL_S(floatOf(0x7F000000u), 2.0f)), kTwoPow128, "2^127 * 2 should be 2^128, not saturated");
            t.Equals(bitsOf(FPU_ADD_S(floatOf(0x7F000000u), floatOf(0x7F000000u))), kTwoPow128, "2^127 + 2^127 should be 2^128");
            t.Equals(bitsOf(FPU_SUB_S(floatOf(0x7FC00000u), twoPow128)), 0x7F000000u, "1.5*2^128 - 2^128 should be 2^127");
            t.Equals(bitsOf(FPU_DIV_S(floatOf(0x7FC00000u), 3.0f)), 0x7F000000u, "1.5*2^128 / 3 should be 2^127");
            t.Equals(bitsOf(FPU_SQRT_S(twoPow128)), 0x5F800000u, "sqrt(2^128) should be 2^64");
        });

        tc.Run("FPU results saturate at 0x7FFFFFFF and flush below 2^-126", [](TestCase &t)
        {
            const float maxf = floatOf(kPlusMax);
            const float minf = floatOf(kMinusMax);
            t.Equals(bitsOf(FPU_ADD_S(maxf, maxf)), kPlusMax, "max + max should saturate to +max");
            t.Equals(bitsOf(FPU_ADD_S(minf, minf)), kMinusMax, "-max + -max should saturate to -max");
            t.Equals(bitsOf(FPU_SUB_S(maxf, minf)), kPlusMax, "max - (-max) should saturate to +max");
            t.Equals(bitsOf(FPU_MUL_S(maxf, 2.0f)), kPlusMax, "max * 2 should saturate to +max");
            t.Equals(bitsOf(FPU_MUL_S(maxf, -2.0f)), kMinusMax, "max * -2 should saturate to -max");
            t.Equals(bitsOf(FPU_DIV_S(maxf, 0.5f)), kPlusMax, "max / 0.5 should saturate to +max");
            t.Equals(bitsOf(FPU_MUL_S(floatOf(0x00800000u), 0.5f)), 0x00000000u, "2^-126 * 0.5 should flush to +0");
            t.Equals(bitsOf(FPU_MUL_S(floatOf(0x80800000u), 0.5f)), 0x80000000u, "-2^-126 * 0.5 should flush to -0");
            t.Equals(bitsOf(FPU_MUL_S(floatOf(0x00000001u), 1.0f)), 0x00000000u, "a result with exponent 0 should flush to +0");
            t.Equals(bitsOf(FPU_ADD_S(1.0f, 2.0f)), bitsOf(3.0f), "ordinary addition should be unchanged");
            t.Equals(bitsOf(FPU_MUL_S(1.5f, 4.0f)), bitsOf(6.0f), "ordinary multiplication should be unchanged");
        });

        tc.Run("FPU DIV.S by zero gives +-max with the sign of fs xor ft", [](TestCase &t)
        {
            t.Equals(bitsOf(FPU_DIV_S(1.0f, 0.0f)), kPlusMax, "1/+0 should be +max");
            t.Equals(bitsOf(FPU_DIV_S(-1.0f, 0.0f)), kMinusMax, "-1/+0 should be -max");
            t.Equals(bitsOf(FPU_DIV_S(1.0f, -0.0f)), kMinusMax, "1/-0 should be -max");
            t.Equals(bitsOf(FPU_DIV_S(-1.0f, -0.0f)), kPlusMax, "-1/-0 should be +max");
            t.Equals(bitsOf(FPU_DIV_S(0.0f, 0.0f)), kPlusMax, "0/0 should be +max");
            t.Equals(bitsOf(FPU_DIV_S(1.0f, floatOf(0x00000001u))), kPlusMax, "an exponent-0 divisor should count as zero");
            t.Equals(bitsOf(FPU_DIV_S(6.0f, 3.0f)), bitsOf(2.0f), "ordinary division should be unchanged");
        });

        tc.Run("FPU RSQRT.S computes fs / sqrt(|ft|) and takes the sign of fs alone on a zero ft", [](TestCase &t)
        {
            t.Equals(bitsOf(FPU_RSQRT_S(4.0f, 16.0f)), bitsOf(1.0f), "4 / sqrt(16) should be 1");
            t.Equals(bitsOf(FPU_RSQRT_S(1.0f, 4.0f)), bitsOf(0.5f), "1 / sqrt(4) should be 0.5");
            t.Equals(bitsOf(FPU_RSQRT_S(1.0f, -4.0f)), bitsOf(0.5f), "the sign of the radicand should be ignored");
            t.Equals(bitsOf(FPU_RSQRT_S(2.0f, 0.0f)), kPlusMax, "x / sqrt(+0) should be +max");
            t.Equals(bitsOf(FPU_RSQRT_S(2.0f, -0.0f)), kPlusMax, "x / sqrt(-0) should be +max: the sign of ft is ignored");
            t.Equals(bitsOf(FPU_RSQRT_S(-2.0f, 0.0f)), kMinusMax, "-x / sqrt(+0) should be -max");
            t.Equals(bitsOf(FPU_RSQRT_S(-2.0f, -0.0f)), kMinusMax, "-x / sqrt(-0) should be -max");
        });

        tc.Run("FPU SQRT.S takes the square root of the magnitude", [](TestCase &t)
        {
            t.Equals(bitsOf(FPU_SQRT_S(16.0f)), bitsOf(4.0f), "sqrt(16) should be 4");
            t.Equals(bitsOf(FPU_SQRT_S(-16.0f)), bitsOf(4.0f), "sqrt(-16) should be 4");
            t.Equals(bitsOf(FPU_SQRT_S(-0.0f)), bitsOf(0.0f), "sqrt(-0) should be +0");
        });

        tc.Run("VU0 ADD/SUB/MUL and the Q unit round toward zero, without the FPU's alignment", [](TestCase &t)
        {
            // _mm_set_ps lists lanes 3..0
            const __m128 a = _mm_set_ps(1.0f, floatOf(0xBFFFFFFFu), floatOf(0x3FFFFFFFu), floatOf(0x3FA265B1u));
            const __m128 b = _mm_set_ps(floatOf(0xBE800001u), floatOf(0xBF800000u), 1.0f, floatOf(0x3F9027C4u));
            const __m128 sum = PS2_VADD(a, b);
            t.Equals(laneBits(sum, 1), 0x403FFFFFu, "lane 1: 1.9999999 + 1 should chop to 2.9999998");
            t.Equals(laneBits(sum, 2), 0xC03FFFFFu, "lane 2: -1.9999999 - 1 should chop toward zero");
            // The VU adder keeps every bit of the smaller operand: 0.75 - 2^-25 chops to 0x3F3FFFFF, where the
            // FPU's ADD.S gives exactly 0.75.
            t.Equals(laneBits(sum, 3), 0x3F3FFFFFu, "lane 3: 1 - (0.25 + ulp) should chop the exact difference");
            const __m128 diff = PS2_VSUB(a, _mm_xor_ps(b, _mm_set1_ps(-0.0f)));
            t.Equals(laneBits(diff, 3), 0x3F3FFFFFu, "SUB of the negated operand should give the same result");
            const __m128 prod = PS2_VMUL(a, b);
            t.Equals(laneBits(prod, 0), 0x3FB6E4DAu, "lane 0: the product should chop (nearest gives 3fb6e4db)");

            t.Equals(bitsOf(PS2_VU_DIV_Q(1.0f, 3.0f)), 0x3EAAAAAAu, "DIV 1/3 should chop to 0x3EAAAAAA");
            t.Equals(bitsOf(PS2_VU_DIV_Q(-1.0f, 3.0f)), 0xBEAAAAAAu, "DIV -1/3 should chop toward zero");
            t.Equals(bitsOf(PS2_VU_DIV_Q(1.0f, 4.0f)), bitsOf(0.25f), "an exact quotient is unchanged");
            t.Equals(bitsOf(PS2_VU_SQRT_Q(5.0f)), 0x400F1BBCu, "SQRT 5 should chop to 0x400F1BBC");
            t.Equals(bitsOf(PS2_VU_SQRT_Q(16.0f)), bitsOf(4.0f), "SQRT 16 should be 4");
            t.Equals(bitsOf(PS2_VU_RSQRT_Q(1.0f, 4.0f)), bitsOf(0.5f), "RSQRT 1/sqrt(4) should be 0.5");
            t.Equals(bitsOf(PS2_VU_RSQRT_Q(1.0f, 5.0f)), bitsOf(PS2_VU_DIV_Q(1.0f, floatOf(0x400F1BBCu))),
                     "RSQRT divides by the chopped root");
            t.Equals(bitsOf(PS2_VU_DIV_Q(1.0f, 0.0f)), 0x7F7FFFFFu, "DIV by zero still gives +0x7F7FFFFF");
        });

        tc.Run("VU vector ADD/SUB/MUL follow the same rules in every lane", [](TestCase &t)
        {
            // _mm_set_ps lists lanes 3..0
            const __m128 a = _mm_set_ps(1.0f, floatOf(kTwoPow128), floatOf(kMinusMax), floatOf(kPlusMax));
            const __m128 b = _mm_set_ps(2.0f, 1.0f, floatOf(kMinusMax), floatOf(kPlusMax));

            const __m128 sum = PS2_VADD(a, b);
            t.Equals(laneBits(sum, 0), kPlusMax, "lane 0: max + max should saturate to +max");
            t.Equals(laneBits(sum, 1), kMinusMax, "lane 1: -max + -max should saturate to -max");
            t.Equals(laneBits(sum, 2), kTwoPow128, "lane 2: 2^128 + 1 should stay 2^128");
            t.Equals(laneBits(sum, 3), bitsOf(3.0f), "lane 3: 1 + 2 should be 3");

            const __m128 diff = PS2_VSUB(a, b);
            t.Equals(laneBits(diff, 0), bitsOf(0.0f), "lane 0: max - max should be 0");
            // The VU adder has no alignment step, so the exact 2^128 - 1 chops to the value just below 2^128.
            t.Equals(laneBits(diff, 2), 0x7F7FFFFFu, "lane 2: 2^128 - 1 should chop to 0x7F7FFFFF");
            t.Equals(laneBits(diff, 3), bitsOf(-1.0f), "lane 3: 1 - 2 should be -1");

            const __m128 prod = PS2_VMUL(a, b);
            t.Equals(laneBits(prod, 0), kPlusMax, "lane 0: max * max should saturate to +max");
            t.Equals(laneBits(prod, 1), kPlusMax, "lane 1: -max * -max should saturate to +max");
            t.Equals(laneBits(prod, 2), kTwoPow128, "lane 2: 2^128 * 1 should stay 2^128");
            t.Equals(laneBits(prod, 3), bitsOf(2.0f), "lane 3: 1 * 2 should be 2");

            const __m128 tiny = _mm_set_ps(1.0f, floatOf(0x80800000u), floatOf(0x00800000u), 3.0f);
            const __m128 half = _mm_set1_ps(0.5f);
            const __m128 flushed = PS2_VMUL(tiny, half);
            t.Equals(laneBits(flushed, 0), bitsOf(1.5f), "lane 0: 3 * 0.5 should be 1.5");
            t.Equals(laneBits(flushed, 1), 0x00000000u, "lane 1: 2^-126 * 0.5 should flush to +0");
            t.Equals(laneBits(flushed, 2), 0x80000000u, "lane 2: -2^-126 * 0.5 should flush to -0");
            t.Equals(laneBits(flushed, 3), bitsOf(0.5f), "lane 3: 1 * 0.5 should be 0.5");
        });

        tc.Run("DIV.S and RSQRT.S translate to the PS2 semantics helpers", [](TestCase &t)
        {
            CodeGenerator gen({}, {});

            // fd = f2, fs = f3, ft = f4 (all distinct so a wrong operand shows up)
            const std::string div = gen.translateInstruction(makeCop1S(COP1_S_DIV, 2, 3, 4));
            t.IsTrue(div.find("ctx->f[2] = FPU_DIV_S(ctx->f[3], ctx->f[4])") != std::string::npos,
                     "DIV.S should write fs / ft to fd but got: " + div);
            t.IsTrue(div.find("INFINITY") == std::string::npos, "DIV.S must not produce IEEE infinity but got: " + div);

            const std::string rsqrt = gen.translateInstruction(makeCop1S(COP1_S_RSQRT, 2, 3, 4));
            t.IsTrue(rsqrt.find("ctx->f[2] = FPU_RSQRT_S(ctx->f[3], ctx->f[4])") != std::string::npos,
                     "RSQRT.S should compute fs / sqrt(ft) but got: " + rsqrt);
        });
    });
}
