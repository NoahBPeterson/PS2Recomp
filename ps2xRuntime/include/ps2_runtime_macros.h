#ifndef PS2_RUNTIME_MACROS_H
#define PS2_RUNTIME_MACROS_H
#include <cstdint>
#include <cmath>
#include <cstring>
#include <bit>
#include <limits>
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(USE_SSE2NEON)
#include "sse2neon.h"
#else
#include <immintrin.h> // For SSE/AVX intrinsics
#endif

#include "ps2_recomp_abi.h"

static inline int32_t Ps2ExtractEpi32(__m128i v, int index)
{
    switch (index & 3)
    {
    case 0:
        return _mm_extract_epi32(v, 0);
    case 1:
        return _mm_extract_epi32(v, 1);
    case 2:
        return _mm_extract_epi32(v, 2);
    default:
        return _mm_extract_epi32(v, 3);
    }
}

static inline int64_t Ps2ExtractEpi64(__m128i v, int index)
{
    if ((index & 1) == 0)
    {
        return _mm_cvtsi128_si64(v);
    }
    else
    {
        return _mm_extract_epi64(v, 1);
    }
}

static inline uint32_t ps2_clz32(uint32_t x)
{
    return static_cast<uint32_t>(std::countl_zero(x));
}

static inline uint64_t Ps2HiLoToU64(uint64_t hi, uint64_t lo)
{
    return ((hi & 0xFFFFFFFFull) << 32) | (lo & 0xFFFFFFFFull);
}

static inline uint64_t Ps2SignExt32ToU64(uint32_t v)
{
    return (uint64_t)(int64_t)(int32_t)v;
}

// PLZCW: Count leading bits that match the sign bit, minus 1.
// For positive values: count leading zeros minus 1 (excludes sign bit).
// For negative values: count leading ones minus 1 (excludes sign bit).
// Special cases: 0x00000000 -> 31, 0xFFFFFFFF -> 31.
static inline uint32_t ps2_plzcw32(uint32_t x)
{
    if (x == 0 || x == 0xFFFFFFFF)
        return 31;
    if (x & 0x80000000u)
        x = ~x; // If sign bit set, invert to count leading ones as zeros
    return static_cast<uint32_t>(std::countl_zero(x)) - 1;
}

#define PS2_BLENDV_PS(a, b, mask) _mm_blendv_ps((a), (b), (mask))
#define PS2_MIN_EPI32(a, b) _mm_min_epi32((a), (b))
#define PS2_MAX_EPI32(a, b) _mm_max_epi32((a), (b))
#define PS2_SHUFFLE_EPI8(v, mask) _mm_shuffle_epi8((v), (mask))

#define PS2_EXTRACT_EPI32(v, i) Ps2ExtractEpi32((v), (i))
#define PS2_EXTRACT_EPI64(v, i) Ps2ExtractEpi64((v), (i))

#define PS2_EXTRACT_EPI32_0(v) Ps2ExtractEpi32((v), 0)
#define PS2_EXTRACT_EPI32_1(v) Ps2ExtractEpi32((v), 1)
#define PS2_EXTRACT_EPI32_2(v) Ps2ExtractEpi32((v), 2)
#define PS2_EXTRACT_EPI32_3(v) Ps2ExtractEpi32((v), 3)

#define PS2_EXTRACT_EPI64_0(v) Ps2ExtractEpi64((v), 0)
#define PS2_EXTRACT_EPI64_1(v) Ps2ExtractEpi64((v), 1)

// Basic MIPS arithmetic operations
#define ADD32(a, b) ((uint32_t)((a) + (b)))
#define ADD32_OV(rs, rt, result32, overflow)              \
    do                                                    \
    {                                                     \
        int32_t _a = (int32_t)(rs);                       \
        int32_t _b = (int32_t)(rt);                       \
        int32_t _r = (int32_t)((uint32_t)_a + (uint32_t)_b); \
        overflow = (((_a ^ _b) >= 0) && ((_a ^ _r) < 0)); \
        result32 = (uint32_t)_r;                          \
    } while (0);
#define SUB32(a, b) ((uint32_t)((a) - (b)))
#define SUB32_OV(rs, rt, result32, overflow)             \
    do                                                   \
    {                                                    \
        int32_t _a = (int32_t)(rs);                      \
        int32_t _b = (int32_t)(rt);                      \
        int32_t _r = (int32_t)((uint32_t)_a - (uint32_t)_b); \
        overflow = (((_a ^ _b) < 0) && ((_a ^ _r) < 0)); \
        result32 = (uint32_t)_r;                         \
    } while (0);
#define MUL32(a, b) ((uint32_t)((a) * (b)))
#define DIV32(a, b) ((uint32_t)((a) / (b)))
#define AND32(a, b) ((uint32_t)((a) & (b)))
#define OR32(a, b) ((uint32_t)((a) | (b)))
#define XOR32(a, b) ((uint32_t)((a) ^ (b)))
#define NOR32(a, b) ((uint32_t)(~((a) | (b))))
#define SLL32(a, b) ((uint32_t)((a) << (b)))
#define SRL32(a, b) ((uint32_t)((a) >> (b)))
#define SRA32(a, b) ((uint32_t)((int32_t)(a) >> (b)))
#define SLT32(a, b) ((uint32_t)((int32_t)(a) < (int32_t)(b) ? 1 : 0))
#define SLTU32(a, b) ((uint32_t)((a) < (b) ? 1 : 0))

// PS2-specific 128-bit MMI operations
#define PS2_PEXTLW(a, b) _mm_unpacklo_epi32((__m128i)(b), (__m128i)(a))
#define PS2_PEXTUW(a, b) _mm_unpackhi_epi32((__m128i)(b), (__m128i)(a))
#define PS2_PEXTLH(a, b) _mm_unpacklo_epi16((__m128i)(b), (__m128i)(a))
#define PS2_PEXTUH(a, b) _mm_unpackhi_epi16((__m128i)(b), (__m128i)(a))
#define PS2_PEXTLB(a, b) _mm_unpacklo_epi8((__m128i)(b), (__m128i)(a))
#define PS2_PEXTUB(a, b) _mm_unpackhi_epi8((__m128i)(b), (__m128i)(a))
#define PS2_PADDW(a, b) _mm_add_epi32((__m128i)(a), (__m128i)(b))
#define PS2_PSUBW(a, b) _mm_sub_epi32((__m128i)(a), (__m128i)(b))
#define PS2_PMAXW(a, b) PS2_MAX_EPI32((__m128i)(a), (__m128i)(b))
#define PS2_PMINW(a, b) PS2_MIN_EPI32((__m128i)(a), (__m128i)(b))
#define PS2_PADDH(a, b) _mm_add_epi16((__m128i)(a), (__m128i)(b))
#define PS2_PSUBH(a, b) _mm_sub_epi16((__m128i)(a), (__m128i)(b))
#define PS2_PMAXH(a, b) _mm_max_epi16((__m128i)(a), (__m128i)(b))
#define PS2_PMINH(a, b) _mm_min_epi16((__m128i)(a), (__m128i)(b))
#define PS2_PADDB(a, b) _mm_add_epi8((__m128i)(a), (__m128i)(b))
#define PS2_PSUBB(a, b) _mm_sub_epi8((__m128i)(a), (__m128i)(b))
#define PS2_PAND(a, b) _mm_and_si128((__m128i)(a), (__m128i)(b))
#define PS2_POR(a, b) _mm_or_si128((__m128i)(a), (__m128i)(b))
#define PS2_PXOR(a, b) _mm_xor_si128((__m128i)(a), (__m128i)(b))
#define PS2_PNOR(a, b) _mm_xor_si128(_mm_or_si128((__m128i)(a), (__m128i)(b)), _mm_set1_epi32(0xFFFFFFFF))

// PS2 float semantics (EE FPU and VU0 macro mode). The PS2 format has no Inf, NaN or denormals:
//   - exponent 255 is an ordinary exponent, so 0x7F800000 is 2^128 and the largest value is 0x7FFFFFFF;
//   - exponent 0 is zero whatever the mantissa holds;
//   - a result of 2^129 or more saturates to +-0x7FFFFFFF, one below 2^-126 becomes +-0;
//   - x/0 gives +-0x7FFFFFFF.
// A host float cannot hold the exponent-255 values, so anything that touches them is computed in double.
// That path is rare and kept out of line; the inline part is the host operation and a range check.
// Results round toward zero except the FPU's DIV.S and SQRT.S (see ps2_fadd and ps2_vadd).
// Checked against the hardware results in ps2autotests (tests/cpu/ee_fpu).
#if defined(_MSC_VER)
#define PS2_FLOAT_NOINLINE __declspec(noinline)
#else
#define PS2_FLOAT_NOINLINE __attribute__((noinline))
#endif

constexpr uint32_t PS2_FLOAT_MAX_BITS = 0x7FFFFFFFu;
constexpr uint32_t PS2_FLOAT_EXP_MASK = 0x7F800000u;
constexpr uint32_t PS2_FLOAT_SIGN_MASK = 0x80000000u;

// True when a host float operation cannot be trusted with x: exponent 255, or exponent 0 with a mantissa.
inline bool ps2_fspecial(float x)
{
    const uint32_t t = std::bit_cast<uint32_t>(x) & 0x7FFFFFFFu;
    return t != 0u && (t - 0x00800000u) >= 0x7F000000u;
}

inline double ps2_to_double(float x)
{
    uint32_t b = std::bit_cast<uint32_t>(x);
    const uint32_t exponent = b & PS2_FLOAT_EXP_MASK;
    if (exponent == 0u)
        return static_cast<double>(std::bit_cast<float>(b & PS2_FLOAT_SIGN_MASK));
    if (exponent == PS2_FLOAT_EXP_MASK)
        return static_cast<double>(std::bit_cast<float>(b - 0x00800000u)) * 2.0;
    return static_cast<double>(x);
}

inline float ps2_from_double(double d)
{
    const double magnitude = std::fabs(d);
    const uint32_t sign = std::signbit(d) ? PS2_FLOAT_SIGN_MASK : 0u;
    if (magnitude >= 0x1p129)
        return std::bit_cast<float>(sign | PS2_FLOAT_MAX_BITS);
    if (magnitude < 0x1p-126)
        return std::bit_cast<float>(sign);
    if (magnitude >= 0x1p128)
    {
        // Convert one exponent lower, then put the exponent back as 255.
        const uint32_t b = std::bit_cast<uint32_t>(static_cast<float>(d * 0.5));
        if ((b & PS2_FLOAT_EXP_MASK) == PS2_FLOAT_EXP_MASK)
            return std::bit_cast<float>(sign | PS2_FLOAT_MAX_BITS);
        return std::bit_cast<float>(b + 0x00800000u);
    }
    // A value that rounds up to 2^128 comes out as the host's Inf pattern, which is 2^128 on the PS2.
    return static_cast<float>(d);
}

// A zero divisor gives +-max with the sign of fs xor ft.
PS2_FLOAT_NOINLINE inline float ps2_fdiv_slow(float a, float b)
{
    const uint32_t ua = std::bit_cast<uint32_t>(a);
    const uint32_t ub = std::bit_cast<uint32_t>(b);
    if ((ub & PS2_FLOAT_EXP_MASK) == 0u)
        return std::bit_cast<float>(((ua ^ ub) & PS2_FLOAT_SIGN_MASK) | PS2_FLOAT_MAX_BITS);
    const float r = a / b;
    if (!(ps2_fspecial(a) | ps2_fspecial(b) | ps2_fspecial(r)))
        return r;
    return ps2_from_double(ps2_to_double(a) / ps2_to_double(b));
}

// sqrt(|ft|); the result is never negative, sqrt(-0) is +0.
PS2_FLOAT_NOINLINE inline float ps2_fsqrt_slow(float a)
{
    if ((std::bit_cast<uint32_t>(a) & PS2_FLOAT_EXP_MASK) == 0u)
        return 0.0f;
    return ps2_from_double(std::sqrt(std::fabs(ps2_to_double(a))));
}

// True when a, b and the host result r are all normal numbers, so r is also the PS2 result.
inline bool ps2_fall_normal(float a, float b, float r)
{
    // Doubling drops the sign; adding one exponent step wraps exponent 255 round to 0 next to exponent 0.
    const uint32_t sa = std::bit_cast<uint32_t>(a) * 2u + 0x01000000u;
    const uint32_t sb = std::bit_cast<uint32_t>(b) * 2u + 0x01000000u;
    const uint32_t sr = std::bit_cast<uint32_t>(r) * 2u + 0x01000000u;
    uint32_t lowest = sa < sb ? sa : sb;
    lowest = lowest < sr ? lowest : sr;
    return lowest >= 0x02000000u;
}

// EE FPU rounding: ADD.S, SUB.S and MUL.S (and the MADD/MSUB/ADDA/MULA forms built from them) round toward
// zero, as the hardware does (ps2autotests tests/cpu/ee_fpu) and as PCSX2 and ps2-float model it; a host
// that rounds to nearest gets the last bit wrong in about half of all inexact results. Before an addition
// the operand with the smaller exponent keeps one guard bit: with an exponent difference d its low d - 1
// mantissa bits are dropped, and at d >= 25 only its sign is left. Operands with exponent 0 read as zero.
// The aligned sum and a 24x24-bit product are exact in double, so the only rounding is the chop.
// DIV.S and SQRT.S keep rounding to nearest (their own unit).

// The PS2 value of an exact real d, chopped toward zero, saturated and flushed like every result.
PS2_FLOAT_NOINLINE inline float ps2_chop_from_double(double d)
{
    const double magnitude = std::fabs(d);
    const uint32_t sign = std::signbit(d) ? PS2_FLOAT_SIGN_MASK : 0u;
    if (magnitude >= 0x1p129)
        return std::bit_cast<float>(sign | PS2_FLOAT_MAX_BITS);
    if (magnitude < 0x1p-126)
        return std::bit_cast<float>(sign);
    // Exponent 255 is converted one exponent lower and put back.
    const bool top = magnitude >= 0x1p128;
    const double scaled = top ? magnitude * 0.5 : magnitude;
    uint32_t b = std::bit_cast<uint32_t>(static_cast<float>(scaled));
    if (static_cast<double>(std::bit_cast<float>(b)) > scaled) // rounded up (or to Inf): one step back
        --b;
    if (top)
        b += 0x00800000u;
    return std::bit_cast<float>(sign | b);
}

// Chop an exact host result s; the inline part handles every normal result and exact zeros.
inline float ps2_fchop(double s)
{
    const double magnitude = std::fabs(s);
    if (magnitude >= 0x1p-126 && magnitude < 0x1p128) [[likely]]
    {
        uint32_t b = std::bit_cast<uint32_t>(static_cast<float>(s));
        if (std::fabs(static_cast<double>(std::bit_cast<float>(b))) > magnitude)
            --b;
        return std::bit_cast<float>(b);
    }
    if (magnitude == 0.0)
        return static_cast<float>(s); // keeps the sign of zero
    return ps2_chop_from_double(s);
}

// An operand with exponent 0 reads as zero (sign kept).
inline uint32_t ps2_foperand(uint32_t bits)
{
    return (bits & PS2_FLOAT_EXP_MASK) ? bits : (bits & PS2_FLOAT_SIGN_MASK);
}

// The smaller operand of an addition after alignment, d = exponent difference >= 1.
inline uint32_t ps2_falign(uint32_t bits, uint32_t d)
{
    return d >= 25u ? (bits & PS2_FLOAT_SIGN_MASK) : (bits & ~((1u << (d - 1u)) - 1u));
}

inline double ps2_fexact(uint32_t bits)
{
    return (bits & PS2_FLOAT_EXP_MASK) != PS2_FLOAT_EXP_MASK ? static_cast<double>(std::bit_cast<float>(bits))
                                                             : ps2_to_double(std::bit_cast<float>(bits));
}

inline float ps2_fadd_bits(uint32_t ua, uint32_t ub)
{
    ua = ps2_foperand(ua);
    ub = ps2_foperand(ub);
    const uint32_t ea = (ua >> 23) & 0xFFu, eb = (ub >> 23) & 0xFFu;
    if (ea > eb)
        ub = ps2_falign(ub, ea - eb);
    else if (eb > ea)
        ua = ps2_falign(ua, eb - ea);
    return ps2_fchop(ps2_fexact(ua) + ps2_fexact(ub));
}

inline float ps2_fadd(float a, float b)
{
    return ps2_fadd_bits(std::bit_cast<uint32_t>(a), std::bit_cast<uint32_t>(b));
}

inline float ps2_fsub(float a, float b)
{
    return ps2_fadd_bits(std::bit_cast<uint32_t>(a), std::bit_cast<uint32_t>(b) ^ PS2_FLOAT_SIGN_MASK);
}

inline float ps2_fmul(float a, float b)
{
    const uint32_t ua = ps2_foperand(std::bit_cast<uint32_t>(a));
    const uint32_t ub = ps2_foperand(std::bit_cast<uint32_t>(b));
    return ps2_fchop(ps2_fexact(ua) * ps2_fexact(ub));
}

// CVT.S.W: chopped like every other FPU result (integers above 2^24 lose their low bits toward zero).
inline float ps2_fcvt_s_w(int32_t v)
{
    uint32_t b = std::bit_cast<uint32_t>(static_cast<float>(v));
    if (std::fabs(static_cast<double>(std::bit_cast<float>(b))) > std::fabs(static_cast<double>(v)))
        --b;
    return std::bit_cast<float>(b);
}

inline float ps2_fdiv(float a, float b)
{
    const float r = a / b;
    return ps2_fall_normal(a, b, r) ? r : ps2_fdiv_slow(a, b);
}

inline float ps2_fsqrt(float a)
{
    const float r = std::sqrt(a);
    return ps2_fall_normal(a, a, r) ? r : ps2_fsqrt_slow(a);
}

// RSQRT.S: fs / sqrt(|ft|), the root and the divide both chopped (the instruction keeps the adder's rounding
// mode, unlike SQRT.S). A zero ft gives +-max with the sign of fs alone.
PS2_FLOAT_NOINLINE inline float ps2_frsqrt(float a, float b)
{
    const uint32_t ub = std::bit_cast<uint32_t>(b) & PS2_FLOAT_MAX_BITS;
    const uint32_t ua = ps2_foperand(std::bit_cast<uint32_t>(a));
    if ((ub & PS2_FLOAT_EXP_MASK) == 0u)
        return std::bit_cast<float>((ua & PS2_FLOAT_SIGN_MASK) | PS2_FLOAT_MAX_BITS);
    const double bd = ps2_fexact(ub);
    // The chopped root: a candidate from the double root, settled with exact squares (24-bit significands).
    uint32_t root = std::bit_cast<uint32_t>(ps2_chop_from_double(std::sqrt(bd)));
    if (ps2_fexact(root) * ps2_fexact(root) > bd)
        --root;
    else if (ps2_fexact(root + 1u) * ps2_fexact(root + 1u) <= bd)
        ++root;
    const double rd = ps2_fexact(root);
    const double ad = std::fabs(ps2_fexact(ua));
    const uint32_t sign = ua & PS2_FLOAT_SIGN_MASK;
    uint32_t q = std::bit_cast<uint32_t>(ps2_chop_from_double(ad / rd));
    if (q != 0u && q != PS2_FLOAT_MAX_BITS)
    {
        if (ps2_fexact(q) * rd > ad)
            --q;
        else if (ps2_fexact(q + 1u) * rd <= ad)
            ++q;
    }
    return std::bit_cast<float>(sign | q);
}

// One step toward zero from a chopped candidate that came out one too large; exponent 0 flushes to +-0.
inline float ps2_fstep_down(uint32_t bits)
{
    --bits;
    return std::bit_cast<float>((bits & PS2_FLOAT_EXP_MASK) ? bits : (bits & PS2_FLOAT_SIGN_MASK));
}

enum class Ps2VecOp
{
    Add,
    Sub,
    Mul
};

// VU0 macro-mode ADD/SUB/MUL (and the MADD/MSUB forms built from them) round toward zero like the VU1 core
// (which runs in FE_TOWARDZERO), without the FPU adder's alignment step.

// One VU lane through doubles, chopped; for lanes whose host result has exponent 255.
PS2_FLOAT_NOINLINE inline __m128 ps2_vslow(__m128 a, __m128 b, Ps2VecOp op)
{
    alignas(16) float la[4], lb[4], lr[4];
    _mm_store_ps(la, a);
    _mm_store_ps(lb, b);
    for (int i = 0; i < 4; ++i)
    {
        const double x = ps2_to_double(la[i]);
        double y = ps2_to_double(lb[i]);
        if (op == Ps2VecOp::Mul)
        {
            lr[i] = ps2_chop_from_double(x * y); // exact: two 24-bit significands
            continue;
        }
        if (op == Ps2VecOp::Sub)
            y = -y;
        // The double sum can round a far smaller addend away; its exact error (TwoSum) says whether the
        // true sum lies just inside a value the double sum hit exactly.
        const double s = x + y;
        const double bb = s - x;
        const double e = (x - (s - bb)) + (y - bb);
        uint32_t r = std::bit_cast<uint32_t>(ps2_chop_from_double(s));
        if (e != 0.0 && std::signbit(e) != std::signbit(s) && ps2_fexact(r) == s)
            r = std::bit_cast<uint32_t>(ps2_fstep_down(r));
        lr[i] = std::bit_cast<float>(r);
    }
    return _mm_load_ps(lr);
}

// r is the host result (rounded to nearest) of a op b and away the lanes where that rounded away from zero
// (all ones there). Exponent-255 lanes go through the double path; the rest step one value toward zero
// where needed, and a result with exponent 0 is flushed to +-0.
inline __m128 ps2_vfinish(__m128 a, __m128 b, __m128 r, __m128i away, Ps2VecOp op)
{
    const __m128i expMask = _mm_set1_epi32(0x7F800000);
    if (_mm_movemask_epi8(_mm_cmpeq_epi32(_mm_and_si128(_mm_castps_si128(r), expMask), expMask)) != 0)
        return ps2_vslow(a, b, op);
    const __m128i chopped = _mm_add_epi32(_mm_castps_si128(r), away);
    const __m128i rexp = _mm_and_si128(chopped, expMask);
    const __m128i flush = _mm_and_si128(_mm_cmpeq_epi32(rexp, _mm_setzero_si128()), _mm_set1_epi32(0x7FFFFFFF));
    return _mm_castsi128_ps(_mm_andnot_si128(flush, chopped));
}

// The lanes of a sum r = a + b that rounded away from zero: the exact error (TwoSum) is non-zero with the
// opposite sign of r.
inline __m128i ps2_vadd_away(__m128 a, __m128 b, __m128 r)
{
    const __m128 bb = _mm_sub_ps(r, a);
    const __m128 e = _mm_add_ps(_mm_sub_ps(a, _mm_sub_ps(r, bb)), _mm_sub_ps(b, bb));
    const __m128i opposite = _mm_srai_epi32(_mm_xor_si128(_mm_castps_si128(e), _mm_castps_si128(r)), 31);
    return _mm_andnot_si128(_mm_castps_si128(_mm_cmpeq_ps(e, _mm_setzero_ps())), opposite);
}

// Operand lanes with exponent 0 read as +-0, as on the VU.
inline __m128 ps2_vdaz(__m128 x)
{
    const __m128i bits = _mm_castps_si128(x);
    const __m128i zeroExp = _mm_cmpeq_epi32(_mm_and_si128(bits, _mm_set1_epi32(0x7F800000)), _mm_setzero_si128());
    return _mm_castsi128_ps(_mm_andnot_si128(_mm_and_si128(zeroExp, _mm_set1_epi32(0x7FFFFFFF)), bits));
}

inline __m128 ps2_vadd(__m128 a, __m128 b)
{
    a = ps2_vdaz(a);
    b = ps2_vdaz(b);
    const __m128 r = _mm_add_ps(a, b);
    return ps2_vfinish(a, b, r, ps2_vadd_away(a, b, r), Ps2VecOp::Add);
}

inline __m128 ps2_vsub(__m128 a, __m128 b)
{
    a = ps2_vdaz(a);
    b = ps2_vdaz(b);
    const __m128 nb = _mm_xor_ps(b, _mm_castsi128_ps(_mm_set1_epi32(static_cast<int>(0x80000000u))));
    const __m128 r = _mm_add_ps(a, nb);
    return ps2_vfinish(a, b, r, ps2_vadd_away(a, nb, r), Ps2VecOp::Sub);
}

// A 24x24-bit product is exact in double: the lanes where |r| > |a * b| rounded away from zero.
inline __m128 ps2_vmul(__m128 a, __m128 b)
{
    a = ps2_vdaz(a);
    b = ps2_vdaz(b);
    const __m128 r = _mm_mul_ps(a, b);
    const __m128d absMask = _mm_castsi128_pd(_mm_set1_epi64x(0x7FFFFFFFFFFFFFFFll));
    const __m128d plo = _mm_and_pd(_mm_mul_pd(_mm_cvtps_pd(a), _mm_cvtps_pd(b)), absMask);
    const __m128d phi = _mm_and_pd(_mm_mul_pd(_mm_cvtps_pd(_mm_movehl_ps(a, a)), _mm_cvtps_pd(_mm_movehl_ps(b, b))), absMask);
    const __m128d rlo = _mm_and_pd(_mm_cvtps_pd(r), absMask);
    const __m128d rhi = _mm_and_pd(_mm_cvtps_pd(_mm_movehl_ps(r, r)), absMask);
    const __m128 away = _mm_shuffle_ps(_mm_castpd_ps(_mm_cmpgt_pd(rlo, plo)), _mm_castpd_ps(_mm_cmpgt_pd(rhi, phi)),
                                       _MM_SHUFFLE(2, 0, 2, 0));
    return ps2_vfinish(a, b, r, _mm_castps_si128(away), Ps2VecOp::Mul);
}

// PS2 VU (Vector Unit) operations
#define PS2_VADD(a, b) ps2_vadd((__m128)(a), (__m128)(b))
#define PS2_VSUB(a, b) ps2_vsub((__m128)(a), (__m128)(b))
#define PS2_VMUL(a, b) ps2_vmul((__m128)(a), (__m128)(b))
#define PS2_VDIV(a, b) _mm_div_ps((__m128)(a), (__m128)(b))
#define PS2_VMULQ(a, q) _mm_mul_ps((__m128)(a), _mm_set1_ps(q))
#define PS2_VBLEND(a, b, mask) PS2_BLENDV_PS((__m128)(a), (__m128)(b), (__m128)(mask))

// Memory access helpers - Hybrid Fast/Slow Path
// Fast path: Direct RDRAM access (masked).
// Slow path: Full runtime->Load/Store

static inline bool Ps2FastRangeIsContiguous(uint32_t offset, uint32_t bytes)
{
    return offset <= (PS2_RAM_SIZE - bytes);
}

static inline uint8_t Ps2FastRead8(const uint8_t *rdram, uint32_t addr)
{
    return rdram[addr & PS2_RAM_MASK];
}

static inline uint16_t Ps2FastRead16(const uint8_t *rdram, uint32_t addr)
{
    const uint32_t offset = addr & PS2_RAM_MASK;
    if (!Ps2FastRangeIsContiguous(offset, sizeof(uint16_t)))
    {
        uint8_t wrapped[sizeof(uint16_t)];
        for (uint32_t i = 0; i < sizeof(uint16_t); ++i)
        {
            wrapped[i] = rdram[(offset + i) & PS2_RAM_MASK];
        }
        uint16_t value;
        std::memcpy(&value, wrapped, sizeof(value));
        return value;
    }

    uint16_t value;
    std::memcpy(&value, rdram + offset, sizeof(value));
    return value;
}

static inline uint32_t Ps2FastRead32(const uint8_t *rdram, uint32_t addr)
{
    const uint32_t offset = addr & PS2_RAM_MASK;
    if (!Ps2FastRangeIsContiguous(offset, sizeof(uint32_t)))
    {
        uint8_t wrapped[sizeof(uint32_t)];
        for (uint32_t i = 0; i < sizeof(uint32_t); ++i)
        {
            wrapped[i] = rdram[(offset + i) & PS2_RAM_MASK];
        }
        uint32_t value;
        std::memcpy(&value, wrapped, sizeof(value));
        return value;
    }

    uint32_t value;
    std::memcpy(&value, rdram + offset, sizeof(value));
    return value;
}

static inline uint64_t Ps2FastRead64(const uint8_t *rdram, uint32_t addr)
{
    const uint32_t offset = addr & PS2_RAM_MASK;
    if (!Ps2FastRangeIsContiguous(offset, sizeof(uint64_t)))
    {
        uint8_t wrapped[sizeof(uint64_t)];
        for (uint32_t i = 0; i < sizeof(uint64_t); ++i)
        {
            wrapped[i] = rdram[(offset + i) & PS2_RAM_MASK];
        }
        uint64_t value;
        std::memcpy(&value, wrapped, sizeof(value));
        return value;
    }

    uint64_t value;
    std::memcpy(&value, rdram + offset, sizeof(value));
    return value;
}

static inline __m128i Ps2FastRead128(const uint8_t *rdram, uint32_t addr)
{
    const uint32_t offset = addr & PS2_RAM_MASK;
    if (!Ps2FastRangeIsContiguous(offset, sizeof(__m128i)))
    {
        alignas(16) uint8_t wrapped[sizeof(__m128i)];
        for (uint32_t i = 0; i < sizeof(__m128i); ++i)
        {
            wrapped[i] = rdram[(offset + i) & PS2_RAM_MASK];
        }
        __m128i value;
        std::memcpy(&value, wrapped, sizeof(value));
        return value;
    }

    __m128i value;
    std::memcpy(&value, rdram + offset, sizeof(value));
    return value;
}

static inline void Ps2FastWrite8(uint8_t *rdram, uint32_t addr, uint8_t value)
{
    rdram[addr & PS2_RAM_MASK] = value;
}

static inline void Ps2FastWrite16(uint8_t *rdram, uint32_t addr, uint16_t value)
{
    const uint32_t offset = addr & PS2_RAM_MASK;
    if (!Ps2FastRangeIsContiguous(offset, sizeof(uint16_t)))
    {
        uint8_t wrapped[sizeof(uint16_t)];
        std::memcpy(wrapped, &value, sizeof(value));
        for (uint32_t i = 0; i < sizeof(uint16_t); ++i)
        {
            rdram[(offset + i) & PS2_RAM_MASK] = wrapped[i];
        }
        return;
    }
    std::memcpy(rdram + offset, &value, sizeof(value));
}

static inline void Ps2FastWrite32(uint8_t *rdram, uint32_t addr, uint32_t value)
{
    const uint32_t offset = addr & PS2_RAM_MASK;
    if (!Ps2FastRangeIsContiguous(offset, sizeof(uint32_t)))
    {
        uint8_t wrapped[sizeof(uint32_t)];
        std::memcpy(wrapped, &value, sizeof(value));
        for (uint32_t i = 0; i < sizeof(uint32_t); ++i)
        {
            rdram[(offset + i) & PS2_RAM_MASK] = wrapped[i];
        }
        return;
    }
    std::memcpy(rdram + offset, &value, sizeof(value));
}

static inline void Ps2FastWrite64(uint8_t *rdram, uint32_t addr, uint64_t value)
{
    const uint32_t offset = addr & PS2_RAM_MASK;
    if (!Ps2FastRangeIsContiguous(offset, sizeof(uint64_t)))
    {
        uint8_t wrapped[sizeof(uint64_t)];
        std::memcpy(wrapped, &value, sizeof(value));
        for (uint32_t i = 0; i < sizeof(uint64_t); ++i)
        {
            rdram[(offset + i) & PS2_RAM_MASK] = wrapped[i];
        }
        return;
    }
    std::memcpy(rdram + offset, &value, sizeof(value));
}

static inline void Ps2FastWrite128(uint8_t *rdram, uint32_t addr, __m128i value)
{
    const uint32_t offset = addr & PS2_RAM_MASK;
    if (!Ps2FastRangeIsContiguous(offset, sizeof(__m128i)))
    {
        alignas(16) uint8_t wrapped[sizeof(__m128i)];
        std::memcpy(wrapped, &value, sizeof(value));
        for (uint32_t i = 0; i < sizeof(__m128i); ++i)
        {
            rdram[(offset + i) & PS2_RAM_MASK] = wrapped[i];
        }
        return;
    }
    std::memcpy(rdram + offset, &value, sizeof(value));
}

#define FAST_READ8(addr) Ps2FastRead8(rdram, (uint32_t)(addr))
#define FAST_READ16(addr) Ps2FastRead16(rdram, (uint32_t)(addr))
#define FAST_READ32(addr) Ps2FastRead32(rdram, (uint32_t)(addr))
#define FAST_READ64(addr) Ps2FastRead64(rdram, (uint32_t)(addr))
#define FAST_READ128(addr) Ps2FastRead128(rdram, (uint32_t)(addr))

#define FAST_WRITE8(addr, val) Ps2FastWrite8(rdram, (uint32_t)(addr), (uint8_t)(val))
#define FAST_WRITE16(addr, val) Ps2FastWrite16(rdram, (uint32_t)(addr), (uint16_t)(val))
#define FAST_WRITE32(addr, val) Ps2FastWrite32(rdram, (uint32_t)(addr), (uint32_t)(val))
#define FAST_WRITE64(addr, val) Ps2FastWrite64(rdram, (uint32_t)(addr), (uint64_t)(val))
#define FAST_WRITE128(addr, val) Ps2FastWrite128(rdram, (uint32_t)(addr), (val))

#define READ8(addr) ([&]() -> uint8_t {                       \
    uint32_t _addr = (uint32_t)(addr);                        \
    return Ps2IsSpecialAddress(_addr)                \
        ? ps2x_abi::Load8(runtime, rdram, ctx, _addr)                   \
        : FAST_READ8(_addr); }())

#define READ16(addr) ([&]() -> uint16_t {                     \
    uint32_t _addr = (uint32_t)(addr);                        \
    return Ps2IsSpecialAddress(_addr)                \
        ? ps2x_abi::Load16(runtime, rdram, ctx, _addr)                  \
        : FAST_READ16(_addr); }())

#define READ32(addr) ([&]() -> uint32_t {                     \
    uint32_t _addr = (uint32_t)(addr);                        \
    return Ps2IsSpecialAddress(_addr)                \
        ? ps2x_abi::Load32(runtime, rdram, ctx, _addr)                  \
        : FAST_READ32(_addr); }())

#define READ64(addr) ([&]() -> uint64_t {                     \
    uint32_t _addr = (uint32_t)(addr);                        \
    return Ps2IsSpecialAddress(_addr)                \
        ? ps2x_abi::Load64(runtime, rdram, ctx, _addr)                  \
        : FAST_READ64(_addr); }())

#define READ128(addr) ([&]() -> __m128i {                     \
    uint32_t _addr = (uint32_t)(addr);                        \
    return Ps2IsSpecialAddress(_addr)                \
        ? ps2x_abi::Load128(runtime, rdram, ctx, _addr)                 \
        : FAST_READ128(_addr); }())

#define WRITE8(addr, val)                                                            \
    do                                                                               \
    {                                                                                \
        uint32_t _addr = (addr);                                                     \
        if (Ps2IsSpecialAddress(_addr))                                     \
            ps2x_abi::Store8(runtime, rdram, ctx, _addr, (val));                               \
        else                                                                         \
        {                                                                            \
            ps2TraceGuestWrite(rdram, _addr, 1u, (uint8_t)(val), 0u, "WRITE8", ctx); \
            FAST_WRITE8(_addr, (val));                                               \
        }                                                                            \
    } while (0)

#define WRITE16(addr, val)                                                             \
    do                                                                                 \
    {                                                                                  \
        uint32_t _addr = (addr);                                                       \
        if (Ps2IsSpecialAddress(_addr))                                       \
            ps2x_abi::Store16(runtime, rdram, ctx, _addr, (val));                                \
        else                                                                           \
        {                                                                              \
            ps2TraceGuestWrite(rdram, _addr, 2u, (uint16_t)(val), 0u, "WRITE16", ctx); \
            FAST_WRITE16(_addr, (val));                                                \
        }                                                                              \
    } while (0)

#define WRITE32(addr, val)                                                             \
    do                                                                                 \
    {                                                                                  \
        uint32_t _addr = (addr);                                                       \
        if (Ps2IsSpecialAddress(_addr))                                       \
            ps2x_abi::Store32(runtime, rdram, ctx, _addr, (val));                                \
        else                                                                           \
        {                                                                              \
            ps2TraceGuestWrite(rdram, _addr, 4u, (uint32_t)(val), 0u, "WRITE32", ctx); \
            FAST_WRITE32(_addr, (val));                                                \
        }                                                                              \
    } while (0)

#define WRITE64(addr, val)                                                             \
    do                                                                                 \
    {                                                                                  \
        uint32_t _addr = (addr);                                                       \
        if (Ps2IsSpecialAddress(_addr))                                       \
            ps2x_abi::Store64(runtime, rdram, ctx, _addr, (val));                                \
        else                                                                           \
        {                                                                              \
            ps2TraceGuestWrite(rdram, _addr, 8u, (uint64_t)(val), 0u, "WRITE64", ctx); \
            FAST_WRITE64(_addr, (val));                                                \
        }                                                                              \
    } while (0)

#define WRITE128(addr, val)                                                          \
    do                                                                               \
    {                                                                                \
        uint32_t _addr = (addr);                                                     \
        __m128i _value = (val);                                                      \
        if (Ps2IsSpecialAddress(_addr))                                     \
            ps2x_abi::Store128(runtime, rdram, ctx, _addr, _value);                            \
        else                                                                         \
        {                                                                            \
            const uint64_t _lo = static_cast<uint64_t>(PS2_EXTRACT_EPI64_0(_value)); \
            const uint64_t _hi = static_cast<uint64_t>(PS2_EXTRACT_EPI64_1(_value)); \
            ps2TraceGuestWrite(rdram, _addr, 16u, _lo, _hi, "WRITE128", ctx);        \
            FAST_WRITE128(_addr, _value);                                            \
        }                                                                            \
    } while (0)

// Packed Compare Greater Than (PCGT)
#define PS2_PCGTW(a, b) _mm_cmpgt_epi32((__m128i)(a), (__m128i)(b))
#define PS2_PCGTH(a, b) _mm_cmpgt_epi16((__m128i)(a), (__m128i)(b))
#define PS2_PCGTB(a, b) _mm_cmpgt_epi8((__m128i)(a), (__m128i)(b))

// Packed Add with Signed Saturation Word (PADDSW)
inline __m128i ps2_paddsw(__m128i a, __m128i b)
{

    __m128i sum = _mm_add_epi32(a, b);
    // Check for over/underflow. Clamp to either INT32_MIN/INT32_MAX.
    __m128i overflow = _mm_and_si128(_mm_xor_si128(a, sum),
                                     _mm_xor_si128(b, sum));
    // Extract input sign.
    overflow = _mm_srai_epi32(overflow, 31);
    __m128i input_sign = _mm_srai_epi32(a, 31);
    // Select saturation value based on overflow sign.
    #if defined(__SSE4_1__)
    __m128i sat = _mm_blendv_epi8(
        _mm_set1_epi32(INT32_MAX),
        _mm_set1_epi32(INT32_MIN),
        input_sign);
    return _mm_blendv_epi8(sum, sat, overflow);
    #else
    __m128i sat = _mm_or_si128(_mm_and_si128(input_sign, _mm_set1_epi32(INT32_MIN)),
                               _mm_andnot_si128(input_sign, _mm_set1_epi32(INT32_MAX)));
    return _mm_or_si128(_mm_and_si128(overflow, sat),
                        _mm_andnot_si128(overflow, sum));
    #endif
}
#define PS2_PADDSW(a, b) ps2_paddsw((__m128i)(a), (__m128i)(b))

// Packed Subtract with Signed Saturation Word (PSUBSW)
inline __m128i ps2_psubsw(__m128i a, __m128i b)
{
    __m128i diff = _mm_sub_epi32(a, b);
    // Check for over/underflow. Clamp to either INT32_MIN/INT32_MAX.
    __m128i overflow = _mm_and_si128(_mm_xor_si128(a, b),
                                     _mm_xor_si128(a, diff));
    // Extract input sign.
    overflow = _mm_srai_epi32(overflow, 31);
    __m128i input_sign = _mm_srai_epi32(a, 31);
    // Select saturation value based on overflow sign.
    #if defined(__SSE4_1__)
    __m128i sat = _mm_blendv_epi8(
        _mm_set1_epi32(INT32_MAX),
        _mm_set1_epi32(INT32_MIN),
        input_sign);
    return _mm_blendv_epi8(diff, sat, overflow);
    #else
    __m128i sat = _mm_or_si128(_mm_and_si128(input_sign, _mm_set1_epi32(INT32_MIN)),
                               _mm_andnot_si128(input_sign, _mm_set1_epi32(INT32_MAX)));
    return _mm_or_si128(_mm_and_si128(overflow, sat),
                        _mm_andnot_si128(overflow, diff));
    #endif

}
#define PS2_PSUBSW(a, b) ps2_psubsw((__m128i)(a), (__m128i)(b))

// Packed Compare Equal (PCEQ)
#define PS2_PCEQW(a, b) _mm_cmpeq_epi32((__m128i)(a), (__m128i)(b))
#define PS2_PCEQH(a, b) _mm_cmpeq_epi16((__m128i)(a), (__m128i)(b))
#define PS2_PCEQB(a, b) _mm_cmpeq_epi8((__m128i)(a), (__m128i)(b))

// Packed Absolute (PABS)
#define PS2_PABSW(a) _mm_abs_epi32((__m128i)(a))
#define PS2_PABSH(a) _mm_abs_epi16((__m128i)(a))
#define PS2_PABSB(a) _mm_abs_epi8((__m128i)(a))

// Packed Pack (PPAC) - Packs larger elements into smaller ones
inline __m128i ps2_paddu32(__m128i a, __m128i b)
{
    __m128i sum = _mm_add_epi32(a, b);
    __m128i overflow = _mm_cmpgt_epi32(_mm_xor_si128(a, _mm_set1_epi32(INT32_MIN)),
                                       _mm_xor_si128(sum, _mm_set1_epi32(INT32_MIN)));
    return _mm_or_si128(sum, overflow); // overflow lanes become all-1s
}
inline __m128i ps2_psubu32(__m128i a, __m128i b)
{
    __m128i diff = _mm_sub_epi32(a, b);
    // Underflow if a < b (unsigned). Clamp to 0.
    __m128i underflow = _mm_cmpgt_epi32(_mm_xor_si128(b, _mm_set1_epi32(INT32_MIN)),
                                        _mm_xor_si128(a, _mm_set1_epi32(INT32_MIN)));
    return _mm_andnot_si128(underflow, diff); // underflow lanes become 0
}

inline __m128i ps2_ppacw(__m128i rs, __m128i rt)
{
    // rs = [rs3 rs2 rs1 rs0], rt = [rt3 rt2 rt1 rt0]
    return _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(rt), _mm_castsi128_ps(rs), _MM_SHUFFLE(2, 0, 2, 0)));
}
#define PS2_PPACW(a, b) ps2_ppacw((__m128i)(a), (__m128i)(b))

inline __m128i ps2_ppach(__m128i rs, __m128i rt)
{
    const __m128i mask = _mm_setr_epi8(
        0, 1, 4, 5, 8, 9, 12, 13,  // from rt: halfwords 0,2,4,6
        0, 1, 4, 5, 8, 9, 12, 13); // from rs: halfwords 0,2,4,6
    __m128i lo = _mm_shuffle_epi8(rt, mask);
    __m128i hi = _mm_shuffle_epi8(rs, mask);
    return _mm_unpacklo_epi64(lo, hi);
}
#define PS2_PPACH(a, b) ps2_ppach((__m128i)(a), (__m128i)(b))

inline __m128i ps2_ppacb(__m128i rs, __m128i rt)
{
    const __m128i mask = _mm_setr_epi8(
        0, 2, 4, 6, 8, 10, 12, 14,  // from rt: bytes 0,2,4,6,8,10,12,14
        0, 2, 4, 6, 8, 10, 12, 14); // from rs
    __m128i lo = _mm_shuffle_epi8(rt, mask);
    __m128i hi = _mm_shuffle_epi8(rs, mask);
    return _mm_unpacklo_epi64(lo, hi);
}
#define PS2_PPACB(a, b) ps2_ppacb((__m128i)(a), (__m128i)(b))

// Packed Interleave (PINT)
// PINTH: rt's lower four halfwords interleaved with rs's upper four.
#define PS2_PINTH(rs, rt) _mm_unpacklo_epi16((__m128i)(rt), _mm_unpackhi_epi64((__m128i)(rs), (__m128i)(rs)))
// PINTEH: in each word, the even halfword of rt (low) and of rs (high).
#define PS2_PINTEH(rs, rt) _mm_or_si128(_mm_and_si128((__m128i)(rt), _mm_set1_epi32(0xFFFF)), _mm_slli_epi32((__m128i)(rs), 16))

// Packed Multiply-Add (PMADD)
#define PS2_PMADDW(a, b) _mm_add_epi32(_mm_mullo_epi32(_mm_shuffle_epi32((__m128i)(a), _MM_SHUFFLE(1, 0, 3, 2)), _mm_shuffle_epi32((__m128i)(b), _MM_SHUFFLE(1, 0, 3, 2))), _mm_mullo_epi32(_mm_shuffle_epi32((__m128i)(a), _MM_SHUFFLE(3, 2, 1, 0)), _mm_shuffle_epi32((__m128i)(b), _MM_SHUFFLE(3, 2, 1, 0))))

// Packed Variable Shifts
#define PS2_PSLLVW(a, b) _mm_custom_sllv_epi32((__m128i)(a), (__m128i)(b))
#define PS2_PSRLVW(a, b) _mm_custom_srlv_epi32((__m128i)(a), (__m128i)(b))
#define PS2_PSRAVW(a, b) _mm_custom_srav_epi32((__m128i)(a), (__m128i)(b))

inline __m128i _mm_custom_sllv_epi32(__m128i a, __m128i count)
{
    alignas(16) int32_t a_arr[4];
    alignas(16) int32_t count_arr[4];
    alignas(16) int32_t result[4];

    std::memcpy(a_arr, &a, sizeof(a));
    std::memcpy(count_arr, &count, sizeof(count));

    for (int i = 0; i < 4; i++)
    {
        result[i] = a_arr[i] << (count_arr[i] & 0x1F);
    }

    __m128i out;
    std::memcpy(&out, result, sizeof(out));
    return out;
}

inline __m128i _mm_custom_srlv_epi32(__m128i a, __m128i count)
{
    int32_t a_arr[4], count_arr[4], result[4];
    _mm_storeu_si128((__m128i *)a_arr, a);
    _mm_storeu_si128((__m128i *)count_arr, count);
    for (int i = 0; i < 4; i++)
    {
        result[i] = (uint32_t)a_arr[i] >> (count_arr[i] & 0x1F);
    }
    return _mm_loadu_si128((__m128i *)result);
}

inline __m128i _mm_custom_srav_epi32(__m128i a, __m128i count)
{
    int32_t a_arr[4], count_arr[4], result[4];
    _mm_storeu_si128((__m128i *)a_arr, a);
    _mm_storeu_si128((__m128i *)count_arr, count);
    for (int i = 0; i < 4; i++)
    {
        result[i] = a_arr[i] >> (count_arr[i] & 0x1F);
    }
    return _mm_loadu_si128((__m128i *)result);
}

// PMFHL function implementations
inline __m128i ps2_u64_to_epi64_pair(uint64_t value)
{
    return _mm_set1_epi64x(static_cast<long long>(value));
}

#define PS2_PMFHL_LW(hi, lo) _mm_unpacklo_epi64(ps2_u64_to_epi64_pair(lo), ps2_u64_to_epi64_pair(hi))
#define PS2_PMFHL_UW(hi, lo) _mm_unpackhi_epi64(ps2_u64_to_epi64_pair(lo), ps2_u64_to_epi64_pair(hi))
#define PS2_PMFHL_SLW(hi, lo) _mm_packs_epi32(ps2_u64_to_epi64_pair(lo), ps2_u64_to_epi64_pair(hi))
#define PS2_PMFHL_LH(hi, lo) _mm_shuffle_epi32(_mm_packs_epi32(ps2_u64_to_epi64_pair(lo), ps2_u64_to_epi64_pair(hi)), _MM_SHUFFLE(3, 1, 2, 0))
#define PS2_PMFHL_SH(hi, lo) _mm_shufflehi_epi16(_mm_shufflelo_epi16(_mm_packs_epi32(ps2_u64_to_epi64_pair(lo), ps2_u64_to_epi64_pair(hi)), _MM_SHUFFLE(3, 1, 2, 0)), _MM_SHUFFLE(3, 1, 2, 0))

// FPU (COP1) operations
#define FPU_SET_ACC(ctx, res) (ctx->f_acc = res)
#define FPU_ADD_S(a, b) ps2_fadd((float)(a), (float)(b))
#define FPU_SUB_S(a, b) ps2_fsub((float)(a), (float)(b))
#define FPU_MUL_S(a, b) ps2_fmul((float)(a), (float)(b))
#define FPU_DIV_S(a, b) ps2_fdiv((float)(a), (float)(b))
#define FPU_RSQRT_S(a, b) ps2_frsqrt((float)(a), (float)(b))
#define FPU_SQRT_S(a) ps2_fsqrt((float)(a))
#define FPU_ABS_S(a) fabsf((float)(a))
#define FPU_MOV_S(a) ((float)(a))
#define FPU_NEG_S(a) (-(float)(a))
#define FPU_ROUND_L_S(a) ((int64_t)roundf((float)(a)))
#define FPU_TRUNC_L_S(a) ((int64_t)(float)(a))
#define FPU_CEIL_L_S(a) ((int64_t)ceilf((float)(a)))
#define FPU_FLOOR_L_S(a) ((int64_t)floorf((float)(a)))
#define FPU_ROUND_W_S(a) ((int32_t)nearbyintf((float)(a)))
#define FPU_TRUNC_W_S(a) ((int32_t)(float)(a))
#define FPU_CEIL_W_S(a) ((int32_t)ceilf((float)(a)))
#define FPU_FLOOR_W_S(a) ((int32_t)floorf((float)(a)))
#define FPU_CVT_S_W(a) ps2_fcvt_s_w((int32_t)(a))
#define FPU_CVT_S_L(a) ((float)(int64_t)(a))
// CVT.W.S: truncate toward zero regardless of the rounding mode, saturating on
// overflow. Like PCSX2, any operand whose exponent field is >= 158 (|a| >= 2^31,
// including Inf/NaN) yields INT32_MIN when the sign bit is set, else INT32_MAX.
static inline int32_t ps2_fpu_cvt_w_s(float a)
{
    uint32_t bits;
    std::memcpy(&bits, &a, sizeof(bits));
    if (((bits >> 23) & 0xFFu) >= 158u)
        return (bits & 0x80000000u) ? INT32_MIN : INT32_MAX;
    return (int32_t)a;
}
#define FPU_CVT_W_S(a) (ps2_fpu_cvt_w_s((float)(a)))
#define FPU_CVT_L_S(a) ((int64_t)(float)(a))
#define FPU_C_F_S(a, b) (0)
#define FPU_C_UN_S(a, b) (isnan((float)(a)) || isnan((float)(b)))
#define FPU_C_EQ_S(a, b) ((float)(a) == (float)(b))
#define FPU_C_UEQ_S(a, b) ((float)(a) == (float)(b) || isnan((float)(a)) || isnan((float)(b)))
#define FPU_C_OLT_S(a, b) ((float)(a) < (float)(b))
#define FPU_C_ULT_S(a, b) ((float)(a) < (float)(b) || isnan((float)(a)) || isnan((float)(b)))
#define FPU_C_OLE_S(a, b) ((float)(a) <= (float)(b))
#define FPU_C_ULE_S(a, b) ((float)(a) <= (float)(b) || isnan((float)(a)) || isnan((float)(b)))
#define FPU_C_SF_S(a, b) (0)
#define FPU_C_NGLE_S(a, b) (isnan((float)(a)) || isnan((float)(b)))
#define FPU_C_SEQ_S(a, b) ((float)(a) == (float)(b))
#define FPU_C_NGL_S(a, b) ((float)(a) == (float)(b) || isnan((float)(a)) || isnan((float)(b)))
#define FPU_C_LT_S(a, b) ((float)(a) < (float)(b))
#define FPU_C_NGE_S(a, b) ((float)(a) < (float)(b) || isnan((float)(a)) || isnan((float)(b)))
#define FPU_C_LE_S(a, b) ((float)(a) <= (float)(b))
#define FPU_C_NGT_S(a, b) ((float)(a) <= (float)(b) || isnan((float)(a)) || isnan((float)(b)))

// QFSRV: Quadword Funnel Shift Right Variable
// Concatenates rs || rt (256 bits) and right-shifts by SA bits, taking lower 128 bits.
inline __m128i ps2_qfsrv(__m128i rs, __m128i rt, uint32_t sa)
{
    if (sa == 0)
        return rt;
    if (sa >= 128)
    {
        if (sa >= 256)
            return _mm_setzero_si128();
        uint32_t shift = sa - 128;
        if (shift == 0)
            return rs;
        // Shift rs right by (sa-128) bits
        uint32_t byteShift = shift / 8;
        uint32_t bitShift = shift % 8;
        // Byte shift rs right
        alignas(16) uint8_t buf[16] = {};
        alignas(16) uint8_t src[16];
        _mm_store_si128((__m128i *)src, rs);
        for (uint32_t i = 0; i + byteShift < 16; i++)
            buf[i] = src[i + byteShift];
        __m128i result = _mm_load_si128((__m128i *)buf);
        if (bitShift > 0)
            result = _mm_or_si128(_mm_srli_epi64(result, bitShift),
                                  _mm_slli_epi64(_mm_bsrli_si128(result, 8), 64 - bitShift));
        return result;
    }
    // sa is 1..127: result = (rs || rt) >> sa, lower 128 bits
    uint32_t byteShift = sa / 8;
    uint32_t bitShift = sa % 8;
    alignas(16) uint8_t combined[32];
    _mm_store_si128((__m128i *)(combined), rt);      // low 128 bits
    _mm_store_si128((__m128i *)(combined + 16), rs); // high 128 bits
    // Shift right by byteShift bytes
    alignas(16) uint8_t shifted[16];
    for (uint32_t i = 0; i < 16; i++)
        shifted[i] = (i + byteShift < 32) ? combined[i + byteShift] : 0;
    __m128i result = _mm_load_si128((__m128i *)shifted);
    if (bitShift > 0)
    {
        uint8_t extra = (byteShift + 16 < 32) ? combined[byteShift + 16] : 0;
        __m128i hi_byte = _mm_insert_epi8(_mm_setzero_si128(), extra, 15);
        alignas(16) uint8_t src32[32];
        for (uint32_t i = 0; i < 32; i++)
            src32[i] = combined[i];
        uint64_t lo0, lo1, hi0, hi1;
        std::memcpy(&lo0, src32, 8);
        std::memcpy(&lo1, src32 + 8, 8);
        std::memcpy(&hi0, src32 + 16, 8);
        std::memcpy(&hi1, src32 + 24, 8);
        // 256-bit right shift by sa bits
        uint64_t r0, r1;
        if (sa < 64)
        {
            r0 = (lo0 >> sa) | (lo1 << (64 - sa));
            r1 = (lo1 >> sa) | (hi0 << (64 - sa));
        }
        else if (sa < 128)
        {
            uint32_t s = sa - 64;
            if (s == 0)
            {
                r0 = lo1;
                r1 = hi0;
            }
            else
            {
                r0 = (lo1 >> s) | (hi0 << (64 - s));
                r1 = (hi0 >> s) | (hi1 << (64 - s));
            }
        }
        else
        {
            r0 = 0;
            r1 = 0; // handled above
        }
        result = _mm_set_epi64x((long long)r1, (long long)r0);
    }
    return result;
}
#define PS2_QFSRV(rs, rt, sa) ps2_qfsrv((__m128i)(rs), (__m128i)(rt), (uint32_t)(sa))
#define PS2_PCPYLD(rs, rt) _mm_unpacklo_epi64(rt, rs)
// MMI permutations, per the EE Core Instruction Set Manual. The halfword forms
// act on each 64-bit doubleword independently.
#define PS2_PEXEH(rt) _mm_shufflelo_epi16(_mm_shufflehi_epi16(rt, _MM_SHUFFLE(3, 0, 1, 2)), _MM_SHUFFLE(3, 0, 1, 2)) // h0<->h2
#define PS2_PEXCH(rt) _mm_shufflelo_epi16(_mm_shufflehi_epi16(rt, _MM_SHUFFLE(3, 1, 2, 0)), _MM_SHUFFLE(3, 1, 2, 0)) // h1<->h2
#define PS2_PREVH(rt) _mm_shufflelo_epi16(_mm_shufflehi_epi16(rt, _MM_SHUFFLE(0, 1, 2, 3)), _MM_SHUFFLE(0, 1, 2, 3)) // h3,h2,h1,h0
#define PS2_PEXEW(rt) _mm_shuffle_epi32(rt, _MM_SHUFFLE(3, 0, 1, 2))  // w0<->w2
#define PS2_PEXCW(rt) _mm_shuffle_epi32(rt, _MM_SHUFFLE(3, 1, 2, 0))  // w1<->w2
#define PS2_PROT3W(rt) _mm_shuffle_epi32(rt, _MM_SHUFFLE(3, 0, 2, 1)) // w0,w1,w2 <- w1,w2,w0; w3 kept

// VU0 macro-mode MAC flags for the fields in dest (bit 3 = x ... bit 0 = w): zero in
// bits 0-3, sign in bits 4-7 (overflow and underflow are not modelled). The status
// register keeps Z/S and their sticky copies.
static inline void ps2_vu0_set_mac(R5900Context *ctx, __m128 result, uint32_t dest)
{
    float lanes[4];
    std::memcpy(lanes, &result, sizeof(lanes));
    uint32_t mac = 0u;
    for (int lane = 0; lane < 4; ++lane)
    {
        const uint32_t bit = 3u - static_cast<uint32_t>(lane);
        if (!(dest & (1u << bit)))
            continue;
        if (lanes[lane] == 0.0f)
            mac |= 1u << bit;
        if (std::signbit(lanes[lane]))
            mac |= 1u << (bit + 4u);
    }
    ctx->vu0_mac_flags = mac;
    const uint16_t zs = static_cast<uint16_t>(((mac & 0x0Fu) ? 0x1u : 0u) | ((mac & 0xF0u) ? 0x2u : 0u));
    ctx->vu0_status = static_cast<uint16_t>((ctx->vu0_status & ~0x3u) | zs | (zs << 6));
}
#define PS2_VU0_SET_MAC(ctx_ptr, result, dest) ps2_vu0_set_mac((ctx_ptr), (result), (dest))

// Additional VU0 operations. The Q unit (DIV/SQRT/RSQRT) chops like the rest of the VU and reads exponent-255
// operands as ordinary numbers like the EE FPU does (ps2_f*), but a zero (or denormal) divisor gives
// +-0x7F7FFFFF here rather than the FPU's +-0x7FFFFFFF. RSQRT is a chopped divide by the chopped root.

static inline float Ps2VuDivQ(float fs, float ft)
{
    const uint32_t ufs = ps2_foperand(std::bit_cast<uint32_t>(fs));
    const uint32_t uft = std::bit_cast<uint32_t>(ft);
    if ((uft & PS2_FLOAT_EXP_MASK) == 0u)
        return std::bit_cast<float>(((ufs ^ uft) & PS2_FLOAT_SIGN_MASK) | 0x7F7FFFFFu);
    const double a = ps2_fexact(ufs), b = ps2_fexact(uft);
    // The chopped double quotient is the chopped quotient or one above; a 24x24-bit product decides.
    const uint32_t q = std::bit_cast<uint32_t>(ps2_chop_from_double(a / b));
    const uint32_t mag = q & PS2_FLOAT_MAX_BITS;
    if (mag != 0u && mag != PS2_FLOAT_MAX_BITS && std::fabs(ps2_fexact(q) * b) > std::fabs(a))
        return ps2_fstep_down(q);
    return std::bit_cast<float>(q);
}

static inline float Ps2VuSqrtQ(float ft)
{
    const uint32_t u = std::bit_cast<uint32_t>(ft) & PS2_FLOAT_MAX_BITS;
    if ((u & PS2_FLOAT_EXP_MASK) == 0u)
        return 0.0f;
    const double x = ps2_fexact(u);
    const uint32_t r = std::bit_cast<uint32_t>(ps2_chop_from_double(std::sqrt(x)));
    if (ps2_fexact(r) * ps2_fexact(r) > x)
        return ps2_fstep_down(r);
    return std::bit_cast<float>(r);
}

static inline float Ps2VuRsqrtQ(float fs, float ft)
{
    return Ps2VuDivQ(fs, Ps2VuSqrtQ(ft));
}

#define PS2_VU_DIV_Q(fs, ft) Ps2VuDivQ((fs), (ft))
#define PS2_VU_SQRT_Q(ft) Ps2VuSqrtQ((ft))
#define PS2_VU_RSQRT_Q(fs, ft) Ps2VuRsqrtQ((fs), (ft))

#define PS2_VSQRT(x) sqrtf(x)
#define PS2_VRSQRT(x) (1.0f / sqrtf(x))

#define GPR_U32(ctx_ptr, reg_idx) ((reg_idx == 0) ? 0U : static_cast<uint32_t>(PS2_EXTRACT_EPI32_0(ctx_ptr->r[reg_idx])))
#define GPR_S32(ctx_ptr, reg_idx) ((reg_idx == 0) ? 0 : PS2_EXTRACT_EPI32_0(ctx_ptr->r[reg_idx]))
#define GPR_U64(ctx_ptr, reg_idx) ((reg_idx == 0) ? 0ULL : static_cast<uint64_t>(PS2_EXTRACT_EPI64_0(ctx_ptr->r[reg_idx])))
#define GPR_S64(ctx_ptr, reg_idx) ((reg_idx == 0) ? 0LL : PS2_EXTRACT_EPI64_0(ctx_ptr->r[reg_idx]))
#define GPR_VEC(ctx_ptr, reg_idx) ((reg_idx == 0) ? _mm_setzero_si128() : ctx_ptr->r[reg_idx])

static inline void Ps2SetGprLow64(R5900Context *ctx, int reg, __m128i new_low)
{
    if (reg != 0)
    {
        ctx->r[reg] = _mm_castpd_si128(_mm_move_sd(_mm_castsi128_pd(ctx->r[reg]), _mm_castsi128_pd(new_low)));
    }
}

#define SET_GPR_U32(ctx_ptr, reg_idx, val)                                \
    do                                                                    \
    {                                                                     \
        if ((reg_idx) != 0)                                               \
        {                                                                 \
            __m128i _newVal = _mm_cvtsi64_si128((int64_t)(int32_t)(val)); \
                                                                          \
            Ps2SetGprLow64(ctx_ptr, reg_idx, _newVal);                    \
        }                                                                 \
    } while (0)


#define SET_GPR_ZE32(ctx_ptr, reg_idx, val)                               \
    do                                                                    \
    {                                                                     \
        if ((reg_idx) != 0)                                               \
        {                                                                 \
            __m128i _newVal = _mm_cvtsi64_si128((int64_t)(uint32_t)(val)); \
            Ps2SetGprLow64(ctx_ptr, reg_idx, _newVal);                    \
        }                                                                 \
    } while (0)

#define SET_GPR_S32(ctx_ptr, reg_idx, val)                                \
    do                                                                    \
    {                                                                     \
        if ((reg_idx) != 0)                                               \
        {                                                                 \
            __m128i _newVal = _mm_cvtsi64_si128((int64_t)(int32_t)(val)); \
            Ps2SetGprLow64(ctx_ptr, reg_idx, _newVal);                    \
        }                                                                 \
    } while (0)

#define SET_GPR_U64(ctx_ptr, reg_idx, val)                       \
    do                                                           \
    {                                                            \
        if ((reg_idx) != 0)                                      \
        {                                                        \
            __m128i _newVal = _mm_cvtsi64_si128((int64_t)(val)); \
            Ps2SetGprLow64(ctx_ptr, reg_idx, _newVal);           \
        }                                                        \
    } while (0)

#define SET_GPR_S64(ctx_ptr, reg_idx, val) SET_GPR_U64(ctx_ptr, reg_idx, val)

#define SET_GPR_VEC(ctx_ptr, reg_idx, val) \
    do                                     \
    {                                      \
        if (reg_idx != 0)                  \
            ctx_ptr->r[reg_idx] = (val);   \
    } while (0)

#endif // PS2_RUNTIME_MACROS_H
