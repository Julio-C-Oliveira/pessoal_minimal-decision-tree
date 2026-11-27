/**
 * @file ht_math_fixed.c
 * @brief Zero-libm Q16.16 integer transcendental engine (sqrt, log2, log, exp, erf, erfc).
 *
 * Compiled with zero floating-point operations, zero <math.h> headers, and zero -lm
 * dependencies when HT_USE_FIXED_POINT is defined.
 *
 * Strictly adheres to ISO C11 (-std=c11 -Wall -Wextra -Werror -pedantic).
 */

#include "ht_config.h"

#ifdef HT_USE_FIXED_POINT

/* ========================================================================== */
/* Internal Q30 Fixed-Point Helpers & Minimax Polynomial Constants            */
/* ========================================================================== */

#define HT_Q30_ONE        (1LL << 30)
#define HT_Q30_HALF       (1LL << 29)
#define HT_Q30_MASK       ((1ULL << 30) - 1ULL)

/* High-precision Q30 constants */
#define HT_LN2_Q30        744261118LL    /* ln(2)    * 2^30 */
#define HT_LOG2E_Q30      1549082005LL   /* log2(e)  * 2^30 */

/* Degree-7 Chebyshev minimax polynomial for log2(1 + u), u in [0, 1] (max err ~0.12 ULP) */
#define HT_LOG2_C1_Q30    1549079795LL
#define HT_LOG2_C2_Q30   (-774322573LL)
#define HT_LOG2_C3_Q30     512707816LL
#define HT_LOG2_C4_Q30   (-363329749LL)
#define HT_LOG2_C5_Q30     229719775LL
#define HT_LOG2_C6_Q30   (-101604763LL)
#define HT_LOG2_C7_Q30      21491523LL

/* Degree-5 Chebyshev minimax polynomial for 2^f - 1, f in [0, 1] (max err ~0.04 ULP) */
#define HT_EXP2_C1_Q30     744261533LL
#define HT_EXP2_C2_Q30     257920511LL
#define HT_EXP2_C3_Q30      59761307LL
#define HT_EXP2_C4_Q30       9877767LL
#define HT_EXP2_C5_Q30       1920705LL

/* Abramowitz & Stegun 7.1.26 rational approximation constants for erfc(x) in Q30 */
#define HT_AS_P_Q30        351748265LL   /*  0.3275911   * 2^30 */
#define HT_AS_A1_Q30       273621191LL   /*  0.254829592 * 2^30 */
#define HT_AS_A2_Q30     (-305476044LL)  /* -0.284496736 * 2^30 */
#define HT_AS_A3_Q30      1526231383LL   /*  1.421413741 * 2^30 */
#define HT_AS_A4_Q30    (-1560310108LL)  /* -1.453152027 * 2^30 */
#define HT_AS_A5_Q30      1139675401LL   /*  1.061405429 * 2^30 */

/* Range guard thresholds for exp(x) in Q16.16 */
#define HT_EXP_OVERFLOW_Q16   681392     /*  ln(32767.999) * 65536 */
#define HT_EXP_UNDERFLOW_Q16 (-772245)   /*  ln(2^-17)     * 65536 */
#define HT_ERF_SAT_Q16        262144     /*  4.0           * 65536 */

static inline int ht_clz32(uint32_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_clz(x);
#else
    int n = 0;
    if ((x & 0xFFFF0000U) == 0U) { n += 16; x <<= 16; }
    if ((x & 0xFF000000U) == 0U) { n += 8;  x <<= 8;  }
    if ((x & 0xF0000000U) == 0U) { n += 4;  x <<= 4;  }
    if ((x & 0xC0000000U) == 0U) { n += 2;  x <<= 2;  }
    if ((x & 0x80000000U) == 0U) { n += 1; }
    return n;
#endif
}

static inline int64_t ht_q30_mul(int64_t a, int64_t b) {
    int64_t prod = a * b;
    if (prod >= 0) {
        return (int64_t)(((uint64_t)prod + (uint64_t)HT_Q30_HALF) >> 30);
    }
    return -(int64_t)(((uint64_t)(-prod) + (uint64_t)HT_Q30_HALF) >> 30);
}

static inline int32_t ht_q30_to_q16(int64_t v_q30) {
    if (v_q30 >= 0) {
        uint64_t mag = ((uint64_t)v_q30 + (1ULL << 13)) >> 14;
        return (mag >= (uint64_t)INT32_MAX) ? INT32_MAX : (int32_t)mag;
    }
    uint64_t mag = ((uint64_t)(-v_q30) + (1ULL << 13)) >> 14;
    return (mag >= (uint64_t)INT32_MAX) ? -INT32_MAX : -(int32_t)mag;
}

static inline int64_t ht_exp2_frac_q30(int64_t f_q30) {
    int64_t acc = HT_EXP2_C5_Q30;
    acc = HT_EXP2_C4_Q30 + ht_q30_mul(acc, f_q30);
    acc = HT_EXP2_C3_Q30 + ht_q30_mul(acc, f_q30);
    acc = HT_EXP2_C2_Q30 + ht_q30_mul(acc, f_q30);
    acc = HT_EXP2_C1_Q30 + ht_q30_mul(acc, f_q30);
    return HT_Q30_ONE + ht_q30_mul(acc, f_q30);
}

static inline void ht_split_floor_q30(int64_t z_q30, int32_t *out_i, int64_t *out_f) {
    if (z_q30 >= 0) {
        uint64_t uz = (uint64_t)z_q30;
        *out_i = (int32_t)(uz >> 30);
        *out_f = (int64_t)(uz & HT_Q30_MASK);
    } else {
        uint64_t uz = (uint64_t)(-z_q30);
        uint64_t q = uz >> 30;
        uint64_t r = uz & HT_Q30_MASK;
        if (r == 0ULL) {
            *out_i = -(int32_t)q;
            *out_f = 0;
        } else {
            *out_i = -(int32_t)(q + 1ULL);
            *out_f = (int64_t)((1ULL << 30) - r);
        }
    }
}

/* Computes exp(neg_x) in Q30 for neg_x in [-16.0, 0.0] (given in Q16.16) */
static int64_t ht_exp_neg_q30(int64_t neg_x_q16) {
    if (neg_x_q16 >= 0) {
        return HT_Q30_ONE;
    }
    uint64_t ux = (uint64_t)(-neg_x_q16);
    int64_t z_q30 = -(int64_t)((ux * (uint64_t)HT_LOG2E_Q30 + 32768ULL) >> 16);

    int32_t i = 0;
    int64_t f_q30 = 0;
    ht_split_floor_q30(z_q30, &i, &f_q30);

    if (i <= -31) {
        return 0;
    }
    int64_t y_q30 = ht_exp2_frac_q30(f_q30);
    if (i >= 0) {
        return y_q30;
    }
    unsigned int rshift = (unsigned int)(-i);
    return (int64_t)(((uint64_t)y_q30 + (1ULL << (rshift - 1U))) >> rshift);
}

static void ht_log2_decompose(uint32_t ux, int32_t *out_e, int64_t *out_frac_q30) {
    int msb = 31 - ht_clz32(ux);
    *out_e = (int32_t)(msb - 16);

    uint64_t m_q30 = ((uint64_t)ux) << (unsigned int)(30 - msb);
    int64_t u_q30 = (int64_t)(m_q30 - (1ULL << 30));

    int64_t acc = HT_LOG2_C7_Q30;
    acc = HT_LOG2_C6_Q30 + ht_q30_mul(acc, u_q30);
    acc = HT_LOG2_C5_Q30 + ht_q30_mul(acc, u_q30);
    acc = HT_LOG2_C4_Q30 + ht_q30_mul(acc, u_q30);
    acc = HT_LOG2_C3_Q30 + ht_q30_mul(acc, u_q30);
    acc = HT_LOG2_C2_Q30 + ht_q30_mul(acc, u_q30);
    acc = HT_LOG2_C1_Q30 + ht_q30_mul(acc, u_q30);
    *out_frac_q30 = ht_q30_mul(acc, u_q30);
}

/* ========================================================================== */
/* Public Zero-Libm Transcendental Functions                                  */
/* ========================================================================== */

uint32_t ht_fixed_isqrt64(uint64_t n) {
    if (n == 0ULL) {
        return 0U;
    }
    uint64_t rem = n;
    uint64_t res = 0ULL;
    uint64_t bit = 1ULL << 62;
    while (bit > rem) {
        bit >>= 2;
    }
    while (bit != 0ULL) {
        if (rem >= res + bit) {
            rem -= res + bit;
            res = (res >> 1) + bit;
        } else {
            res >>= 1;
        }
        bit >>= 2;
    }
    return (uint32_t)res;
}

int32_t ht_fixed_sqrt(int32_t x) {
    if (x == INT32_MIN || x < 0) {
        return INT32_MIN;
    }
    if (x == 0) {
        return 0;
    }
    if (x == INT32_MAX) {
        return INT32_MAX;
    }
    /* Compute floor(2 * sqrt(x)) in Q16.16, then round to nearest ULP */
    uint32_t r2 = ht_fixed_isqrt64(((uint64_t)(uint32_t)x) << 18);
    return (int32_t)((r2 + 1U) >> 1);
}

int32_t ht_fixed_log2(int32_t x) {
    if (x == INT32_MIN || x < 0) {
        return INT32_MIN;
    }
    if (x == 0) {
        return -INT32_MAX;
    }
    if (x == INT32_MAX) {
        return INT32_MAX;
    }
    int32_t e = 0;
    int64_t frac_q30 = 0;
    ht_log2_decompose((uint32_t)x, &e, &frac_q30);
    int64_t log2_q30 = ((int64_t)e * HT_Q30_ONE) + frac_q30;
    return ht_q30_to_q16(log2_q30);
}

int32_t ht_fixed_log(int32_t x) {
    if (x == INT32_MIN || x < 0) {
        return INT32_MIN;
    }
    if (x == 0) {
        return -INT32_MAX;
    }
    if (x == INT32_MAX) {
        return INT32_MAX;
    }
    int32_t e = 0;
    int64_t frac_q30 = 0;
    ht_log2_decompose((uint32_t)x, &e, &frac_q30);
    int64_t ln_q30 = ((int64_t)e * HT_LN2_Q30) + ht_q30_mul(frac_q30, HT_LN2_Q30);
    return ht_q30_to_q16(ln_q30);
}

int32_t ht_fixed_exp(int32_t x) {
    if (x == INT32_MIN) {
        return INT32_MIN;
    }
    if (x == INT32_MAX || x >= HT_EXP_OVERFLOW_Q16) {
        return INT32_MAX;
    }
    if (x == -INT32_MAX || x <= HT_EXP_UNDERFLOW_Q16) {
        return 0;
    }

    int64_t z_q30;
    if (x >= 0) {
        z_q30 = (int64_t)(((uint64_t)(uint32_t)x * (uint64_t)HT_LOG2E_Q30 + 32768ULL) >> 16);
    } else {
        z_q30 = -(int64_t)(((uint64_t)(-(int64_t)x) * (uint64_t)HT_LOG2E_Q30 + 32768ULL) >> 16);
    }

    int32_t i = 0;
    int64_t f_q30 = 0;
    ht_split_floor_q30(z_q30, &i, &f_q30);

    if (i >= 15) {
        return INT32_MAX;
    }
    if (i <= -17) {
        return 0;
    }

    int64_t y_q30 = ht_exp2_frac_q30(f_q30);
    unsigned int rshift = (unsigned int)(14 - i);
    if (rshift == 0U) {
        return (y_q30 >= (int64_t)INT32_MAX) ? INT32_MAX : (int32_t)y_q30;
    }
    uint64_t val = ((uint64_t)y_q30 + (1ULL << (rshift - 1U))) >> rshift;
    return (val >= (uint64_t)INT32_MAX) ? INT32_MAX : (int32_t)val;
}

static int32_t ht_erfc_nonneg(int32_t ax) {
    if (ax <= 0) {
        return HT_FIXED_ONE;
    }
    if (ax >= HT_ERF_SAT_Q16) {
        return 0;
    }

    uint64_t uax = (uint64_t)(uint32_t)ax;
    int64_t denom_q30 = HT_Q30_ONE + (int64_t)((uax * (uint64_t)HT_AS_P_Q30 + 32768ULL) >> 16);
    int64_t t_q30 = (int64_t)(((1ULL << 60) + ((uint64_t)denom_q30 >> 1)) / (uint64_t)denom_q30);

    int64_t acc = HT_AS_A5_Q30;
    acc = HT_AS_A4_Q30 + ht_q30_mul(acc, t_q30);
    acc = HT_AS_A3_Q30 + ht_q30_mul(acc, t_q30);
    acc = HT_AS_A2_Q30 + ht_q30_mul(acc, t_q30);
    acc = HT_AS_A1_Q30 + ht_q30_mul(acc, t_q30);
    int64_t poly_q30 = ht_q30_mul(acc, t_q30);

    int64_t neg_x2_q16 = -(int64_t)((uax * uax + 32768ULL) >> 16);
    int64_t exp_q30 = ht_exp_neg_q30(neg_x2_q16);
    int64_t res_q30 = ht_q30_mul(poly_q30, exp_q30);

    if (res_q30 <= 0) {
        return 0;
    }
    int32_t res_q16 = ht_q30_to_q16(res_q30);
    if (res_q16 > HT_FIXED_ONE) {
        res_q16 = HT_FIXED_ONE;
    }
    return res_q16;
}

int32_t ht_fixed_erfc(int32_t x) {
    if (x == INT32_MIN) {
        return INT32_MIN;
    }
    if (x >= 0) {
        return ht_erfc_nonneg(x);
    }
    int32_t ax = (x == -INT32_MAX) ? INT32_MAX : -x;
    return (2 * HT_FIXED_ONE) - ht_erfc_nonneg(ax);
}

int32_t ht_fixed_erf(int32_t x) {
    if (x == INT32_MIN) {
        return INT32_MIN;
    }
    if (x == 0) {
        return 0;
    }
    bool neg = (x < 0);
    int32_t ax = neg ? ((x == -INT32_MAX) ? INT32_MAX : -x) : x;
    int32_t mag = HT_FIXED_ONE - ht_erfc_nonneg(ax);
    return neg ? -mag : mag;
}

#else

/* ISO C11 requires at least one external declaration per translation unit */
typedef int ht_math_fixed_unused_t;

#endif /* HT_USE_FIXED_POINT */
