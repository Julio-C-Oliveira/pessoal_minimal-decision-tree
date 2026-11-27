/**
 * @file ht_math_fixed.h
 * @brief Zero-libm Q16.16 fixed-point arithmetic and transcendental engine.
 *
 * Included automatically by ht_config.h when HT_USE_FIXED_POINT is defined.
 * Provides:
 * - Saturated 32-bit Q16.16 arithmetic (int64_t intermediate accumulators)
 *   with strict NaN/Infinity sentinel propagation and zero signed-shift UB.
 * - Zero-libm integer transcendental functions (sqrt, log2, log, exp, erf, erfc)
 *   targeting <= 2 ULPs error across the finite Q16.16 domain.
 * - Zero floating-point instructions, zero <math.h>, zero -lm dependency.
 *
 * Strictly adheres to ISO C11 (-std=c11 -Wall -Wextra -Werror -pedantic).
 */

#ifndef HT_MATH_FIXED_H
#define HT_MATH_FIXED_H

#ifdef HT_USE_FIXED_POINT

#include <stdint.h>
#include <stdbool.h>
#include <limits.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================== */
/* Saturated Q16.16 Elementary Arithmetic (Inline, 0% libm, 0% soft-fp)       */
/* ========================================================================== */

/**
 * @brief Saturates a 64-bit Q16.16 intermediate value into [-INT32_MAX, +INT32_MAX].
 * Never produces INT32_MIN (reserved for HT_NAN_VAL).
 */
static inline int32_t ht_fixed_saturate64(int64_t v) {
    if (v >= (int64_t)INT32_MAX) {
        return INT32_MAX;
    }
    if (v <= (int64_t)(-INT32_MAX)) {
        return -INT32_MAX;
    }
    return (int32_t)v;
}

/**
 * @brief Converts a signed 32-bit integer to Q16.16 with saturation.
 */
static inline int32_t ht_fixed_from_int(int32_t i) {
    if (i >= 32768) {
        return INT32_MAX;
    }
    if (i <= -32768) {
        return -INT32_MAX;
    }
    return (int32_t)((int64_t)i * 65536LL);
}

/**
 * @brief Converts an unsigned 32-bit integer to Q16.16 with saturation.
 */
static inline int32_t ht_fixed_from_uint(uint32_t u) {
    if (u >= 32768U) {
        return INT32_MAX;
    }
    return (int32_t)(u << 16);
}

/**
 * @brief Truncates a Q16.16 value toward zero into a signed 32-bit integer.
 */
static inline int32_t ht_fixed_to_int(int32_t q) {
    if (q == INT32_MIN) {
        return 0;
    }
    if (q >= 0) {
        return (int32_t)((uint32_t)q >> 16);
    }
    return -(int32_t)((uint32_t)(-(int64_t)q) >> 16);
}

/**
 * @brief Absolute value in Q16.16 with sentinel safety.
 */
static inline int32_t ht_fixed_abs(int32_t x) {
    if (x == INT32_MIN) {
        return INT32_MIN;
    }
    if (x == -INT32_MAX) {
        return INT32_MAX;
    }
    return (x < 0) ? -x : x;
}

/**
 * @brief Divides a finite Q16.16 value by 2 toward zero (used for Block-Rescaling).
 */
static inline int32_t ht_fixed_halve(int32_t x) {
    if (x == INT32_MIN || x == INT32_MAX || x == -INT32_MAX) {
        return x;
    }
    if (x >= 0) {
        return (int32_t)((uint32_t)x >> 1);
    }
    return -(int32_t)((uint32_t)(-(int64_t)x) >> 1);
}

/**
 * @brief Saturated addition in Q16.16 with NaN and +/-Infinity semantics.
 */
static inline int32_t ht_fixed_add(int32_t a, int32_t b) {
    if (a == INT32_MIN || b == INT32_MIN) {
        return INT32_MIN;
    }
    if (a == INT32_MAX) {
        return (b == -INT32_MAX) ? INT32_MIN : INT32_MAX;
    }
    if (a == -INT32_MAX) {
        return (b == INT32_MAX) ? INT32_MIN : -INT32_MAX;
    }
    if (b == INT32_MAX || b == -INT32_MAX) {
        return b;
    }
    return ht_fixed_saturate64((int64_t)a + (int64_t)b);
}

/**
 * @brief Saturated subtraction in Q16.16 with NaN and +/-Infinity semantics.
 */
static inline int32_t ht_fixed_sub(int32_t a, int32_t b) {
    if (a == INT32_MIN || b == INT32_MIN) {
        return INT32_MIN;
    }
    if (b == INT32_MAX) {
        return (a == INT32_MAX) ? INT32_MIN : -INT32_MAX;
    }
    if (b == -INT32_MAX) {
        return (a == -INT32_MAX) ? INT32_MIN : INT32_MAX;
    }
    if (a == INT32_MAX || a == -INT32_MAX) {
        return a;
    }
    return ht_fixed_saturate64((int64_t)a - (int64_t)b);
}

/**
 * @brief Saturated multiplication in Q16.16 with symmetric rounding to nearest.
 */
static inline int32_t ht_fixed_mul(int32_t a, int32_t b) {
    if (a == INT32_MIN || b == INT32_MIN) {
        return INT32_MIN;
    }
    bool a_inf = (a == INT32_MAX || a == -INT32_MAX);
    bool b_inf = (b == INT32_MAX || b == -INT32_MAX);
    if (a_inf || b_inf) {
        if (a == 0 || b == 0) {
            return INT32_MIN; /* 0 * inf = NaN */
        }
        return ((a > 0) == (b > 0)) ? INT32_MAX : -INT32_MAX;
    }
    if (a == 0 || b == 0) {
        return 0;
    }
    int64_t prod = (int64_t)a * (int64_t)b;
    if (prod >= 0) {
        uint64_t mag = ((uint64_t)prod + 32768ULL) >> 16;
        return (mag >= (uint64_t)INT32_MAX) ? INT32_MAX : (int32_t)mag;
    } else {
        uint64_t mag = ((uint64_t)(-prod) + 32768ULL) >> 16;
        return (mag >= (uint64_t)INT32_MAX) ? -INT32_MAX : -(int32_t)mag;
    }
}

/**
 * @brief Saturated division in Q16.16 with symmetric rounding to nearest.
 */
static inline int32_t ht_fixed_div(int32_t a, int32_t b) {
    if (a == INT32_MIN || b == INT32_MIN) {
        return INT32_MIN;
    }
    bool a_inf = (a == INT32_MAX || a == -INT32_MAX);
    bool b_inf = (b == INT32_MAX || b == -INT32_MAX);
    if (b == 0) {
        if (a == 0) {
            return INT32_MIN; /* 0 / 0 = NaN */
        }
        return (a > 0) ? INT32_MAX : -INT32_MAX;
    }
    if (a_inf) {
        if (b_inf) {
            return INT32_MIN; /* inf / inf = NaN */
        }
        return ((a > 0) == (b > 0)) ? INT32_MAX : -INT32_MAX;
    }
    if (b_inf) {
        return 0;
    }
    if (a == 0) {
        return 0;
    }
    bool neg = (a < 0) != (b < 0);
    uint64_t ua = (a < 0) ? (uint64_t)(-(int64_t)a) : (uint64_t)a;
    uint64_t ub = (b < 0) ? (uint64_t)(-(int64_t)b) : (uint64_t)b;
    uint64_t num = (ua << 16) + (ub >> 1);
    uint64_t q = num / ub;
    if (q >= (uint64_t)INT32_MAX) {
        return neg ? -INT32_MAX : INT32_MAX;
    }
    return neg ? -(int32_t)q : (int32_t)q;
}

/* ========================================================================== */
/* Zero-Libm Integer Transcendental Engine (implemented in ht_math_fixed.c)   */
/* ========================================================================== */

/**
 * @brief Exact integer square root in Q16.16 (error <= 1 ULP).
 * Returns 0 for x == 0, HT_NAN_VAL for x < 0 or NaN, HT_INFINITY for +Inf.
 */
int32_t ht_fixed_sqrt(int32_t x);

/**
 * @brief Base-2 logarithm in Q16.16 via CLZ normalization + degree-7 minimax polynomial.
 * Returns -HT_INFINITY for x == 0, HT_NAN_VAL for x < 0 or NaN, +HT_INFINITY for +Inf.
 */
int32_t ht_fixed_log2(int32_t x);

/**
 * @brief Natural logarithm in Q16.16 (ln(x) = log2(x) * ln(2)).
 */
int32_t ht_fixed_log(int32_t x);

/**
 * @brief Exponential e^x in Q16.16 via argument reduction e^x = 2^i * 2^f.
 */
int32_t ht_fixed_exp(int32_t x);

/**
 * @brief Error function erf(x) in Q16.16 via Abramowitz & Stegun (7.1.26).
 */
int32_t ht_fixed_erf(int32_t x);

/**
 * @brief Complementary error function erfc(x) = 1 - erf(x) in Q16.16.
 */
int32_t ht_fixed_erfc(int32_t x);

/**
 * @brief Exact integer square root of an unsigned 64-bit integer (floor(sqrt(n))).
 * Useful for high-dynamic-range variance/ADWIN computations without overflow.
 */
uint32_t ht_fixed_isqrt64(uint64_t n);

#ifdef __cplusplus
}
#endif

#endif /* HT_USE_FIXED_POINT */

#endif /* HT_MATH_FIXED_H */
