/**
 * @file test_fixed.c
 * @brief Level 1 & 2 unit tests for Q16.16 zero-libm fixed-point math engine
 *        and precision-agnostic arithmetic abstraction macros.
 *
 * Can be executed inside test_runner (all 3 precision modes) or standalone
 * with -DHT_USE_FIXED_POINT -DHT_FIXED_STANDALONE_TEST without -lm.
 */

#include "test_harness.h"

#ifdef HT_FIXED_STANDALONE_TEST
int g_tests_run = 0;
int g_tests_failed = 0;
int g_assertions_run = 0;
bool g_current_test_failed = false;
#endif

static void test_abstract_arithmetic_macros(void) {
    ht_float_t a = HT_CONST(12.5);
    ht_float_t b = HT_CONST(2.5);

    HT_ASSERT_FLOAT_EQ(HT_ADD(a, b), HT_CONST(15.0));
    HT_ASSERT_FLOAT_EQ(HT_SUB(a, b), HT_CONST(10.0));
    HT_ASSERT_FLOAT_EQ(HT_MUL(a, b), HT_CONST(31.25));
    HT_ASSERT_FLOAT_EQ(HT_DIV(a, b), HT_CONST(5.0));

    HT_ASSERT_FLOAT_EQ(HT_FROM_INT(-7), HT_CONST(-7.0));
    HT_ASSERT_FLOAT_EQ(HT_FROM_UINT(42U), HT_CONST(42.0));
    HT_ASSERT_EQ(HT_TO_INT(HT_CONST(12.9)), 12);
    HT_ASSERT_EQ(HT_TO_INT(HT_CONST(-12.9)), -12);
    HT_ASSERT_FLOAT_NEAR(HT_LN2, HT_CONST(0.6931471805599453), HT_FLOAT_TOL);
    HT_ASSERT_FLOAT_NEAR(HT_LN10, HT_CONST(2.3025850929940457), HT_FLOAT_TOL);
    HT_ASSERT_FLOAT_NEAR(HT_LN_2PI, HT_CONST(1.8378770664093455), HT_FLOAT_TOL);
}

#ifdef HT_USE_FIXED_POINT

_Static_assert(sizeof(ht_float_t) == 4U, "Q16.16 ht_float_t must be exactly 4 bytes");
_Static_assert(sizeof(ht_value_t) == 4U, "Q16.16 ht_value_t must be exactly 4 bytes");

static void test_fixed_sentinels_and_saturation(void) {
    /* Sentinel identification */
    HT_ASSERT(HT_ISNAN(HT_NAN_VAL));
    HT_ASSERT(!HT_ISFINITE(HT_NAN_VAL));
    HT_ASSERT(!HT_ISFINITE(HT_INFINITY));
    HT_ASSERT(!HT_ISFINITE(-HT_INFINITY));
    HT_ASSERT(!HT_ISFINITE(HT_NEG_INFINITY));
    HT_ASSERT(HT_ISFINITE(HT_CONST(0.0)));
    HT_ASSERT(HT_ISFINITE(HT_CONST(32767.0)));
    HT_ASSERT(HT_ISFINITE(HT_CONST(-32767.0)));

    /* NaN propagation across all elementary operators */
    HT_ASSERT(HT_ISNAN(HT_ADD(HT_NAN_VAL, HT_CONST(1.0))));
    HT_ASSERT(HT_ISNAN(HT_SUB(HT_CONST(1.0), HT_NAN_VAL)));
    HT_ASSERT(HT_ISNAN(HT_MUL(HT_NAN_VAL, HT_CONST(2.0))));
    HT_ASSERT(HT_ISNAN(HT_DIV(HT_CONST(2.0), HT_NAN_VAL)));
    HT_ASSERT(HT_ISNAN(HT_FABS(HT_NAN_VAL)));

    /* Indeterminate forms -> NaN */
    HT_ASSERT(HT_ISNAN(HT_ADD(HT_INFINITY, -HT_INFINITY)));
    HT_ASSERT(HT_ISNAN(HT_SUB(HT_INFINITY, HT_INFINITY)));
    HT_ASSERT(HT_ISNAN(HT_MUL(HT_CONST(0.0), HT_INFINITY)));
    HT_ASSERT(HT_ISNAN(HT_DIV(HT_CONST(0.0), HT_CONST(0.0))));
    HT_ASSERT(HT_ISNAN(HT_DIV(HT_INFINITY, HT_INFINITY)));

    /* Division by zero -> +/-Infinity */
    HT_ASSERT_EQ(HT_DIV(HT_CONST(5.0), HT_CONST(0.0)), HT_INFINITY);
    HT_ASSERT_EQ(HT_DIV(HT_CONST(-5.0), HT_CONST(0.0)), -HT_INFINITY);
    HT_ASSERT_EQ(HT_DIV(HT_CONST(5.0), HT_INFINITY), 0);

    /* Saturation must never wrap around into INT32_MIN (HT_NAN_VAL) */
    ht_float_t big_pos = HT_CONST(30000.0);
    ht_float_t big_neg = HT_CONST(-30000.0);
    HT_ASSERT_EQ(HT_ADD(big_pos, big_pos), HT_INFINITY);
    HT_ASSERT_EQ(HT_ADD(big_neg, big_neg), -HT_INFINITY);
    HT_ASSERT_EQ(HT_SUB(big_neg, big_pos), -HT_INFINITY);
    HT_ASSERT_EQ(HT_MUL(HT_CONST(500.0), HT_CONST(500.0)), HT_INFINITY);
    HT_ASSERT_EQ(HT_MUL(HT_CONST(-500.0), HT_CONST(500.0)), -HT_INFINITY);
    HT_ASSERT_EQ(HT_DIV(HT_CONST(20000.0), HT_CONST(0.1)), HT_INFINITY);
    HT_ASSERT_EQ(HT_DIV(HT_CONST(-20000.0), HT_CONST(0.1)), -HT_INFINITY);
    HT_ASSERT_EQ(HT_FROM_INT(40000), HT_INFINITY);
    HT_ASSERT_EQ(HT_FROM_INT(-40000), -HT_INFINITY);
    HT_ASSERT_EQ(HT_FROM_UINT(40000U), HT_INFINITY);

    /* Halving helper for Block-Rescaling */
    HT_ASSERT_EQ(ht_fixed_halve(HT_CONST(16384.0)), HT_CONST(8192.0));
    HT_ASSERT_EQ(ht_fixed_halve(HT_CONST(-100.5)), HT_CONST(-50.25));
    HT_ASSERT_EQ(ht_fixed_halve(HT_NAN_VAL), HT_NAN_VAL);
}

static void test_fixed_sqrt_accuracy(void) {
    /* Edge cases */
    HT_ASSERT_EQ(HT_SQRT(HT_CONST(0.0)), 0);
    HT_ASSERT(HT_ISNAN(HT_SQRT(HT_CONST(-1.0))));
    HT_ASSERT(HT_ISNAN(HT_SQRT(HT_NAN_VAL)));
    HT_ASSERT_EQ(HT_SQRT(HT_INFINITY), HT_INFINITY);

    /* Ground truth table (tolerance <= 1 ULP!) */
    static const struct { ht_float_t x; ht_float_t expected; } cases[] = {
        { 1,                     256 },                /* sqrt(2^-16) = 2^-8 = 256/65536 */
        { HT_CONST(0.0625),      HT_CONST(0.25) },
        { HT_CONST(0.25),        HT_CONST(0.5) },
        { HT_CONST(0.5),         HT_CONST(0.70710678) },
        { HT_CONST(1.0),         HT_CONST(1.0) },
        { HT_CONST(2.0),         HT_CONST(1.41421356) },
        { HT_CONST(3.0),         HT_CONST(1.73205081) },
        { HT_CONST(4.0),         HT_CONST(2.0) },
        { HT_CONST(9.0),         HT_CONST(3.0) },
        { HT_CONST(100.0),       HT_CONST(10.0) },
        { HT_CONST(1000.0),      HT_CONST(31.6227766) },
        { HT_CONST(10000.0),     HT_CONST(100.0) },
        { HT_CONST(32767.0),     HT_CONST(181.0165738) }
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        ht_float_t got = HT_SQRT(cases[i].x);
        ht_float_t diff = HT_FABS(HT_SUB(got, cases[i].expected));
        HT_ASSERT(diff <= 1);
    }

    /* 64-bit integer sqrt helper */
    HT_ASSERT_EQ(ht_fixed_isqrt64(0ULL), 0U);
    HT_ASSERT_EQ(ht_fixed_isqrt64(1ULL), 1U);
    HT_ASSERT_EQ(ht_fixed_isqrt64(1000000000000ULL), 1000000U);
    HT_ASSERT_EQ(ht_fixed_isqrt64(10000000000000000ULL), 100000000U);
}

static void test_fixed_log2_and_log_accuracy(void) {
    /* Edge cases */
    HT_ASSERT_EQ(HT_LOG2(0), -HT_INFINITY);
    HT_ASSERT_EQ(HT_LOG(0), -HT_INFINITY);
    HT_ASSERT(HT_ISNAN(HT_LOG2(HT_CONST(-1.0))));
    HT_ASSERT(HT_ISNAN(HT_LOG(HT_CONST(-1.0))));
    HT_ASSERT(HT_ISNAN(HT_LOG2(HT_NAN_VAL)));
    HT_ASSERT_EQ(HT_LOG2(HT_INFINITY), HT_INFINITY);
    HT_ASSERT_EQ(HT_LOG(HT_INFINITY), HT_INFINITY);

    /* log2 & ln ground truth table for exact Q16.16 inputs (strict tolerance <= 1 ULP!) */
    static const struct { ht_float_t x; ht_float_t log2_exp; ht_float_t ln_exp; } cases[] = {
        { 1,                     HT_CONST(-16.0),        HT_CONST(-11.09035489) },
        { HT_CONST(0.01),        HT_CONST(-6.64464890),  HT_CONST(-4.60571965) }, /* 655/65536 */
        { HT_CONST(0.1),         HT_CONST(-3.32184004),  HT_CONST(-2.30252406) }, /* 6554/65536 */
        { HT_CONST(0.25),        HT_CONST(-2.0),         HT_CONST(-1.38629436) },
        { HT_CONST(0.5),         HT_CONST(-1.0),         HT_CONST(-0.69314718) },
        { HT_CONST(0.75),        HT_CONST(-0.41503750),  HT_CONST(-0.28768207) },
        { HT_CONST(1.0),         HT_CONST(0.0),          HT_CONST(0.0) },
        { HT_CONST(1.5),         HT_CONST(0.58496250),   HT_CONST(0.40546511) },
        { HT_CONST(2.0),         HT_CONST(1.0),          HT_CONST(0.69314718) },
        { HT_CONST(2.71828183),  HT_CONST(1.44269247),   HT_CONST(0.99999822) }, /* 178145/65536 */
        { HT_CONST(3.0),         HT_CONST(1.58496250),   HT_CONST(1.09861229) },
        { HT_CONST(8.0),         HT_CONST(3.0),          HT_CONST(2.07944154) },
        { HT_CONST(10.0),        HT_CONST(3.32192809),   HT_CONST(2.30258509) },
        { HT_CONST(100.0),       HT_CONST(6.64385619),   HT_CONST(4.60517019) },
        { HT_CONST(1000.0),      HT_CONST(9.96578428),   HT_CONST(6.90775528) },
        { HT_CONST(16384.0),     HT_CONST(14.0),         HT_CONST(9.70406053) },
        { HT_CONST(32767.0),     HT_CONST(14.99995597),  HT_CONST(10.39717719) }
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        ht_float_t l2 = HT_LOG2(cases[i].x);
        ht_float_t ln = HT_LOG(cases[i].x);
        HT_ASSERT(HT_FABS(HT_SUB(l2, cases[i].log2_exp)) <= 1);
        HT_ASSERT(HT_FABS(HT_SUB(ln, cases[i].ln_exp)) <= 1);
    }
}

static void test_fixed_exp_accuracy(void) {
    /* Edge cases */
    HT_ASSERT(HT_ISNAN(HT_EXP(HT_NAN_VAL)));
    HT_ASSERT_EQ(HT_EXP(HT_INFINITY), HT_INFINITY);
    HT_ASSERT_EQ(HT_EXP(-HT_INFINITY), 0);
    HT_ASSERT_EQ(HT_EXP(HT_CONST(11.0)), HT_INFINITY);
    HT_ASSERT_EQ(HT_EXP(HT_CONST(-12.0)), 0);
    HT_ASSERT_EQ(HT_EXP(HT_CONST(0.0)), HT_FIXED_ONE);

    static const struct { ht_float_t x; ht_float_t expected; ht_float_t max_ulp; } cases[] = {
        { HT_CONST(-8.0),  HT_CONST(0.00033546),     2 },
        { HT_CONST(-4.0),  HT_CONST(0.01831564),     2 },
        { HT_CONST(-2.0),  HT_CONST(0.13533528),     2 },
        { HT_CONST(-1.0),  HT_CONST(0.36787944),     2 },
        { HT_CONST(-0.5),  HT_CONST(0.60653066),     2 },
        { HT_CONST(0.5),   HT_CONST(1.64872127),     2 },
        { HT_CONST(1.0),   HT_CONST(2.71828183),     2 },
        { HT_CONST(2.0),   HT_CONST(7.38905610),     2 },
        { HT_CONST(5.0),   HT_CONST(148.41315910),   4 },
        { HT_CONST(8.0),   HT_CONST(2980.95798704),  32 },
        { HT_CONST(10.0),  HT_CONST(22026.46579481), 160 } /* < 0.01% relative error */
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        ht_float_t got = HT_EXP(cases[i].x);
        HT_ASSERT(HT_FABS(HT_SUB(got, cases[i].expected)) <= cases[i].max_ulp);
    }
}

static void test_fixed_erf_and_erfc_accuracy(void) {
    /* Edge cases */
    HT_ASSERT(HT_ISNAN(HT_ERF(HT_NAN_VAL)));
    HT_ASSERT(HT_ISNAN(HT_ERFC(HT_NAN_VAL)));
    HT_ASSERT_EQ(HT_ERF(0), 0);
    HT_ASSERT_EQ(HT_ERFC(0), HT_FIXED_ONE);
    HT_ASSERT_EQ(HT_ERF(HT_CONST(4.5)), HT_FIXED_ONE);
    HT_ASSERT_EQ(HT_ERF(HT_CONST(-4.5)), -HT_FIXED_ONE);
    HT_ASSERT_EQ(HT_ERFC(HT_CONST(4.5)), 0);
    HT_ASSERT_EQ(HT_ERFC(HT_CONST(-4.5)), 2 * HT_FIXED_ONE);

    static const struct { ht_float_t x; ht_float_t erf_exp; ht_float_t erfc_exp; } cases[] = {
        { HT_CONST(0.1),     HT_CONST(0.11246292), HT_CONST(0.88753708) },
        { HT_CONST(0.25),    HT_CONST(0.27632639), HT_CONST(0.72367361) },
        { HT_CONST(0.5),     HT_CONST(0.52049988), HT_CONST(0.47950012) },
        { HT_CONST(0.70711), HT_CONST(0.68269214), HT_CONST(0.31730786) },
        { HT_CONST(1.0),     HT_CONST(0.84270079), HT_CONST(0.15729921) },
        { HT_CONST(1.5),     HT_CONST(0.96610515), HT_CONST(0.03389485) },
        { HT_CONST(2.0),     HT_CONST(0.99532227), HT_CONST(0.00467773) },
        { HT_CONST(3.0),     HT_CONST(0.99997791), HT_CONST(0.00002209) }
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        ht_float_t e_pos = HT_ERF(cases[i].x);
        ht_float_t ec_pos = HT_ERFC(cases[i].x);
        ht_float_t e_neg = HT_ERF(-cases[i].x);
        ht_float_t ec_neg = HT_ERFC(-cases[i].x);

        HT_ASSERT(HT_FABS(HT_SUB(e_pos, cases[i].erf_exp)) <= 3);
        HT_ASSERT(HT_FABS(HT_SUB(ec_pos, cases[i].erfc_exp)) <= 3);
        HT_ASSERT_EQ(e_neg, -e_pos);
        HT_ASSERT_EQ(HT_ADD(e_pos, ec_pos), HT_FIXED_ONE);
        HT_ASSERT_EQ(HT_ADD(e_neg, ec_neg), HT_FIXED_ONE);
    }
}

static void test_fixed_extreme_overflow_and_boundary_branches(void) {
    /* Extreme saturation, negative div/mul, exp/log/sqrt/erf corner cases */
    HT_ASSERT_EQ(ht_fixed_add(HT_INFINITY, HT_CONST(1.0)), HT_INFINITY);
    HT_ASSERT_EQ(ht_fixed_add(-HT_INFINITY, HT_CONST(-1.0)), -HT_INFINITY);
    HT_ASSERT_EQ(ht_fixed_sub(-HT_INFINITY, HT_CONST(1.0)), -HT_INFINITY);
    HT_ASSERT_EQ(ht_fixed_sub(HT_INFINITY, HT_CONST(-1.0)), HT_INFINITY);
    HT_ASSERT_EQ(ht_fixed_mul(HT_CONST(-20000.0), HT_CONST(20000.0)), -HT_INFINITY);
    HT_ASSERT_EQ(ht_fixed_mul(HT_CONST(20000.0), HT_CONST(20000.0)), HT_INFINITY);
    HT_ASSERT_EQ(ht_fixed_div(HT_CONST(-10000.0), HT_CONST(0.001)), -HT_INFINITY);
    HT_ASSERT_EQ(ht_fixed_div(HT_CONST(10000.0), HT_CONST(0.001)), HT_INFINITY);
    HT_ASSERT_EQ(ht_fixed_halve(-HT_FIXED_ONE), -(HT_FIXED_ONE / 2));
    HT_ASSERT(HT_ISNAN(ht_fixed_halve(HT_NAN_VAL)));
    HT_ASSERT_EQ(ht_fixed_from_int(100000), HT_INFINITY);
    HT_ASSERT_EQ(ht_fixed_from_int(-100000), -HT_INFINITY);
    HT_ASSERT_EQ(ht_fixed_from_uint(100000U), HT_INFINITY);
    HT_ASSERT_EQ(ht_fixed_to_int(HT_NAN_VAL), 0);

    /* Exact negative integer log2/exp boundaries */
    HT_ASSERT_EQ(HT_EXP(HT_CONST(-15.0)), 0);
    HT_ASSERT_EQ(HT_EXP(HT_CONST(15.0)), HT_INFINITY);
    HT_ASSERT_EQ(HT_EXP(HT_CONST(-11.0)), 1);
    HT_ASSERT(HT_LOG(HT_CONST(0.0001)) < HT_CONST(-8.0));
    HT_ASSERT(HT_SQRT(HT_INFINITY) > HT_CONST(180.0));
    HT_ASSERT_EQ(HT_ERF(HT_CONST(3.99)), HT_FIXED_ONE);
    HT_ASSERT_EQ(HT_ERF(HT_CONST(-3.99)), -HT_FIXED_ONE);
}

#endif /* HT_USE_FIXED_POINT */

void test_fixed_run(void) {
    HT_TEST_SUITE_BEGIN("Nivel 1 & 2: Aritmetica Abstrata e Motor Q16.16 Zero-Libm");
    HT_RUN_TEST(test_abstract_arithmetic_macros);
#ifdef HT_USE_FIXED_POINT
    HT_RUN_TEST(test_fixed_sentinels_and_saturation);
    HT_RUN_TEST(test_fixed_sqrt_accuracy);
    HT_RUN_TEST(test_fixed_log2_and_log_accuracy);
    HT_RUN_TEST(test_fixed_exp_accuracy);
    HT_RUN_TEST(test_fixed_erf_and_erfc_accuracy);
    HT_RUN_TEST(test_fixed_extreme_overflow_and_boundary_branches);
#endif
}

#ifdef HT_FIXED_STANDALONE_TEST
int main(void) {
    printf("==================================================\n");
    printf(" Standalone Zero-Libm Q16.16 Math Engine Test\n");
    printf("==================================================\n");
    test_fixed_run();
    HT_TEST_SUMMARY();
    return (g_tests_failed == 0) ? 0 : 1;
}
#endif
