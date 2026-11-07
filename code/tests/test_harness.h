/**
 * @file test_harness.h
 * @brief Lightweight, zero-dependency C11 test harness for Hoeffding Tree.
 */

#ifndef TEST_HARNESS_H
#define TEST_HARNESS_H

#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <inttypes.h>
#include "ht_config.h"

/* Global test execution state (instantiated in test_main.c) */
extern int g_tests_run;
extern int g_tests_failed;
extern int g_assertions_run;
extern bool g_current_test_failed;

#define HT_TEST_SUITE_BEGIN(name) \
    printf("\n==================================================\n" \
           "=== SUITE: %s\n" \
           "==================================================\n", name)

#define HT_ASSERT(cond) do { \
    g_assertions_run++; \
    if (!(cond)) { \
        printf("\n    [FAIL] %s:%d: Assertion failed: %s\n   ", __FILE__, __LINE__, #cond); \
        g_current_test_failed = true; \
        return; \
    } \
} while (0)

#define HT_ASSERT_MSG(cond, msg) do { \
    g_assertions_run++; \
    if (!(cond)) { \
        printf("\n    [FAIL] %s:%d: Assertion failed: %s (%s)\n   ", __FILE__, __LINE__, #cond, msg); \
        g_current_test_failed = true; \
        return; \
    } \
} while (0)

#define HT_ASSERT_EQ(a, b) do { \
    g_assertions_run++; \
    int64_t _val_a = (int64_t)(a); \
    int64_t _val_b = (int64_t)(b); \
    if (_val_a != _val_b) { \
        printf("\n    [FAIL] %s:%d: %s == %s (expected %" PRId64 ", got %" PRId64 ")\n   ", \
               __FILE__, __LINE__, #a, #b, _val_b, _val_a); \
        g_current_test_failed = true; \
        return; \
    } \
} while (0)

#define HT_ASSERT_NEQ(a, b) do { \
    g_assertions_run++; \
    int64_t _val_a = (int64_t)(a); \
    int64_t _val_b = (int64_t)(b); \
    if (_val_a == _val_b) { \
        printf("\n    [FAIL] %s:%d: %s != %s (both are %" PRId64 ")\n   ", \
               __FILE__, __LINE__, #a, #b, _val_a); \
        g_current_test_failed = true; \
        return; \
    } \
} while (0)

#define HT_ASSERT_FLOAT_NEAR(a, b, tol) do { \
    g_assertions_run++; \
    ht_float_t _fa = (ht_float_t)(a); \
    ht_float_t _fb = (ht_float_t)(b); \
    if (HT_ISNAN(_fa) || HT_ISNAN(_fb)) { \
        printf("\n    [FAIL] %s:%d: %s ~= %s (unexpected NaN: %g vs %g)\n   ", \
               __FILE__, __LINE__, #a, #b, (double)_fa, (double)_fb); \
        g_current_test_failed = true; \
        return; \
    } \
    if (_fa != _fb) { \
        ht_float_t _diff = HT_FABS(_fa - _fb); \
        if (_diff > (ht_float_t)(tol)) { \
            printf("\n    [FAIL] %s:%d: %s ~= %s (tol %e): got %g vs %g (diff %e)\n   ", \
                   __FILE__, __LINE__, #a, #b, (double)(tol), (double)_fa, (double)_fb, (double)_diff); \
            g_current_test_failed = true; \
            return; \
        } \
    } \
} while (0)

#define HT_ASSERT_FLOAT_EQ(a, b) HT_ASSERT_FLOAT_NEAR(a, b, HT_FLOAT_TOL)

#define HT_ASSERT_NAN(a) do { \
    g_assertions_run++; \
    ht_float_t _val = (ht_float_t)(a); \
    if (!HT_ISNAN(_val)) { \
        printf("\n    [FAIL] %s:%d: Expected %s to be NaN (got %g)\n   ", \
               __FILE__, __LINE__, #a, (double)_val); \
        g_current_test_failed = true; \
        return; \
    } \
} while (0)

#define HT_ASSERT_NOT_NAN(a) do { \
    g_assertions_run++; \
    ht_float_t _val = (ht_float_t)(a); \
    if (HT_ISNAN(_val)) { \
        printf("\n    [FAIL] %s:%d: Expected %s not to be NaN\n   ", \
               __FILE__, __LINE__, #a); \
        g_current_test_failed = true; \
        return; \
    } \
} while (0)

#define HT_ASSERT_NULL(p) do { \
    g_assertions_run++; \
    if ((p) != NULL) { \
        printf("\n    [FAIL] %s:%d: Expected %s to be NULL\n   ", __FILE__, __LINE__, #p); \
        g_current_test_failed = true; \
        return; \
    } \
} while (0)

#define HT_ASSERT_NOT_NULL(p) do { \
    g_assertions_run++; \
    if ((p) == NULL) { \
        printf("\n    [FAIL] %s:%d: Expected %s not to be NULL\n   ", __FILE__, __LINE__, #p); \
        g_current_test_failed = true; \
        return; \
    } \
} while (0)

#define HT_RUN_TEST(test_func) do { \
    g_tests_run++; \
    g_current_test_failed = false; \
    printf("  [TEST] %-45s ", #test_func); \
    test_func(); \
    if (g_current_test_failed) { \
        g_tests_failed++; \
        printf("[FAILED]\n"); \
    } else { \
        printf("[PASSED]\n"); \
    } \
} while (0)

#define HT_TEST_SUMMARY() do { \
    printf("\n==================================================\n"); \
    printf(" Test Execution Summary:\n"); \
    printf("   Total tests run:       %d\n", g_tests_run); \
    printf("   Tests passed:          %d\n", g_tests_run - g_tests_failed); \
    printf("   Tests failed:          %d\n", g_tests_failed); \
    printf("   Assertions evaluated:  %d\n", g_assertions_run); \
    if (g_tests_failed == 0) { \
        printf(" STATUS: ALL TESTS PASSED (100%% GREEN)\n"); \
    } else { \
        printf(" STATUS: %d TEST(S) FAILED\n", g_tests_failed); \
    } \
    printf("==================================================\n"); \
} while (0)

/* Test suite declarations */
void test_config_run(void);
void test_e2e_flow_run(void);

#endif /* TEST_HARNESS_H */
