/**
 * @file test_main.c
 * @brief Main entry point for the Embedded Hoeffding Tree test runner.
 *
 * Sequentially executes all Level 1 (Unit), Level 2 (Integration),
 * and Level 3 (Macro / E2E) test suites.
 */

#include "test_harness.h"
#include <string.h>

/* Define global test counters */
int g_tests_run = 0;
int g_tests_failed = 0;
int g_assertions_run = 0;
bool g_current_test_failed = false;

/* --- Milestone 1.1: Config, Types, and Harness Unit Tests --- */

static void test_config_constants(void) {
    HT_ASSERT_EQ(HT_MAX_FEATURES, 16U);
    HT_ASSERT_EQ(HT_MAX_CLASSES, 4U);
    HT_ASSERT_EQ(HT_MAX_CATEGORIES, 5U);
    HT_ASSERT_EQ(HT_MAX_NODES, 4096U);
    HT_ASSERT_EQ(HT_MAX_ACTIVE_LEAVES, 256U);
    HT_ASSERT_EQ(HT_GAUSSIAN_SPLITS, 10U);

    HT_ASSERT_EQ(HT_ADWIN_MAX_BUCKETS, 5U);
#if HT_ADWIN_LEVELS == 14U
    HT_ASSERT_EQ(HT_ADWIN_LEVELS, 14U);
#else
    HT_ASSERT(HT_ADWIN_LEVELS >= 4U && HT_ADWIN_LEVELS <= 32U);
#endif
#if HT_MAX_ADWIN_DETECTORS == 128U
    HT_ASSERT_EQ(HT_MAX_ADWIN_DETECTORS, 128U);
#else
    HT_ASSERT(HT_MAX_ADWIN_DETECTORS > 0U);
#endif

    HT_ASSERT_EQ(HT_INVALID_IDX, 0xFFFFU);
    HT_ASSERT_EQ(HT_NO_FEATURE, 0xFFU);

    HT_ASSERT_EQ(HT_FEAT_NUMERIC, 0);
    HT_ASSERT_EQ(HT_FEAT_CATEGORICAL, 1);

    HT_ASSERT_EQ(HT_OK, 0);
    HT_ASSERT_EQ(HT_ERR_INVALID_ARG, -1);
    HT_ASSERT_EQ(HT_ERR_OUT_OF_NODES, -2);
    HT_ASSERT_EQ(HT_ERR_OUT_OF_STATS, -3);
    HT_ASSERT_EQ(HT_ERR_IO, -4);
    HT_ASSERT_EQ(HT_ERR_CORRUPT_IMAGE, -5);

    /* Versioning assertions (SemVer) */
    HT_ASSERT_EQ(HT_VERSION_MAJOR, 0U);
    HT_ASSERT_EQ(HT_VERSION_MINOR, 6U);
    HT_ASSERT_EQ(HT_VERSION_PATCH, 0U);
    HT_ASSERT(HT_VERSION_CHECK(0, 6, 0));
    HT_ASSERT(!HT_VERSION_CHECK(0, 6, 1));
    HT_ASSERT(!HT_VERSION_CHECK(1, 0, 0));
    HT_ASSERT_EQ(strcmp(HT_VERSION_STRING, "0.6.0"), 0);
    HT_ASSERT_EQ(strcmp(ht_version(), "0.6.0"), 0);
}

static void test_config_types_and_sizes(void) {
    HT_ASSERT_EQ(sizeof(uint8_t), 1);
    HT_ASSERT_EQ(sizeof(uint16_t), 2);
    HT_ASSERT_EQ(sizeof(uint32_t), 4);
    HT_ASSERT_EQ(sizeof(int32_t), 4);

#ifndef HT_USE_DOUBLE
    HT_ASSERT_EQ(sizeof(ht_float_t), 4);
    HT_ASSERT_EQ(sizeof(ht_value_t), 4);
#else
    HT_ASSERT_EQ(sizeof(ht_float_t), 8);
    HT_ASSERT_EQ(sizeof(ht_value_t), 8);
#endif
}

static void test_math_macros_basic(void) {
    HT_ASSERT_FLOAT_EQ(HT_SQRT(HT_CONST(4.0)), HT_CONST(2.0));
    HT_ASSERT_FLOAT_EQ(HT_SQRT(HT_CONST(0.0)), HT_CONST(0.0));
    HT_ASSERT_FLOAT_EQ(HT_LOG2(HT_CONST(8.0)), HT_CONST(3.0));
    HT_ASSERT_FLOAT_EQ(HT_LOG2(HT_CONST(1.0)), HT_CONST(0.0));
    HT_ASSERT_FLOAT_EQ(HT_EXP(HT_CONST(0.0)), HT_CONST(1.0));
    HT_ASSERT_FLOAT_EQ(HT_FABS(HT_CONST(-42.5)), HT_CONST(42.5));
    HT_ASSERT_FLOAT_EQ(HT_FABS(HT_CONST(42.5)), HT_CONST(42.5));

    /* ln(e^1) = 1.0 */
    ht_float_t e1 = HT_EXP(HT_CONST(1.0));
    HT_ASSERT_FLOAT_EQ(HT_LOG(e1), HT_CONST(1.0));

    /* Check fundamental constants */
    HT_ASSERT_FLOAT_NEAR(HT_PI, HT_CONST(3.141592653589793), HT_FLOAT_TOL);
    HT_ASSERT_FLOAT_NEAR(HT_SQRT2, HT_CONST(1.414213562373095), HT_FLOAT_TOL);
}

static void test_math_macros_edge_cases(void) {
    ht_float_t zero = HT_CONST(0.0);
    ht_float_t nan_val = HT_DIV(zero, zero);

    HT_ASSERT_NAN(nan_val);
    HT_ASSERT(HT_ISNAN(nan_val));
    HT_ASSERT(!HT_ISNAN(HT_CONST(1.0)));

    HT_ASSERT(HT_ISFINITE(HT_CONST(1.0)));
    HT_ASSERT(!HT_ISFINITE(HT_INFINITY));
    HT_ASSERT(!HT_ISFINITE(-HT_INFINITY));
    HT_ASSERT(!HT_ISFINITE(nan_val));

    HT_ASSERT_FLOAT_EQ(HT_ERF(HT_CONST(0.0)), HT_CONST(0.0));
    HT_ASSERT_FLOAT_NEAR(HT_ERF(HT_CONST(1.0)), HT_CONST(0.8427007929497149), HT_CONST(1e-3));
}

static void test_value_union(void) {
    ht_value_t v;

    v.num = HT_CONST(123.456);
    HT_ASSERT_FLOAT_NEAR(v.num, HT_CONST(123.456), HT_CONST(1e-3));

    v.cat = 42;
    HT_ASSERT_EQ(v.cat, 42);

    v.cat = -1;
    HT_ASSERT_EQ(v.cat, -1);
}

static void test_harness_assertions(void) {
    HT_ASSERT(true);
    HT_ASSERT_MSG(true, "truth value");
    HT_ASSERT_EQ(100, 100);
    HT_ASSERT_NEQ(100, 200);
    HT_ASSERT_FLOAT_EQ(HT_CONST(5.0), HT_CONST(5.0));
    HT_ASSERT_FLOAT_NEAR(HT_CONST(1.0), HT_CONST(1.0001), HT_CONST(1e-3));

    ht_float_t nan_val = HT_DIV(HT_CONST(0.0), HT_CONST(0.0));
    HT_ASSERT_NAN(nan_val);
    HT_ASSERT_NOT_NAN(HT_CONST(3.14));

    HT_ASSERT_NULL(NULL);
    int dummy = 42;
    HT_ASSERT_NOT_NULL(&dummy);
}

void test_config_run(void) {
    HT_TEST_SUITE_BEGIN("Nivel 1: Config, Tipos e Test Harness");
    HT_RUN_TEST(test_config_constants);
    HT_RUN_TEST(test_config_types_and_sizes);
    HT_RUN_TEST(test_math_macros_basic);
    HT_RUN_TEST(test_math_macros_edge_cases);
    HT_RUN_TEST(test_value_union);
    HT_RUN_TEST(test_harness_assertions);
}

int main(void) {
    printf("==================================================\n");
    printf(" Hoeffding Tree Test Suite (VFDT / HAT in C11)\n");
#ifdef HT_USE_FIXED_POINT
    printf(" Precision Mode: Q16.16 fixed-point (32-bit zero-libm)\n");
#elif defined(HT_USE_DOUBLE)
    printf(" Precision Mode: double (64-bit double precision)\n");
#else
    printf(" Precision Mode: float (32-bit single precision)\n");
#endif
    printf("==================================================\n");

    /* Nível 1: Módulos e Tipos Básicos */
    test_config_run();

    /* Nível 1 & 2: Motor Matemático Q16.16 e Operadores Abstratos */
    test_fixed_run();

    /* Nível 1 & 2: Slab / Pool Allocator */
    test_memory_run();

    /* Nível 1 & 2: Observadores de Atributos */
    test_observer_run();

    /* Nível 1 & 2: Critérios de Impureza e Hoeffding Bound */
    test_criterion_run();

    /* Nível 1 & 2: Árvore VFDT Completa */
    test_tree_run();

    /* Nível 1 & 2: Detector ADWIN com Histograma Exponencial Fixo */
    test_drift_run();

    /* Nível 1 & 2: Micro-Pipelines de Pré-Processamento (libht_prep) */
    test_prep_run();

    /* Nível 1 & 2: Fundação Compartilhada de Ensembles (ht_compact_tree & ht_voting) */
    test_compact_tree_run();

    /* Nível 1, 2 & 3: Aprendizado Federado Modo 1 (Fed-Forest) */
    test_fed_forest_run();

    /* Nível 1, 2 & 3: Aprendizado Federado Modo 2 (Fed-VFDT) */
    test_fed_vfdt_run();

    /* Nível 3: Fluxo Ponta a Ponta Acumulado */
    test_e2e_flow_run();

    /* Relatório Final */
    HT_TEST_SUMMARY();

    return (g_tests_failed == 0) ? 0 : 1;
}
