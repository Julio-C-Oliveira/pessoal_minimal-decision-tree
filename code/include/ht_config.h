/**
 * @file ht_config.h
 * @brief Core configuration, numerical precision types, math macros,
 *        and common status codes for the Embedded Hoeffding Tree (VFDT / HAT).
 */

#ifndef HT_CONFIG_H
#define HT_CONFIG_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <math.h>
#include <float.h>

#ifdef HT_USE_DOUBLE
typedef double ht_float_t;
#define HT_SQRT(x)      sqrt(x)
#define HT_LOG(x)       log(x)
#define HT_LOG2(x)      log2(x)
#define HT_EXP(x)       exp(x)
#define HT_ERF(x)       erf(x)
#define HT_ERFC(x)      erfc(x)
#define HT_FABS(x)      fabs(x)
#define HT_ISNAN(x)     isnan(x)
#define HT_ISFINITE(x)  isfinite(x)
#define HT_INFINITY     ((ht_float_t)INFINITY)
#define HT_EPSILON      ((ht_float_t)1e-12)
#define HT_FLOAT_TOL    ((ht_float_t)1e-12)
#else
typedef float ht_float_t;
#define HT_SQRT(x)      sqrtf(x)
#define HT_LOG(x)       logf(x)
#define HT_LOG2(x)      log2f(x)
#define HT_EXP(x)       expf(x)
#define HT_ERF(x)       erff(x)
#define HT_ERFC(x)      erfcf(x)
#define HT_FABS(x)      fabsf(x)
#define HT_ISNAN(x)     isnan(x)
#define HT_ISFINITE(x)  isfinite(x)
#define HT_INFINITY     ((ht_float_t)INFINITY)
#define HT_EPSILON      ((ht_float_t)1e-7f)
#define HT_FLOAT_TOL    ((ht_float_t)1e-5f)
#endif

#define HT_PI           ((ht_float_t)3.14159265358979323846)
#define HT_SQRT2        ((ht_float_t)1.41421356237309504880)

#ifndef HT_MAX_FEATURES
#define HT_MAX_FEATURES      16U
#endif

#ifndef HT_MAX_CLASSES
#define HT_MAX_CLASSES       4U
#endif

#ifndef HT_MAX_CATEGORIES
#define HT_MAX_CATEGORIES    5U
#endif

#ifndef HT_MAX_NODES
#define HT_MAX_NODES         4096U
#endif

#ifndef HT_MAX_ACTIVE_LEAVES
#define HT_MAX_ACTIVE_LEAVES 256U
#endif

#ifndef HT_GAUSSIAN_SPLITS
#define HT_GAUSSIAN_SPLITS   10U
#endif

#ifndef HT_REG_NUM_SLOTS
#define HT_REG_NUM_SLOTS     5U
#endif

#ifndef HT_ADWIN_MAX_BUCKETS
#define HT_ADWIN_MAX_BUCKETS 5U
#endif

#ifndef HT_ADWIN_LEVELS
#define HT_ADWIN_LEVELS      14U
#endif

#ifndef HT_MAX_ADWIN_DETECTORS
#define HT_MAX_ADWIN_DETECTORS 128U
#endif

#define HT_INVALID_IDX       ((uint16_t)0xFFFFU)
#define HT_NO_FEATURE        ((uint8_t)0xFFU)

/**
 * @brief Machine learning task type: Classification or Continuous Regression.
 */
typedef enum {
    HT_TASK_CLASSIFICATION = 0,
    HT_TASK_REGRESSION     = 1
} ht_task_type_t;

/**
 * @brief Feature type indicator: numeric (continuous) or categorical (nominal).
 */
typedef enum {
    HT_FEAT_NUMERIC     = 0,
    HT_FEAT_CATEGORICAL = 1
} ht_feature_type_t;

/**
 * @brief Unified feature value representation.
 *
 * Stores either a continuous float or a discrete categorical integer index.
 * Missing numeric values are marked as NaN (checked via HT_ISNAN).
 * Missing categorical values are marked as negative integers (e.g., -1).
 */
typedef union {
    ht_float_t num;
    int32_t    cat;
} ht_value_t;

/**
 * @brief Return status codes for Hoeffding Tree operations.
 */
typedef enum {
    HT_OK                =  0,
    HT_ERR_INVALID_ARG   = -1,
    HT_ERR_OUT_OF_NODES  = -2,
    HT_ERR_OUT_OF_STATS  = -3,
    HT_ERR_IO            = -4,
    HT_ERR_CORRUPT_IMAGE = -5
} ht_status_t;

/* Static assertions for fundamental data type sizes */
_Static_assert(sizeof(uint8_t) == 1, "uint8_t must be exactly 1 byte");
_Static_assert(sizeof(uint16_t) == 2, "uint16_t must be exactly 2 bytes");
_Static_assert(sizeof(uint32_t) == 4, "uint32_t must be exactly 4 bytes");
_Static_assert(sizeof(int32_t) == 4, "int32_t must be exactly 4 bytes");

#ifndef HT_USE_DOUBLE
_Static_assert(sizeof(ht_float_t) == 4, "ht_float_t must be 4 bytes in single-precision mode");
_Static_assert(sizeof(ht_value_t) == 4, "ht_value_t must be 4 bytes in single-precision mode");
#else
_Static_assert(sizeof(ht_float_t) == 8, "ht_float_t must be 8 bytes in double-precision mode");
_Static_assert(sizeof(ht_value_t) == 8, "ht_value_t must be 8 bytes in double-precision mode");
#endif

#endif /* HT_CONFIG_H */
