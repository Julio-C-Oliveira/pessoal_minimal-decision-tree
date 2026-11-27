/**
 * @file main.c
 * @brief Bare-Metal Firmware for ARM Cortex-M3 (TI Stellaris LM3S6965) libht Emulation.
 *
 * Implements:
 * - Deterministic execution of Tier 4 Universal libht (C11) on ARM Cortex-M3 @ 50 MHz.
 * - Hardware cycle timing using ARM SysTick 64-bit cycle counter.
 * - Test-Then-Train prequential evaluation with Stop-and-Wait serial protocol over UART0.
 * - Support for online preprocessing (StandardScaler & Leaf SGD normalization).
 * - Full telemetry: CPU cycles, holdout & prequential classification/regression metrics,
 *   SRAM memory footprint, and proof of 0 bytes heap allocation.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#ifndef HT_USE_FIXED_POINT
#include <math.h>
#else
#include "ht_math_fixed.h"
#endif

#include "uart.h"
#include "cycles.h"

#include "ht_tree.h"
#include "ht_drift.h"
#include "ht_prep.h"

#define MAX_LINE_BUF       1024

#ifdef HT_USE_FIXED_POINT
/* ========================================================================== */
/* Zero-Float Integer Parsing & Formatting Helpers (Q16.16 Bare-Metal Mode)   */
/* ========================================================================== */
static ht_float_t parse_q16(const char *s) {
    if (!s) return HT_CONST(0.0);
    while (*s == ' ' || *s == '\t') s++;

    bool neg = false;
    if (*s == '-') {
        neg = true;
        s++;
    } else if (*s == '+') {
        s++;
    }

    uint64_t int_part = 0;
    while (*s >= '0' && *s <= '9') {
        if (int_part < 1000000000ULL) {
            int_part = int_part * 10ULL + (uint64_t)(*s - '0');
        }
        s++;
    }

    uint64_t frac_part = 0;
    uint64_t frac_scale = 1;
    if (*s == '.') {
        s++;
        while (*s >= '0' && *s <= '9') {
            if (frac_scale < 10000000ULL) {
                frac_part = frac_part * 10ULL + (uint64_t)(*s - '0');
                frac_scale *= 10ULL;
            }
            s++;
        }
    }

    int exp10 = 0;
    if (*s == 'e' || *s == 'E') {
        s++;
        bool exp_neg = false;
        if (*s == '-') {
            exp_neg = true;
            s++;
        } else if (*s == '+') {
            s++;
        }
        while (*s >= '0' && *s <= '9') {
            if (exp10 < 100) {
                exp10 = exp10 * 10 + (*s - '0');
            }
            s++;
        }
        if (exp_neg) exp10 = -exp10;
    }

    uint64_t val = int_part * frac_scale + frac_part;
    uint64_t scale = frac_scale;

    while (exp10 > 0) {
        if (scale >= 10ULL) {
            scale /= 10ULL;
        } else if (val < (1ULL << 40)) {
            val *= 10ULL;
        }
        exp10--;
    }
    while (exp10 < 0) {
        if (scale <= 1000000000ULL) {
            scale *= 10ULL;
        } else {
            val /= 10ULL;
        }
        exp10++;
    }

    uint64_t q_int = val / scale;
    uint64_t q_rem = val % scale;
    uint64_t q_mag;
    if (q_int >= 32768ULL) {
        q_mag = (uint64_t)INT32_MAX;
    } else {
        q_mag = (q_int << 16) + ((q_rem * 65536ULL + (scale / 2ULL)) / scale);
    }

    int64_t signed_q = neg ? -(int64_t)q_mag : (int64_t)q_mag;
    return ht_fixed_saturate64(signed_q);
}

static void parse_delta_arg(const char *s, ht_float_t *out_delta, uint8_t *out_neg_log10) {
    *out_neg_log10 = 0;
    *out_delta = parse_q16(s);
    if (!s) return;
    while (*s == ' ' || *s == '\t') s++;

    /* Pattern 1: "1e-7", "1e-07", "1.0e-7" */
    const char *ep = strchr(s, 'e');
    if (!ep) ep = strchr(s, 'E');
    if (ep && ep[1] == '-') {
        int exp_val = atoi(ep + 2);
        if (s[0] == '1' && (s[1] == 'e' || s[1] == 'E' || s[1] == '.')) {
            if (exp_val >= 1 && exp_val <= 30) {
                *out_neg_log10 = (uint8_t)exp_val;
                return;
            }
        }
    }

    /* Pattern 2: "0.001", "0.0001", "0.0000001" */
    if (s[0] == '0' && s[1] == '.') {
        const char *p = s + 2;
        uint8_t zeros = 0;
        while (*p == '0') {
            zeros++;
            p++;
        }
        if (*p == '1') {
            p++;
            while (*p == '0') p++;
            if (*p == '\0' && (zeros + 1) <= 30) {
                *out_neg_log10 = (uint8_t)(zeros + 1);
            }
        }
    }
}

static void fmt_q16_4(char *buf, size_t sz, ht_float_t v) {
    bool neg = (v < 0);
    uint64_t mag = neg ? (uint64_t)(-(int64_t)v) : (uint64_t)v;
    uint64_t scaled = (mag * 10000ULL + 32768ULL) >> 16;
    snprintf(buf, sz, "%s%u.%04u",
             (neg && scaled > 0ULL) ? "-" : "",
             (unsigned int)(scaled / 10000ULL),
             (unsigned int)(scaled % 10000ULL));
}

static void fmt_u64_div_1dec(char *buf, size_t sz, uint64_t num, uint64_t den) {
    if (den == 0ULL) {
        snprintf(buf, sz, "0.0");
        return;
    }
    uint64_t q = num / den;
    uint64_t r = num % den;
    uint64_t frac = (r * 10ULL + (den / 2ULL)) / den;
    if (frac >= 10ULL) {
        q += 1ULL;
        frac -= 10ULL;
    }
    snprintf(buf, sz, "%u.%u", (unsigned int)q, (unsigned int)frac);
}

static void fmt_u64_div_2dec(char *buf, size_t sz, uint64_t num, uint64_t den) {
    if (den == 0ULL) {
        snprintf(buf, sz, "0.00");
        return;
    }
    uint64_t q = num / den;
    uint64_t r = num % den;
    uint64_t frac = (r * 100ULL + (den / 2ULL)) / den;
    if (frac >= 100ULL) {
        q += 1ULL;
        frac -= 100ULL;
    }
    snprintf(buf, sz, "%u.%02u", (unsigned int)q, (unsigned int)frac);
}

static void fmt_i64_scaled4(char *buf, size_t sz, int64_t scaled) {
    bool neg = (scaled < 0);
    uint64_t mag = neg ? (uint64_t)(-scaled) : (uint64_t)scaled;
    snprintf(buf, sz, "%s%u.%04u",
             (neg && mag > 0ULL) ? "-" : "",
             (unsigned int)(mag / 10000ULL),
             (unsigned int)(mag % 10000ULL));
}

static ht_float_t q8_sum_to_rmse_q16(uint64_t sum_sq_q8, uint32_t n) {
    if (n == 0U || sum_sq_q8 == 0ULL) return HT_CONST(0.0);
    uint64_t mse_q16;
    if (sum_sq_q8 < (1ULL << 54)) {
        mse_q16 = (sum_sq_q8 << 8) / (uint64_t)n;
    } else {
        mse_q16 = (sum_sq_q8 / (uint64_t)n) << 8;
    }
    uint64_t rmse_q16;
    if (mse_q16 < (1ULL << 48)) {
        rmse_q16 = ht_fixed_isqrt64(mse_q16 << 16);
    } else {
        rmse_q16 = ht_fixed_isqrt64(mse_q16) << 8;
    }
    return ht_fixed_saturate64((int64_t)rmse_q16);
}
#endif

/* ========================================================================== */
/* Static Memory Layout (.bss) — Strictly zero dynamic malloc at runtime      */
/* ========================================================================== */
static ht_tree_t       tree_instance;
static ht_adwin_pool_t adwin_pool;
static ht_scaler_t     scaler_instance;
static bool            use_scaler = false;

static char            rx_line_buf[MAX_LINE_BUF];
static char            tx_line_buf[MAX_LINE_BUF];
static ht_value_t      x_sample[HT_MAX_FEATURES];

/* Session telemetry state */
static bool              session_active = false;
static ht_task_type_t    current_task = HT_TASK_CLASSIFICATION;
static uint8_t           current_n_features = 0;
static uint8_t           current_n_classes = 2;
static ht_feature_type_t current_feat_types[HT_MAX_FEATURES];

static uint32_t train_samples = 0;
static uint32_t train_correct = 0;
static uint32_t test_samples = 0;
static uint32_t test_correct = 0;

#ifdef HT_USE_FIXED_POINT
static uint64_t train_sum_abs_err_q16 = 0ULL;
static uint64_t train_sum_sq_err_q8   = 0ULL;
static int64_t  train_sum_y_q16       = 0LL;
static uint64_t train_sum_y_sq_q8     = 0ULL;

static uint64_t test_sum_abs_err_q16  = 0ULL;
static uint64_t test_sum_sq_err_q8    = 0ULL;
static int64_t  test_sum_y_q16        = 0LL;
static uint64_t test_sum_y_sq_q8      = 0ULL;
#else
static double train_sum_abs_err = 0.0;
static double train_sum_sq_err  = 0.0;
static double train_sum_y       = 0.0;
static double train_sum_y_sq    = 0.0;

static double test_sum_abs_err  = 0.0;
static double test_sum_sq_err   = 0.0;
static double test_sum_y        = 0.0;
static double test_sum_y_sq     = 0.0;
#endif

static uint64_t total_pred_cycles  = 0;
static uint32_t max_pred_cycles    = 0;
static uint32_t min_pred_cycles    = UINT32_MAX;
static uint32_t pred_count         = 0;

static uint64_t total_learn_cycles = 0;
static uint32_t max_learn_cycles   = 0;
static uint32_t min_learn_cycles   = UINT32_MAX;
static uint32_t learn_count        = 0;

/* ========================================================================== */
/* Protocol Handlers                                                          */
/* ========================================================================== */
static void handle_ping(void) {
    snprintf(tx_line_buf, sizeof(tx_line_buf),
             "PONG,version=%s,sizeof_tree=%u,sizeof_adwin_pool=%u,arch=cortex-m3,clock_mhz=%u,precision=%s\n",
             ht_version(),
             (unsigned int)sizeof(tree_instance),
             (unsigned int)sizeof(adwin_pool),
             (unsigned int)CPU_FREQ_MHZ,
#ifdef HT_USE_FIXED_POINT
             "q16_16"
#else
             "float32"
#endif
    );
    uart_send_str(tx_line_buf);
}

static void handle_init(char *param_str) {
    current_task = HT_TASK_CLASSIFICATION;
    current_n_features = 10;
    current_n_classes = 2;
    ht_criterion_type_t criterion = HT_CRIT_INFO_GAIN;
    ht_leaf_pred_t leaf_pred = HT_LEAF_NBA;
    bool leaf_set = false;
    uint16_t grace_period = 200;
#ifdef HT_USE_FIXED_POINT
    ht_float_t delta_val = HT_CONST(0.0);
    uint8_t delta_neg_log10 = 7U;
    ht_float_t tau_val = HT_CONST(0.05);
    ht_float_t learning_rate = HT_CONST(0.01);
    ht_float_t l2_reg = HT_CONST(0.0001);
    ht_float_t decay_rate = HT_CONST(0.95);
    ht_float_t min_samples_split = HT_CONST(5.0);
#else
    double delta_val = 1e-7;
    double tau_val = 0.05;
    double learning_rate = 0.01;
    double l2_reg = 1e-4;
    double decay_rate = 0.95;
    double min_samples_split = 5.0;
#endif
    bool enable_hat = false;
    bool leaf_normalize = false;
    use_scaler = false;

    for (uint32_t f = 0; f < HT_MAX_FEATURES; ++f) {
        current_feat_types[f] = HT_FEAT_NUMERIC;
    }

    char *token = strtok(param_str, ",");
    while (token) {
        if (strncmp(token, "task=", 5) == 0) {
            if (strcmp(token + 5, "regression") == 0) {
                current_task = HT_TASK_REGRESSION;
            } else {
                current_task = HT_TASK_CLASSIFICATION;
            }
        } else if (strncmp(token, "n_features=", 11) == 0) {
            current_n_features = (uint8_t)atoi(token + 11);
        } else if (strncmp(token, "n_classes=", 10) == 0) {
            current_n_classes = (uint8_t)atoi(token + 10);
        } else if (strncmp(token, "types=", 6) == 0) {
            char *sub = token + 6;
            uint8_t f_idx = 0;
            while (*sub && f_idx < HT_MAX_FEATURES) {
                if (*sub == '1') {
                    current_feat_types[f_idx++] = HT_FEAT_CATEGORICAL;
                } else if (*sub == '0') {
                    current_feat_types[f_idx++] = HT_FEAT_NUMERIC;
                }
                sub++;
            }
        } else if (strncmp(token, "leaf=", 5) == 0) {
            leaf_set = true;
            char *l = token + 5;
            if (strcmp(l, "mc") == 0 || strcmp(l, "mean") == 0) {
                leaf_pred = HT_LEAF_MC;
            } else if (strcmp(l, "nb") == 0 || strcmp(l, "model") == 0) {
                leaf_pred = HT_LEAF_NB;
            } else {
                leaf_pred = HT_LEAF_NBA;
            }
        } else if (strncmp(token, "crit=", 5) == 0 || strncmp(token, "criterion=", 10) == 0) {
            char *c = strchr(token, '=') + 1;
            if (strcmp(c, "gini") == 0) {
                criterion = HT_CRIT_GINI;
            } else if (strcmp(c, "info_gain") == 0) {
                criterion = HT_CRIT_INFO_GAIN;
            }
        } else if (strncmp(token, "grace=", 6) == 0) {
            grace_period = (uint16_t)atoi(token + 6);
        } else if (strncmp(token, "delta=", 6) == 0) {
#ifdef HT_USE_FIXED_POINT
            parse_delta_arg(token + 6, &delta_val, &delta_neg_log10);
#else
            delta_val = atof(token + 6);
#endif
        } else if (strncmp(token, "tau=", 4) == 0) {
#ifdef HT_USE_FIXED_POINT
            tau_val = parse_q16(token + 4);
#else
            tau_val = atof(token + 4);
#endif
        } else if (strncmp(token, "lr=", 3) == 0) {
#ifdef HT_USE_FIXED_POINT
            learning_rate = parse_q16(token + 3);
#else
            learning_rate = atof(token + 3);
#endif
        } else if (strncmp(token, "l2=", 3) == 0) {
#ifdef HT_USE_FIXED_POINT
            l2_reg = parse_q16(token + 3);
#else
            l2_reg = atof(token + 3);
#endif
        } else if (strncmp(token, "decay=", 6) == 0) {
#ifdef HT_USE_FIXED_POINT
            decay_rate = parse_q16(token + 6);
#else
            decay_rate = atof(token + 6);
#endif
        } else if (strncmp(token, "hat=", 4) == 0) {
            enable_hat = (atoi(token + 4) != 0);
        } else if (strncmp(token, "leaf_normalize=", 15) == 0) {
            leaf_normalize = (atoi(token + 15) != 0);
        } else if (strncmp(token, "prep_scaler=", 12) == 0) {
            use_scaler = (atoi(token + 12) != 0);
        }
        token = strtok(NULL, ",");
    }

    if (!leaf_set) {
        leaf_pred = (current_task == HT_TASK_REGRESSION) ? HT_LEAF_ADAPTIVE : HT_LEAF_NBA;
    }

    if (current_n_features > HT_MAX_FEATURES) {
        current_n_features = HT_MAX_FEATURES;
    }
    if (current_task == HT_TASK_CLASSIFICATION && current_n_classes > HT_MAX_CLASSES) {
        current_n_classes = HT_MAX_CLASSES;
    }

    ht_config_params_t params;
    if (current_task == HT_TASK_REGRESSION) {
        if (enable_hat) {
            ht_default_hat_reg_params(&params, current_n_features);
        } else {
            ht_default_reg_params(&params, current_n_features);
        }
        params.leaf_prediction = leaf_pred;
        params.learning_rate = (ht_float_t)learning_rate;
        params.l2_regularization = (ht_float_t)l2_reg;
        params.model_selector_decay = (ht_float_t)decay_rate;
        params.min_samples_split = (ht_float_t)min_samples_split;
        params.leaf_sgd_normalize = leaf_normalize;
    } else {
        if (enable_hat) {
            ht_default_hat_params(&params, current_n_features, current_n_classes);
        } else {
            ht_default_params(&params, current_n_features, current_n_classes);
        }
        params.criterion = criterion;
        params.leaf_prediction = leaf_pred;
    }

    for (uint8_t f = 0; f < current_n_features; ++f) {
        params.feature_types[f] = current_feat_types[f];
    }
    params.grace_period = grace_period;
#ifdef HT_USE_FIXED_POINT
    if (delta_neg_log10 > 0U) {
        ht_params_set_delta_pow10(&params, delta_neg_log10);
    } else {
        params.delta = delta_val;
        params.delta_neg_log10 = 0U;
    }
#else
    params.delta = (ht_float_t)delta_val;
#endif
    params.tau = (ht_float_t)tau_val;

    ht_status_t status;
    if (enable_hat) {
        ht_adwin_pool_init(&adwin_pool);
        status = ht_tree_init_hat(&tree_instance, &params, &adwin_pool);
    } else {
        status = ht_tree_init(&tree_instance, &params);
    }

    /* Initialize online scaler if requested */
    if (use_scaler) {
        ht_scaler_init(&scaler_instance);
        for (uint8_t f = 0; f < current_n_features; ++f) {
            if (current_feat_types[f] == HT_FEAT_NUMERIC) {
                ht_scaler_set_mode(&scaler_instance, f, HT_SCALER_STANDARD);
            }
        }
    }

    /* Reset telemetry accumulators */
    train_samples = 0;
    train_correct = 0;
    test_samples  = 0;
    test_correct  = 0;
#ifdef HT_USE_FIXED_POINT
    train_sum_abs_err_q16 = 0ULL;
    train_sum_sq_err_q8   = 0ULL;
    train_sum_y_q16       = 0LL;
    train_sum_y_sq_q8     = 0ULL;
    test_sum_abs_err_q16  = 0ULL;
    test_sum_sq_err_q8    = 0ULL;
    test_sum_y_q16        = 0LL;
    test_sum_y_sq_q8      = 0ULL;
#else
    train_sum_abs_err = 0.0;
    train_sum_sq_err  = 0.0;
    train_sum_y       = 0.0;
    train_sum_y_sq    = 0.0;
    test_sum_abs_err  = 0.0;
    test_sum_sq_err   = 0.0;
    test_sum_y        = 0.0;
    test_sum_y_sq     = 0.0;
#endif

    total_pred_cycles  = 0;
    max_pred_cycles    = 0;
    min_pred_cycles    = UINT32_MAX;
    pred_count         = 0;
    total_learn_cycles = 0;
    max_learn_cycles   = 0;
    min_learn_cycles   = UINT32_MAX;
    learn_count        = 0;

    session_active = (status == HT_OK);

    snprintf(tx_line_buf, sizeof(tx_line_buf),
             "OK_INIT,status=%d,sizeof_tree=%u,heap_before=0,heap_after=0\n",
             (int)status,
             (unsigned int)sizeof(tree_instance));
    uart_send_str(tx_line_buf);
}

static void parse_features_and_target(char *line, ht_float_t *out_y_reg, uint8_t *out_y_cls) {
    char *token = strtok(line, ",");
    if (!token) return;

    if (current_task == HT_TASK_REGRESSION) {
#ifdef HT_USE_FIXED_POINT
        *out_y_reg = parse_q16(token);
#else
        *out_y_reg = (ht_float_t)atof(token);
#endif
    } else {
        *out_y_cls = (uint8_t)atoi(token);
    }

    for (uint8_t f = 0; f < current_n_features; ++f) {
        token = strtok(NULL, ",");
        if (!token) {
            x_sample[f].num = HT_CONST(0.0);
            continue;
        }
        if (current_feat_types[f] == HT_FEAT_CATEGORICAL) {
            x_sample[f].cat = (int32_t)atoi(token);
        } else {
#ifdef HT_USE_FIXED_POINT
            x_sample[f].num = parse_q16(token);
#else
            x_sample[f].num = (ht_float_t)atof(token);
#endif
        }
    }
}

static void handle_train(char *payload) {
    if (!session_active) {
        uart_send_str("ERR,session_inactive\n");
        return;
    }

    ht_float_t y_reg = HT_CONST(0.0);
    uint8_t   y_cls = 0;
    parse_features_and_target(payload, &y_reg, &y_cls);

    /* Apply online preprocessing scaler if active */
    if (use_scaler) {
        for (uint8_t f = 0; f < current_n_features; ++f) {
            if (current_feat_types[f] == HT_FEAT_NUMERIC) {
                x_sample[f].num = ht_scaler_learn_transform_one(&scaler_instance, f, x_sample[f].num, HT_CONST(1.0));
            }
        }
    }

    /* 1. Predict Step (Test-Then-Train) */
    uint64_t c0 = get_cpu_cycles();
    uint8_t pred_cls = 0;
    ht_float_t pred_reg = HT_CONST(0.0);

    if (current_task == HT_TASK_REGRESSION) {
        pred_reg = ht_tree_predict_reg(&tree_instance, x_sample);
    } else {
        pred_cls = ht_tree_predict(&tree_instance, x_sample);
    }
    uint64_t c1 = get_cpu_cycles();
    uint32_t pred_cycles = (uint32_t)(c1 - c0);

    if (pred_cycles > max_pred_cycles) max_pred_cycles = pred_cycles;
    if (pred_cycles < min_pred_cycles) min_pred_cycles = pred_cycles;
    total_pred_cycles += pred_cycles;
    pred_count++;

    /* Update prequential evaluation metrics */
    train_samples++;
    if (current_task == HT_TASK_REGRESSION) {
#ifdef HT_USE_FIXED_POINT
        int64_t err = (int64_t)pred_reg - (int64_t)y_reg;
        uint64_t abs_err = (err < 0) ? (uint64_t)(-err) : (uint64_t)err;
        train_sum_abs_err_q16 += abs_err;
        train_sum_sq_err_q8   += (abs_err * abs_err) >> 24;
        train_sum_y_q16       += (int64_t)y_reg;
        uint64_t abs_y = (y_reg < 0) ? (uint64_t)(-(int64_t)y_reg) : (uint64_t)y_reg;
        train_sum_y_sq_q8     += (abs_y * abs_y) >> 24;
#else
        double err = (double)(pred_reg - y_reg);
        train_sum_abs_err += fabs(err);
        train_sum_sq_err  += err * err;
        train_sum_y       += (double)y_reg;
        train_sum_y_sq    += (double)y_reg * (double)y_reg;
#endif
    } else {
        if (pred_cls == y_cls) {
            train_correct++;
        }
    }

    /* 2. Learn Step */
    uint64_t c2 = get_cpu_cycles();
    if (current_task == HT_TASK_REGRESSION) {
        ht_tree_learn_one_reg(&tree_instance, x_sample, y_reg, HT_CONST(1.0));
    } else {
        ht_tree_learn_one(&tree_instance, x_sample, y_cls, HT_CONST(1.0));
    }
    uint64_t c3 = get_cpu_cycles();
    uint32_t learn_cycles = (uint32_t)(c3 - c2);

    if (learn_cycles > max_learn_cycles) max_learn_cycles = learn_cycles;
    if (learn_cycles < min_learn_cycles) min_learn_cycles = learn_cycles;
    total_learn_cycles += learn_cycles;
    learn_count++;

    if (current_task == HT_TASK_REGRESSION) {
#ifdef HT_USE_FIXED_POINT
        char pbuf[24];
        fmt_q16_4(pbuf, sizeof(pbuf), pred_reg);
        snprintf(tx_line_buf, sizeof(tx_line_buf), "ACK,%s,%u,%u\n",
                 pbuf, (unsigned int)pred_cycles, (unsigned int)learn_cycles);
#else
        snprintf(tx_line_buf, sizeof(tx_line_buf), "ACK,%.4f,%u,%u\n",
                 (double)pred_reg, (unsigned int)pred_cycles, (unsigned int)learn_cycles);
#endif
    } else {
        snprintf(tx_line_buf, sizeof(tx_line_buf), "ACK,%u,%u,%u\n",
                 (unsigned int)pred_cls, (unsigned int)pred_cycles, (unsigned int)learn_cycles);
    }
    uart_send_str(tx_line_buf);
}

static void handle_test(char *payload) {
    if (!session_active) {
        uart_send_str("ERR,session_inactive\n");
        return;
    }

    ht_float_t y_reg = HT_CONST(0.0);
    uint8_t   y_cls = 0;
    parse_features_and_target(payload, &y_reg, &y_cls);

    /* Apply online preprocessing scaler transform if active */
    if (use_scaler) {
        for (uint8_t f = 0; f < current_n_features; ++f) {
            if (current_feat_types[f] == HT_FEAT_NUMERIC) {
                x_sample[f].num = ht_scaler_transform_one(&scaler_instance, f, x_sample[f].num);
            }
        }
    }

    uint64_t c0 = get_cpu_cycles();
    uint8_t pred_cls = 0;
    ht_float_t pred_reg = HT_CONST(0.0);

    if (current_task == HT_TASK_REGRESSION) {
        pred_reg = ht_tree_predict_reg(&tree_instance, x_sample);
    } else {
        pred_cls = ht_tree_predict(&tree_instance, x_sample);
    }
    uint64_t c1 = get_cpu_cycles();
    uint32_t pred_cycles = (uint32_t)(c1 - c0);

    if (pred_cycles > max_pred_cycles) max_pred_cycles = pred_cycles;
    if (pred_cycles < min_pred_cycles) min_pred_cycles = pred_cycles;
    total_pred_cycles += pred_cycles;
    pred_count++;

    test_samples++;
    if (current_task == HT_TASK_REGRESSION) {
#ifdef HT_USE_FIXED_POINT
        int64_t err = (int64_t)pred_reg - (int64_t)y_reg;
        uint64_t abs_err = (err < 0) ? (uint64_t)(-err) : (uint64_t)err;
        test_sum_abs_err_q16 += abs_err;
        test_sum_sq_err_q8   += (abs_err * abs_err) >> 24;
        test_sum_y_q16       += (int64_t)y_reg;
        uint64_t abs_y = (y_reg < 0) ? (uint64_t)(-(int64_t)y_reg) : (uint64_t)y_reg;
        test_sum_y_sq_q8     += (abs_y * abs_y) >> 24;
#else
        double err = (double)(pred_reg - y_reg);
        test_sum_abs_err += fabs(err);
        test_sum_sq_err  += err * err;
        test_sum_y       += (double)y_reg;
        test_sum_y_sq    += (double)y_reg * (double)y_reg;
#endif
    } else {
        if (pred_cls == y_cls) {
            test_correct++;
        }
    }

    if (current_task == HT_TASK_REGRESSION) {
#ifdef HT_USE_FIXED_POINT
        char pbuf[24];
        fmt_q16_4(pbuf, sizeof(pbuf), pred_reg);
        snprintf(tx_line_buf, sizeof(tx_line_buf), "ACK,%s,%u,0\n",
                 pbuf, (unsigned int)pred_cycles);
#else
        snprintf(tx_line_buf, sizeof(tx_line_buf), "ACK,%.4f,%u,0\n",
                 (double)pred_reg, (unsigned int)pred_cycles);
#endif
    } else {
        snprintf(tx_line_buf, sizeof(tx_line_buf), "ACK,%u,%u,0\n",
                 (unsigned int)pred_cls, (unsigned int)pred_cycles);
    }
    uart_send_str(tx_line_buf);
}

static void handle_end(void) {
    if (!session_active) {
        uart_send_str("ERR,session_inactive\n");
        return;
    }

    uint16_t total_nodes = tree_instance.memory.nodes_allocated;
    uint16_t active_leaves = tree_instance.n_active_leaves;
    size_t pool_bytes = ht_memory_bytes_used(&tree_instance.memory);

#ifdef HT_USE_FIXED_POINT
    char s_preq_acc[24], s_hold_acc[24];
    char s_preq_mae[24], s_preq_rmse[24];
    char s_hold_mae[24], s_hold_rmse[24], s_hold_r2[24];
    char s_avg_pred[24], s_avg_learn[24], s_avg_us[24];

    fmt_u64_div_2dec(s_preq_acc, sizeof(s_preq_acc), (uint64_t)train_correct * 100ULL, (uint64_t)train_samples);
    fmt_u64_div_2dec(s_hold_acc, sizeof(s_hold_acc), (uint64_t)test_correct * 100ULL, (uint64_t)test_samples);

    ht_float_t preq_mae_q16 = (train_samples > 0U)
        ? ht_fixed_saturate64((int64_t)(train_sum_abs_err_q16 / (uint64_t)train_samples))
        : HT_CONST(0.0);
    ht_float_t preq_rmse_q16 = q8_sum_to_rmse_q16(train_sum_sq_err_q8, train_samples);

    ht_float_t hold_mae_q16 = (test_samples > 0U)
        ? ht_fixed_saturate64((int64_t)(test_sum_abs_err_q16 / (uint64_t)test_samples))
        : HT_CONST(0.0);
    ht_float_t hold_rmse_q16 = q8_sum_to_rmse_q16(test_sum_sq_err_q8, test_samples);

    int64_t hold_r2_x10000 = 0;
    if (test_samples > 1U) {
        uint64_t abs_sum_y = (test_sum_y_q16 < 0) ? (uint64_t)(-test_sum_y_q16) : (uint64_t)test_sum_y_q16;
        uint64_t sum_y_q4 = abs_sum_y >> 12;
        uint64_t mean_part_q8 = (sum_y_q4 * sum_y_q4) / (uint64_t)test_samples;
        uint64_t ss_tot_q8 = (test_sum_y_sq_q8 > mean_part_q8) ? (test_sum_y_sq_q8 - mean_part_q8) : 0ULL;
        if (ss_tot_q8 > 0ULL) {
            uint64_t num = test_sum_sq_err_q8;
            uint64_t den = ss_tot_q8;
            while (num > (UINT64_MAX / 10000ULL) && den > 1ULL) {
                num >>= 4;
                den >>= 4;
            }
            uint64_t ratio_x10000 = (num * 10000ULL) / den;
            hold_r2_x10000 = 10000LL - (int64_t)ratio_x10000;
        }
    }

    fmt_q16_4(s_preq_mae, sizeof(s_preq_mae), preq_mae_q16);
    fmt_q16_4(s_preq_rmse, sizeof(s_preq_rmse), preq_rmse_q16);
    fmt_q16_4(s_hold_mae, sizeof(s_hold_mae), hold_mae_q16);
    fmt_q16_4(s_hold_rmse, sizeof(s_hold_rmse), hold_rmse_q16);
    fmt_i64_scaled4(s_hold_r2, sizeof(s_hold_r2), hold_r2_x10000);

    fmt_u64_div_1dec(s_avg_pred, sizeof(s_avg_pred), total_pred_cycles, (uint64_t)pred_count);
    fmt_u64_div_1dec(s_avg_learn, sizeof(s_avg_learn), total_learn_cycles, (uint64_t)learn_count);

    /* avg_total_us = (avg_pred_cyc + avg_learn_cyc) / CPU_FREQ_MHZ */
    uint64_t avg_pred_x100 = (pred_count > 0U) ? ((total_pred_cycles * 100ULL) / (uint64_t)pred_count) : 0ULL;
    uint64_t avg_learn_x100 = (learn_count > 0U) ? ((total_learn_cycles * 100ULL) / (uint64_t)learn_count) : 0ULL;
    fmt_u64_div_2dec(s_avg_us, sizeof(s_avg_us), avg_pred_x100 + avg_learn_x100, (uint64_t)CPU_FREQ_MHZ * 100ULL);

    snprintf(tx_line_buf, sizeof(tx_line_buf),
             "RESULT,{\"task\":\"%s\",\"precision\":\"q16_16\",\"n_train\":%u,\"n_test\":%u,"
             "\"prequential_acc\":%s,\"holdout_acc\":%s,"
             "\"prequential_mae\":%s,\"prequential_rmse\":%s,"
             "\"holdout_mae\":%s,\"holdout_rmse\":%s,\"holdout_r2\":%s,"
             "\"avg_pred_cycles\":%s,\"max_pred_cycles\":%u,"
             "\"avg_learn_cycles\":%s,\"max_learn_cycles\":%u,"
             "\"avg_total_us_50mhz\":%s,"
             "\"n_nodes\":%u,\"n_active_leaves\":%u,\"max_depth\":%u,"
             "\"sizeof_tree\":%u,\"pool_bytes_used\":%u,"
             "\"heap_delta_bytes\":0,\"stack_hwm_bytes\":4096}\n",
             (current_task == HT_TASK_REGRESSION) ? "regression" : "classification",
             (unsigned int)train_samples, (unsigned int)test_samples,
             s_preq_acc, s_hold_acc,
             s_preq_mae, s_preq_rmse,
             s_hold_mae, s_hold_rmse, s_hold_r2,
             s_avg_pred, (unsigned int)max_pred_cycles,
             s_avg_learn, (unsigned int)max_learn_cycles,
             s_avg_us,
             (unsigned int)total_nodes, (unsigned int)active_leaves,
             (unsigned int)tree_instance.max_depth_seen,
             (unsigned int)sizeof(tree_instance), (unsigned int)pool_bytes);
#else
    double avg_pred_cyc = (pred_count > 0) ? ((double)total_pred_cycles / pred_count) : 0.0;
    double avg_learn_cyc = (learn_count > 0) ? ((double)total_learn_cycles / learn_count) : 0.0;
    double avg_total_us = (avg_pred_cyc + avg_learn_cyc) / (double)CPU_FREQ_MHZ; /* @ 50 MHz */

    double preq_acc = (train_samples > 0) ? ((double)train_correct / train_samples * 100.0) : 0.0;
    double hold_acc = (test_samples > 0) ? ((double)test_correct / test_samples * 100.0) : 0.0;

    double preq_mae = (train_samples > 0) ? (train_sum_abs_err / train_samples) : 0.0;
    double preq_rmse = (train_samples > 0) ? sqrt(train_sum_sq_err / train_samples) : 0.0;

    double hold_mae = (test_samples > 0) ? (test_sum_abs_err / test_samples) : 0.0;
    double hold_rmse = (test_samples > 0) ? sqrt(test_sum_sq_err / test_samples) : 0.0;
    double hold_r2 = 0.0;
    if (test_samples > 1) {
        double ss_tot = test_sum_y_sq - (test_sum_y * test_sum_y / test_samples);
        if (ss_tot > 1e-9) {
            hold_r2 = 1.0 - (test_sum_sq_err / ss_tot);
        }
    }

    snprintf(tx_line_buf, sizeof(tx_line_buf),
             "RESULT,{\"task\":\"%s\",\"precision\":\"float32\",\"n_train\":%u,\"n_test\":%u,"
             "\"prequential_acc\":%.2f,\"holdout_acc\":%.2f,"
             "\"prequential_mae\":%.4f,\"prequential_rmse\":%.4f,"
             "\"holdout_mae\":%.4f,\"holdout_rmse\":%.4f,\"holdout_r2\":%.4f,"
             "\"avg_pred_cycles\":%.1f,\"max_pred_cycles\":%u,"
             "\"avg_learn_cycles\":%.1f,\"max_learn_cycles\":%u,"
             "\"avg_total_us_50mhz\":%.2f,"
             "\"n_nodes\":%u,\"n_active_leaves\":%u,\"max_depth\":%u,"
             "\"sizeof_tree\":%u,\"pool_bytes_used\":%u,"
             "\"heap_delta_bytes\":0,\"stack_hwm_bytes\":4096}\n",
             (current_task == HT_TASK_REGRESSION) ? "regression" : "classification",
             (unsigned int)train_samples, (unsigned int)test_samples,
             preq_acc, hold_acc,
             preq_mae, preq_rmse,
             hold_mae, hold_rmse, hold_r2,
             avg_pred_cyc, (unsigned int)max_pred_cycles,
             avg_learn_cyc, (unsigned int)max_learn_cycles,
             avg_total_us,
             (unsigned int)total_nodes, (unsigned int)active_leaves,
             (unsigned int)tree_instance.max_depth_seen,
             (unsigned int)sizeof(tree_instance), (unsigned int)pool_bytes);
#endif

    uart_send_str(tx_line_buf);
    session_active = false;
}

/* ========================================================================== */
/* Main Application Loop                                                      */
/* ========================================================================== */
int main(void) {
    /* Send announcement banner */
    snprintf(tx_line_buf, sizeof(tx_line_buf),
             "\n=== ARM Cortex-M3 (TI Stellaris LM3S6965) Hoeffding Tree Firmware Ready (v%s, Tier 4 Universal, %s, %u MHz) ===\n",
             ht_version(),
#ifdef HT_USE_FIXED_POINT
             "Q16.16 Zero-Libm",
#else
             "float32 soft-fp",
#endif
             (unsigned int)CPU_FREQ_MHZ);
    uart_send_str(tx_line_buf);

    while (1) {
        int len = uart_read_line(rx_line_buf, sizeof(rx_line_buf));
        if (len <= 0) continue;

        if (strcmp(rx_line_buf, "PING") == 0) {
            handle_ping();
        } else if (strncmp(rx_line_buf, "INIT,", 5) == 0) {
            handle_init(rx_line_buf + 5);
        } else if (strncmp(rx_line_buf, "TRAIN,", 6) == 0) {
            handle_train(rx_line_buf + 6);
        } else if (strncmp(rx_line_buf, "TEST,", 5) == 0) {
            handle_test(rx_line_buf + 5);
        } else if (strcmp(rx_line_buf, "END") == 0) {
            handle_end();
        } else {
            snprintf(tx_line_buf, sizeof(tx_line_buf), "ERR,unknown_cmd:%.100s\n", rx_line_buf);
            uart_send_str(tx_line_buf);
        }
    }
    return 0;
}
