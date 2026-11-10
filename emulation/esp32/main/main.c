/**
 * @file main.c
 * @brief ESP32 FreeRTOS Firmware for Embedded Hoeffding Tree Emulation (QEMU).
 *
 * Implements:
 * - Deterministic execution of Tier 8 Universal libht (C11) on ESP32 Xtensa Dual-Core.
 * - Hardware cycle timing using esp_cpu_get_cycle_count() @ 240 MHz.
 * - Test-Then-Train prequential evaluation with Stop-and-Wait serial handshake over UART0 / TCP Socket.
 * - Telemetry reporting: CPU cycles, classification accuracy / regression MAE/RMSE/R2,
 *   SRAM memory footprint, heap delta (proving 0.0 KB malloc in runtime), and stack high-water mark.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/uart.h"
#include "esp_system.h"
#include "esp_heap_caps.h"

#if __has_include("esp_cpu.h")
#include "esp_cpu.h"
#define GET_CPU_CYCLES() esp_cpu_get_cycle_count()
#else
#include "soc/cpu.h"
#define GET_CPU_CYCLES() esp_cpu_get_cycle_count()
#endif

#include "ht_tree.h"
#include "ht_drift.h"

#define UART_PORT_NUM      UART_NUM_0
#define UART_BAUD_RATE     115200
#define UART_RX_BUF_SIZE   4096
#define MAX_LINE_BUF       2048

/* ========================================================================== */
/* Static Memory Layout (.bss) - Strictly zero malloc in runtime              */
/* ========================================================================== */
static ht_tree_t       tree_instance;
static ht_adwin_pool_t adwin_pool;
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

static double train_sum_abs_err = 0.0;
static double train_sum_sq_err  = 0.0;
static double train_sum_y       = 0.0;
static double train_sum_y_sq    = 0.0;

static double test_sum_abs_err  = 0.0;
static double test_sum_sq_err   = 0.0;
static double test_sum_y        = 0.0;
static double test_sum_y_sq     = 0.0;

static uint64_t total_pred_cycles  = 0;
static uint32_t max_pred_cycles    = 0;
static uint32_t min_pred_cycles    = UINT32_MAX;
static uint32_t pred_count         = 0;

static uint64_t total_learn_cycles = 0;
static uint32_t max_learn_cycles   = 0;
static uint32_t min_learn_cycles   = UINT32_MAX;
static uint32_t learn_count        = 0;

static size_t heap_at_init = 0;
static size_t heap_after_init = 0;

/* ========================================================================== */
/* UART Helpers                                                               */
/* ========================================================================== */
static uint8_t uart_rx_chunk[1024];
static int     uart_rx_chunk_len = 0;
static int     uart_rx_chunk_pos = 0;

static void init_uart(void) {
    uart_config_t uart_config = {
        .baud_rate  = UART_BAUD_RATE,
        .data_bits  = UART_DATA_8_BITS,
        .parity     = UART_PARITY_DISABLE,
        .stop_bits  = UART_STOP_BITS_1,
        .flow_ctrl  = UART_HW_FLOWCTRL_DISABLE,
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 0, 0)
        .source_clk = UART_SCLK_DEFAULT,
#endif
    };
    uart_param_config(UART_PORT_NUM, &uart_config);
    uart_driver_install(UART_PORT_NUM, UART_RX_BUF_SIZE * 2, 0, 0, NULL, 0);
    uart_rx_chunk_len = 0;
    uart_rx_chunk_pos = 0;
}

static int read_line(char *buf, size_t max_len) {
    size_t idx = 0;
    while (idx < max_len - 1) {
        if (uart_rx_chunk_pos >= uart_rx_chunk_len) {
            uart_rx_chunk_pos = 0;
            uart_rx_chunk_len = 0;

            /* Read 1 byte to block efficiently without timeout penalties */
            int r = uart_read_bytes(UART_PORT_NUM, uart_rx_chunk, 1, portMAX_DELAY);
            if (r <= 0) {
                continue;
            }
            uart_rx_chunk_len = 1;

            /* Read all remaining buffered bytes immediately */
            size_t buffered = 0;
            if (uart_get_buffered_data_len(UART_PORT_NUM, &buffered) == ESP_OK && buffered > 0) {
                size_t to_read = buffered;
                if (to_read > sizeof(uart_rx_chunk) - 1) {
                    to_read = sizeof(uart_rx_chunk) - 1;
                }
                int extra = uart_read_bytes(UART_PORT_NUM, uart_rx_chunk + 1, to_read, 0);
                if (extra > 0) {
                    uart_rx_chunk_len += extra;
                }
            }
        }
        uint8_t ch = uart_rx_chunk[uart_rx_chunk_pos++];
        if (ch == '\n') {
            buf[idx] = '\0';
            return (int)idx;
        } else if (ch != '\r') {
            buf[idx++] = (char)ch;
        }
    }
    buf[idx] = '\0';
    return (int)idx;
}

static void send_line(const char *str) {
    uart_write_bytes(UART_PORT_NUM, str, strlen(str));
}

/* ========================================================================== */
/* Protocol Handlers                                                          */
/* ========================================================================== */
static void handle_ping(void) {
    snprintf(tx_line_buf, sizeof(tx_line_buf),
             "PONG,version=%s,sizeof_tree=%u,sizeof_adwin_pool=%u\n",
             ht_version(),
             (unsigned int)sizeof(tree_instance),
             (unsigned int)sizeof(adwin_pool));
    send_line(tx_line_buf);
}

static void handle_init(char *param_str) {
    heap_at_init = esp_get_free_heap_size();

    current_task = HT_TASK_CLASSIFICATION;
    current_n_features = 10;
    current_n_classes = 2;
    ht_criterion_type_t criterion = HT_CRIT_INFO_GAIN;
    ht_leaf_pred_t leaf_pred = HT_LEAF_NBA;
    bool leaf_set = false;
    uint16_t grace_period = 200;
    double delta_val = 1e-7;
    double tau_val = 0.05;
    double learning_rate = 0.01;
    double l2_reg = 1e-4;
    double decay_rate = 0.95;
    double min_samples_split = 5.0;
    bool enable_hat = false;

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
            delta_val = atof(token + 6);
        } else if (strncmp(token, "tau=", 4) == 0) {
            tau_val = atof(token + 4);
        } else if (strncmp(token, "lr=", 3) == 0) {
            learning_rate = atof(token + 3);
        } else if (strncmp(token, "l2=", 3) == 0) {
            l2_reg = atof(token + 3);
        } else if (strncmp(token, "decay=", 6) == 0) {
            decay_rate = atof(token + 6);
        } else if (strncmp(token, "hat=", 4) == 0) {
            enable_hat = (atoi(token + 4) != 0);
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
    params.delta = (ht_float_t)delta_val;
    params.tau = (ht_float_t)tau_val;

    ht_status_t status;
    if (enable_hat) {
        ht_adwin_pool_init(&adwin_pool);
        status = ht_tree_init_hat(&tree_instance, &params, &adwin_pool);
    } else {
        status = ht_tree_init(&tree_instance, &params);
    }

    heap_after_init = esp_get_free_heap_size();

    /* Reset telemetry */
    train_samples = 0;
    train_correct = 0;
    test_samples  = 0;
    test_correct  = 0;
    train_sum_abs_err = 0.0;
    train_sum_sq_err  = 0.0;
    train_sum_y       = 0.0;
    train_sum_y_sq    = 0.0;
    test_sum_abs_err  = 0.0;
    test_sum_sq_err   = 0.0;
    test_sum_y        = 0.0;
    test_sum_y_sq     = 0.0;

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
             "OK_INIT,status=%d,sizeof_tree=%u,heap_before=%u,heap_after=%u\n",
             (int)status,
             (unsigned int)sizeof(tree_instance),
             (unsigned int)heap_at_init,
             (unsigned int)heap_after_init);
    send_line(tx_line_buf);
}

static void parse_features_and_target(char *line, ht_float_t *out_y_reg, uint8_t *out_y_cls) {
    /* Line format: <y>,<x0>,<x1>,... */
    char *token = strtok(line, ",");
    if (!token) return;

    if (current_task == HT_TASK_REGRESSION) {
        *out_y_reg = (ht_float_t)atof(token);
    } else {
        *out_y_cls = (uint8_t)atoi(token);
    }

    for (uint8_t f = 0; f < current_n_features; ++f) {
        token = strtok(NULL, ",");
        if (!token) {
            x_sample[f].num = 0.0f;
            continue;
        }
        if (current_feat_types[f] == HT_FEAT_CATEGORICAL) {
            x_sample[f].cat = (int32_t)atoi(token);
        } else {
            x_sample[f].num = (ht_float_t)atof(token);
        }
    }
}

static void handle_train(char *payload) {
    if (!session_active) {
        send_line("ERR,session_inactive\n");
        return;
    }

    ht_float_t y_reg = 0.0f;
    uint8_t   y_cls = 0;
    parse_features_and_target(payload, &y_reg, &y_cls);

    /* 1. Predict (Test-Then-Train) */
    uint32_t c0 = GET_CPU_CYCLES();
    uint8_t pred_cls = 0;
    ht_float_t pred_reg = 0.0f;

    if (current_task == HT_TASK_REGRESSION) {
        pred_reg = ht_tree_predict_reg(&tree_instance, x_sample);
    } else {
        pred_cls = ht_tree_predict(&tree_instance, x_sample);
    }
    uint32_t c1 = GET_CPU_CYCLES();
    uint32_t pred_cycles = c1 - c0;

    if (pred_cycles > max_pred_cycles) max_pred_cycles = pred_cycles;
    if (pred_cycles < min_pred_cycles) min_pred_cycles = pred_cycles;
    total_pred_cycles += pred_cycles;
    pred_count++;

    /* Update prequential statistics */
    train_samples++;
    if (current_task == HT_TASK_REGRESSION) {
        double err = (double)(pred_reg - y_reg);
        train_sum_abs_err += fabs(err);
        train_sum_sq_err  += err * err;
        train_sum_y       += (double)y_reg;
        train_sum_y_sq    += (double)y_reg * (double)y_reg;
    } else {
        if (pred_cls == y_cls) {
            train_correct++;
        }
    }

    /* 2. Learn Step */
    uint32_t c2 = GET_CPU_CYCLES();
    if (current_task == HT_TASK_REGRESSION) {
        ht_tree_learn_one_reg(&tree_instance, x_sample, y_reg, 1.0f);
    } else {
        ht_tree_learn_one(&tree_instance, x_sample, y_cls, 1.0f);
    }
    uint32_t c3 = GET_CPU_CYCLES();
    uint32_t learn_cycles = c3 - c2;

    if (learn_cycles > max_learn_cycles) max_learn_cycles = learn_cycles;
    if (learn_cycles < min_learn_cycles) min_learn_cycles = learn_cycles;
    total_learn_cycles += learn_cycles;
    learn_count++;

    if (current_task == HT_TASK_REGRESSION) {
        snprintf(tx_line_buf, sizeof(tx_line_buf), "ACK,%.4f,%u,%u\n",
                 (double)pred_reg, (unsigned int)pred_cycles, (unsigned int)learn_cycles);
    } else {
        snprintf(tx_line_buf, sizeof(tx_line_buf), "ACK,%u,%u,%u\n",
                 (unsigned int)pred_cls, (unsigned int)pred_cycles, (unsigned int)learn_cycles);
    }
    send_line(tx_line_buf);
}

static void handle_test(char *payload) {
    if (!session_active) {
        send_line("ERR,session_inactive\n");
        return;
    }

    ht_float_t y_reg = 0.0f;
    uint8_t   y_cls = 0;
    parse_features_and_target(payload, &y_reg, &y_cls);

    uint32_t c0 = GET_CPU_CYCLES();
    uint8_t pred_cls = 0;
    ht_float_t pred_reg = 0.0f;

    if (current_task == HT_TASK_REGRESSION) {
        pred_reg = ht_tree_predict_reg(&tree_instance, x_sample);
    } else {
        pred_cls = ht_tree_predict(&tree_instance, x_sample);
    }
    uint32_t c1 = GET_CPU_CYCLES();
    uint32_t pred_cycles = c1 - c0;

    if (pred_cycles > max_pred_cycles) max_pred_cycles = pred_cycles;
    if (pred_cycles < min_pred_cycles) min_pred_cycles = pred_cycles;
    total_pred_cycles += pred_cycles;
    pred_count++;

    test_samples++;
    if (current_task == HT_TASK_REGRESSION) {
        double err = (double)(pred_reg - y_reg);
        test_sum_abs_err += fabs(err);
        test_sum_sq_err  += err * err;
        test_sum_y       += (double)y_reg;
        test_sum_y_sq    += (double)y_reg * (double)y_reg;
    } else {
        if (pred_cls == y_cls) {
            test_correct++;
        }
    }

    if (current_task == HT_TASK_REGRESSION) {
        snprintf(tx_line_buf, sizeof(tx_line_buf), "ACK,%.4f,%u,0\n",
                 (double)pred_reg, (unsigned int)pred_cycles);
    } else {
        snprintf(tx_line_buf, sizeof(tx_line_buf), "ACK,%u,%u,0\n",
                 (unsigned int)pred_cls, (unsigned int)pred_cycles);
    }
    send_line(tx_line_buf);
}

static void handle_end(void) {
    if (!session_active) {
        send_line("ERR,session_inactive\n");
        return;
    }

    size_t heap_final = esp_get_free_heap_size();
    long heap_delta = (long)heap_after_init - (long)heap_final;
    UBaseType_t stack_hwm = uxTaskGetStackHighWaterMark(NULL);

    double avg_pred_cyc = (pred_count > 0) ? ((double)total_pred_cycles / pred_count) : 0.0;
    double avg_learn_cyc = (learn_count > 0) ? ((double)total_learn_cycles / learn_count) : 0.0;
    double avg_total_us = (avg_pred_cyc + avg_learn_cyc) / 240.0; /* 240 MHz clock */

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

    uint16_t total_nodes = tree_instance.memory.nodes_allocated;
    uint16_t active_leaves = tree_instance.n_active_leaves;
    size_t pool_bytes = ht_memory_bytes_used(&tree_instance.memory);

    snprintf(tx_line_buf, sizeof(tx_line_buf),
             "RESULT,{\"task\":\"%s\",\"n_train\":%u,\"n_test\":%u,"
             "\"prequential_acc\":%.2f,\"holdout_acc\":%.2f,"
             "\"prequential_mae\":%.4f,\"prequential_rmse\":%.4f,"
             "\"holdout_mae\":%.4f,\"holdout_rmse\":%.4f,\"holdout_r2\":%.4f,"
             "\"avg_pred_cycles\":%.1f,\"max_pred_cycles\":%u,"
             "\"avg_learn_cycles\":%.1f,\"max_learn_cycles\":%u,"
             "\"avg_total_us_240mhz\":%.2f,"
             "\"n_nodes\":%u,\"n_active_leaves\":%u,\"max_depth\":%u,"
             "\"sizeof_tree\":%u,\"pool_bytes_used\":%u,"
             "\"heap_delta_bytes\":%ld,\"stack_hwm_bytes\":%u}\n",
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
             (unsigned int)sizeof(tree_instance), (unsigned int)pool_bytes,
             heap_delta, (unsigned int)(stack_hwm * sizeof(StackType_t)));

    send_line(tx_line_buf);
    session_active = false;
}

/* ========================================================================== */
/* Main Task Loop                                                             */
/* ========================================================================== */
void app_main(void) {
    init_uart();

    /* Send ready announcement banner */
    snprintf(tx_line_buf, sizeof(tx_line_buf),
             "\n=== ESP32 Hoeffding Tree Firmware Ready (v%s, Tier 8 Universal, %u MHz) ===\n",
             ht_version(),
             CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ);
    send_line(tx_line_buf);

    while (1) {
        int len = read_line(rx_line_buf, sizeof(rx_line_buf));
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
            send_line(tx_line_buf);
        }
    }
}
