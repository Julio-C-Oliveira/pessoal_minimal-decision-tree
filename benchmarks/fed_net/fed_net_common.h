/**
 * @file fed_net_common.h
 * @brief Shared POSIX TCP framing for HFED binary frames and deterministic Non-IID streams.
 *
 * Used by standalone multi-process binaries fed_net_server and fed_net_client
 * (Delivery 11.10 — Gate A: Real Network Transport Validation).
 */

#ifndef FED_NET_COMMON_H
#define FED_NET_COMMON_H

#define _POSIX_C_SOURCE 200809L

#include "ht_config.h"
#include "ht_tree.h"
#include "ht_fed_proto.h"

#include <arpa/inet.h>
#include <errno.h>
#include <math.h>
#include <netinet/in.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define FED_NET_WIRE_CAP  131072U
#define FED_NET_MAX_CLIENTS 16U

/* ========================================================================= */
/* --- Deterministic PRNG & Synthetic Non-IID Stream Generators ------------ */
/* ========================================================================= */

typedef struct {
    uint64_t state;
} fed_rng_t;

static inline uint32_t fed_rng_u32(fed_rng_t *rng) {
    uint64_t x = rng->state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng->state = x;
    return (uint32_t)((x * 2685821657736338717ULL) >> 32);
}

static inline ht_float_t fed_rng_uniform(fed_rng_t *rng, ht_float_t lo, ht_float_t hi) {
    double u = (double)fed_rng_u32(rng) / 4294967296.0;
    return (ht_float_t)((double)lo + u * ((double)hi - (double)lo));
}

/**
 * @brief 4-Class Quadrant generator (3 numeric features, 4 classes).
 *
 * Class is determined by quadrants of (x0, x1) around 5.0; x2 is uniform noise:
 *   class 0: x0 <= 5, x1 <= 5
 *   class 1: x0 <= 5, x1 >  5
 *   class 2: x0 >  5, x1 <= 5
 *   class 3: x0 >  5, x1 >  5
 *
 * If pure_non_iid is true, client k ONLY sees class ((k - 1) % 4).
 * Otherwise, client k sees 80% of class ((k - 1) % 4) and 20% uniform (still Non-IID).
 */
static inline void fed_gen_quadrant_sample(fed_rng_t *rng,
                                           uint16_t client_id,
                                           bool pure_non_iid,
                                           bool is_global_test,
                                           ht_value_t x[HT_MAX_FEATURES],
                                           uint8_t *y_out) {
    uint8_t target_cls;
    if (is_global_test) {
        target_cls = (uint8_t)(fed_rng_u32(rng) % 4U);
    } else if (pure_non_iid) {
        target_cls = (uint8_t)((client_id - 1U) % 4U);
    } else {
        if ((fed_rng_u32(rng) % 100U) < 80U) {
            target_cls = (uint8_t)((client_id - 1U) % 4U);
        } else {
            target_cls = (uint8_t)(fed_rng_u32(rng) % 4U);
        }
    }

    ht_float_t x0 = (target_cls >= 2U)
        ? fed_rng_uniform(rng, (ht_float_t)5.2, (ht_float_t)9.8)
        : fed_rng_uniform(rng, (ht_float_t)0.2, (ht_float_t)4.8);
    ht_float_t x1 = ((target_cls & 1U) != 0U)
        ? fed_rng_uniform(rng, (ht_float_t)5.2, (ht_float_t)9.8)
        : fed_rng_uniform(rng, (ht_float_t)0.2, (ht_float_t)4.8);
    ht_float_t x2 = fed_rng_uniform(rng, (ht_float_t)0.0, (ht_float_t)10.0);

    x[0].num = x0;
    x[1].num = x1;
    x[2].num = x2;
    *y_out = target_cls;
}

/**
 * @brief Non-IID Piecewise Regression generator (3 numeric features).
 *
 * Ground truth:
 *   x0 <= 5.0 -> y = 10.0 + 2.0 * x1
 *   x0 >  5.0 -> y = 60.0 + 2.0 * x1
 * Odd clients sample predominantly x0 in [0, 5.5], even clients in [4.5, 10.0].
 */
static inline void fed_gen_reg_sample(fed_rng_t *rng,
                                      uint16_t client_id,
                                      bool is_global_test,
                                      ht_value_t x[HT_MAX_FEATURES],
                                      ht_float_t *y_out) {
    ht_float_t x0;
    if (is_global_test) {
        x0 = fed_rng_uniform(rng, (ht_float_t)0.0, (ht_float_t)10.0);
    } else if ((client_id % 2U) == 1U) {
        x0 = ((fed_rng_u32(rng) % 100U) < 85U)
            ? fed_rng_uniform(rng, (ht_float_t)0.0, (ht_float_t)5.0)
            : fed_rng_uniform(rng, (ht_float_t)5.0, (ht_float_t)10.0);
    } else {
        x0 = ((fed_rng_u32(rng) % 100U) < 85U)
            ? fed_rng_uniform(rng, (ht_float_t)5.0, (ht_float_t)10.0)
            : fed_rng_uniform(rng, (ht_float_t)0.0, (ht_float_t)5.0);
    }
    ht_float_t x1 = fed_rng_uniform(rng, (ht_float_t)0.0, (ht_float_t)5.0);
    ht_float_t x2 = fed_rng_uniform(rng, (ht_float_t)0.0, (ht_float_t)10.0);
    ht_float_t noise = fed_rng_uniform(rng, (ht_float_t)-0.5, (ht_float_t)0.5);

    x[0].num = x0;
    x[1].num = x1;
    x[2].num = x2;
    *y_out = ((x0 <= (ht_float_t)5.0) ? (ht_float_t)10.0 : (ht_float_t)60.0)
             + (ht_float_t)2.0 * x1 + noise;
}

/* ========================================================================= */
/* --- POSIX TCP Raw HFED Framing Helpers (with Fragmentation Stress) ------ */
/* ========================================================================= */

static inline bool fed_tcp_send_all(int fd,
                                    const uint8_t *buf,
                                    size_t len,
                                    size_t max_chunk) {
    size_t sent = 0U;
    while (sent < len) {
        size_t want = len - sent;
        if (max_chunk > 0U && want > max_chunk) {
            want = max_chunk;
        }
        ssize_t n = send(fd, buf + sent, want, 0);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (n == 0) {
            return false;
        }
        sent += (size_t)n;
    }
    return true;
}

static inline bool fed_tcp_recv_exact(int fd, uint8_t *buf, size_t len) {
    size_t got = 0U;
    while (got < len) {
        ssize_t n = recv(fd, buf + got, len - got, 0);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (n == 0) {
            return false; /* EOF before frame complete */
        }
        got += (size_t)n;
    }
    return true;
}

/**
 * @brief Receives exactly one raw HFED binary frame from a TCP stream.
 *
 * Reads the 32-byte HFED header first, inspects magic and payload_len safely
 * via memcpy (zero unaligned access), then reads payload_len exact bytes.
 */
static inline bool fed_tcp_recv_hfed_frame(int fd,
                                           uint8_t *buf,
                                           size_t cap,
                                           size_t *out_len,
                                           ht_fed_header_t *out_hdr) {
    if (cap < sizeof(ht_fed_header_t)) {
        return false;
    }
    if (!fed_tcp_recv_exact(fd, buf, sizeof(ht_fed_header_t))) {
        return false;
    }
    ht_fed_header_t hdr;
    memcpy(&hdr, buf, sizeof(hdr));
    if (hdr.magic != HT_FED_MAGIC) {
        return false;
    }
    size_t total = sizeof(ht_fed_header_t) + (size_t)hdr.payload_len;
    if (total > cap) {
        return false;
    }
    if (hdr.payload_len > 0U) {
        if (!fed_tcp_recv_exact(fd, buf + sizeof(ht_fed_header_t), (size_t)hdr.payload_len)) {
            return false;
        }
    }
    if (out_len != NULL) {
        *out_len = total;
    }
    if (out_hdr != NULL) {
        *out_hdr = hdr;
    }
    return true;
}

static inline int fed_tcp_connect_retry(const char *host, uint16_t port, int max_retries) {
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
        return -1;
    }

    for (int attempt = 0; attempt < max_retries; ++attempt) {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) {
            return -1;
        }
        if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) == 0) {
            return fd;
        }
        close(fd);
        struct timespec ts;
        ts.tv_sec = 0;
        ts.tv_nsec = 25000000L; /* 25 ms */
        nanosleep(&ts, NULL);
    }
    return -1;
}

#endif /* FED_NET_COMMON_H */
