/**
 * @file ht_fed_proto.h
 * @brief Transport-agnostic binary framing, versioning, and CRC32 for Federated kansho.
 *
 * Shared wire layer for Mode 1 (Fed-Forest) and Mode 2 (Fed-VFDT).
 * Zero dynamic allocation; portable fixed-width binary headers.
 */

#ifndef HT_FED_PROTO_H
#define HT_FED_PROTO_H

#include "ht_config.h"

#define HT_FED_MAGIC          0x44454648U /**< ASCII "HFED" in little-endian */
#define HT_FED_PROTO_VERSION  1U

#define HT_FED_FLAG_DOUBLE    (1U << 0)
#define HT_FED_FLAG_REG       (1U << 1)

/**
 * @brief Federated wire message types.
 */
typedef enum {
    HT_FED_MSG_FOREST_CLIENT_TREE   = 1, /**< Mode 1 uplink: client compact tree + metrics */
    HT_FED_MSG_FOREST_SERVER_BUNDLE = 2, /**< Mode 1 downlink: server bundle of M compact trees */
    HT_FED_MSG_VFDT_SPLIT_PROPOSAL  = 3, /**< Mode 2A uplink: coordinated split proposals */
    HT_FED_MSG_VFDT_LEAF_STATS      = 4, /**< Mode 2B uplink: active leaf sufficient statistics */
    HT_FED_MSG_VFDT_SPLIT_BROADCAST = 5, /**< Mode 2 downlink: ordered split / reset commands */
    HT_FED_MSG_VFDT_FULL_SNAPSHOT   = 6  /**< Mode 2 downlink: full tree snapshot for straggler resync */
} ht_fed_msg_type_t;

/**
 * @brief Common binary frame header (32 bytes, natural 4-byte alignment).
 */
typedef struct {
    uint32_t magic;
    uint16_t proto_version;
    uint16_t msg_type;
    uint16_t flags;
    uint16_t float_size;
    uint16_t client_id;
    uint16_t reserved;
    uint32_t round_id;
    uint32_t topology_version;
    uint32_t payload_len;
    uint32_t crc32;
} ht_fed_header_t;

_Static_assert(sizeof(ht_fed_header_t) == 32U, "ht_fed_header_t must be exactly 32 bytes");

/**
 * @brief Computes standard IEEE 802.3 CRC32 using a compact 64-byte nibble table.
 */
uint32_t ht_fed_crc32(const void *data, size_t len);

/**
 * @brief Returns protocol flags matching the current compile precision and task type.
 */
uint16_t ht_fed_make_flags(ht_task_type_t task_type);

/**
 * @brief Writes a framed message (header + payload + CRC32) into out_buf.
 *
 * Pass out_buf == NULL to query the exact required byte size in out_written.
 * If payload == NULL and payload_len > 0, reserves payload bytes without copying
 * (caller fills payload and calls ht_fed_finalize_frame).
 */
ht_status_t ht_fed_write_frame(void *out_buf,
                               size_t buf_cap,
                               ht_fed_msg_type_t msg_type,
                               uint16_t flags,
                               uint16_t client_id,
                               uint32_t round_id,
                               uint32_t topology_version,
                               const void *payload,
                               uint32_t payload_len,
                               size_t *out_written);

/**
 * @brief Finalizes an in-place constructed frame by writing header & CRC32 over payload.
 */
ht_status_t ht_fed_finalize_frame(void *buf,
                                  size_t buf_cap,
                                  ht_fed_msg_type_t msg_type,
                                  uint16_t flags,
                                  uint16_t client_id,
                                  uint32_t round_id,
                                  uint32_t topology_version,
                                  uint32_t payload_len,
                                  size_t *out_written);

/**
 * @brief Parses and validates a framed message (magic, version, precision, length, CRC32).
 */
ht_status_t ht_fed_parse_frame(const void *buf,
                               size_t buf_len,
                               ht_fed_header_t *out_hdr,
                               const uint8_t **out_payload);

#endif /* HT_FED_PROTO_H */
