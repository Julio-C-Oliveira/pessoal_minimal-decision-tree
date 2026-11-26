/**
 * @file ht_fed_proto.c
 * @brief Implementation of transport-agnostic binary framing and CRC32.
 */

#include "ht_fed_proto.h"
#include <string.h>

/* Compact 16-entry nibble lookup table (64 bytes in Flash/rodata) for IEEE 802.3 CRC32 */
static const uint32_t ht_crc32_nibble_tbl[16] = {
    0x00000000U, 0x1DB71064U, 0x3B6E20C8U, 0x26D930ACU,
    0x76DC4190U, 0x6B6B51F4U, 0x4DB26158U, 0x5005713CU,
    0xEDB88320U, 0xF00F9344U, 0xD6D6A3E8U, 0xCB61B38CU,
    0x9B64C2B0U, 0x86D3D2D4U, 0xA00AE278U, 0xBDBDF21CU
};

uint32_t ht_fed_crc32(const void *data, size_t len) {
    if (data == NULL || len == 0U) {
        return 0U;
    }
    const uint8_t *bytes = (const uint8_t *)data;
    uint32_t crc = 0xFFFFFFFFU;
    for (size_t i = 0; i < len; ++i) {
        crc ^= (uint32_t)bytes[i];
        crc = (crc >> 4) ^ ht_crc32_nibble_tbl[crc & 0x0FU];
        crc = (crc >> 4) ^ ht_crc32_nibble_tbl[crc & 0x0FU];
    }
    return ~crc;
}

uint16_t ht_fed_make_flags(ht_task_type_t task_type) {
    uint16_t flags = 0U;
#ifdef HT_USE_DOUBLE
    flags |= HT_FED_FLAG_DOUBLE;
#endif
    if (task_type == HT_TASK_REGRESSION) {
        flags |= HT_FED_FLAG_REG;
    }
    return flags;
}

ht_status_t ht_fed_finalize_frame(void *buf,
                                  size_t buf_cap,
                                  ht_fed_msg_type_t msg_type,
                                  uint16_t flags,
                                  uint16_t client_id,
                                  uint32_t round_id,
                                  uint32_t topology_version,
                                  uint32_t payload_len,
                                  size_t *out_written) {
    size_t total = sizeof(ht_fed_header_t) + (size_t)payload_len;
    if (out_written != NULL) {
        *out_written = total;
    }
    if (buf == NULL) {
        return HT_OK;
    }
    if (buf_cap < total) {
        return HT_ERR_BUFFER_TOO_SMALL;
    }

    uint8_t *bytes = (uint8_t *)buf;
    ht_fed_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = HT_FED_MAGIC;
    hdr.proto_version = HT_FED_PROTO_VERSION;
    hdr.msg_type = (uint16_t)msg_type;
    hdr.flags = flags;
    hdr.float_size = (uint16_t)sizeof(ht_float_t);
    hdr.client_id = client_id;
    hdr.round_id = round_id;
    hdr.topology_version = topology_version;
    hdr.payload_len = payload_len;
    hdr.crc32 = ht_fed_crc32(bytes + sizeof(ht_fed_header_t), payload_len);

    memcpy(bytes, &hdr, sizeof(hdr));
    return HT_OK;
}

ht_status_t ht_fed_write_frame(void *out_buf,
                               size_t buf_cap,
                               ht_fed_msg_type_t msg_type,
                               uint16_t flags,
                               uint16_t client_id,
                               uint32_t round_id,
                               uint32_t topology_version,
                               const void *payload,
                               uint32_t payload_len,
                               size_t *out_written) {
    size_t total = sizeof(ht_fed_header_t) + (size_t)payload_len;
    if (out_written != NULL) {
        *out_written = total;
    }
    if (out_buf == NULL) {
        return HT_OK;
    }
    if (buf_cap < total) {
        return HT_ERR_BUFFER_TOO_SMALL;
    }
    if (payload_len > 0U && payload != NULL) {
        memcpy((uint8_t *)out_buf + sizeof(ht_fed_header_t), payload, payload_len);
    }
    return ht_fed_finalize_frame(out_buf, buf_cap, msg_type, flags, client_id,
                                 round_id, topology_version, payload_len, out_written);
}

ht_status_t ht_fed_parse_frame(const void *buf,
                               size_t buf_len,
                               ht_fed_header_t *out_hdr,
                               const uint8_t **out_payload) {
    if (buf == NULL || out_hdr == NULL) {
        return HT_ERR_INVALID_ARG;
    }
    if (buf_len < sizeof(ht_fed_header_t)) {
        return HT_ERR_CORRUPT_IMAGE;
    }

    const uint8_t *bytes = (const uint8_t *)buf;
    ht_fed_header_t hdr;
    memcpy(&hdr, bytes, sizeof(hdr));

    if (hdr.magic != HT_FED_MAGIC ||
        hdr.proto_version != HT_FED_PROTO_VERSION ||
        hdr.float_size != (uint16_t)sizeof(ht_float_t)) {
        return HT_ERR_CORRUPT_IMAGE;
    }

    size_t total = sizeof(ht_fed_header_t) + (size_t)hdr.payload_len;
    if (buf_len < total) {
        return HT_ERR_CORRUPT_IMAGE;
    }

    const uint8_t *payload = bytes + sizeof(ht_fed_header_t);
    uint32_t calc_crc = ht_fed_crc32(payload, hdr.payload_len);
    if (calc_crc != hdr.crc32) {
        return HT_ERR_CORRUPT_IMAGE;
    }

    *out_hdr = hdr;
    if (out_payload != NULL) {
        *out_payload = payload;
    }
    return HT_OK;
}
