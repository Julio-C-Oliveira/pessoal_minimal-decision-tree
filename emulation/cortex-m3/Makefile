# ==============================================================================
# Makefile — Bare-Metal ARM Cortex-M3 (TI Stellaris LM3S6965) libht Firmware
# ==============================================================================

CC      = arm-none-eabi-gcc
OBJCOPY = arm-none-eabi-objcopy
SIZE    = arm-none-eabi-size
NM      = arm-none-eabi-nm

FIXED_POINT ?= 0

PROJECT_ROOT = ../..
CODE_DIR     = $(PROJECT_ROOT)/code
ifeq ($(FIXED_POINT),1)
BUILD_DIR    = build_fixed
else
BUILD_DIR    = build
endif

TARGET_ELF = $(BUILD_DIR)/cortex_m3_hoeffding_tree.elf
TARGET_BIN = $(BUILD_DIR)/cortex_m3_hoeffding_tree.bin

# Architecture & Target Flags
ARCH_FLAGS = -mcpu=cortex-m3 -mthumb -mfloat-abi=soft

# Memory & Capacity Configuration: Tier 4 Universal (fits strictly in 64 KB SRAM)
CONFIG_DEFS = -DHT_MAX_ACTIVE_LEAVES=4 \
              -DHT_MAX_FEATURES=64 \
              -DHT_MAX_CLASSES=8 \
              -DHT_MAX_NODES=128 \
              -DHT_MAX_ADWIN_DETECTORS=2 \
              -DHT_ADWIN_LEVELS=10

ifeq ($(FIXED_POINT),1)
CONFIG_DEFS += -DHT_USE_FIXED_POINT
endif

CFLAGS = $(ARCH_FLAGS) \
         -std=c11 -Wall -Wextra -Werror -pedantic -Os -g3 \
         -ffunction-sections -fdata-sections \
         $(CONFIG_DEFS) \
         -Isrc -I$(CODE_DIR)/include -I$(CODE_DIR)/include/core -I$(CODE_DIR)/include/ensemble -I$(CODE_DIR)/include/federated

ifeq ($(FIXED_POINT),1)
# Zero-Libm & Zero-Float-Printf link flags for Q16.16 fixed-point firmware
LDFLAGS = $(ARCH_FLAGS) \
          -Tlm3s6965.ld \
          -nostartfiles \
          -Wl,--gc-sections \
          -Wl,-Map=$(BUILD_DIR)/firmware.map \
          --specs=nano.specs \
          --specs=nosys.specs \
          -lc
else
LDFLAGS = $(ARCH_FLAGS) \
          -Tlm3s6965.ld \
          -nostartfiles \
          -Wl,--gc-sections \
          -Wl,-Map=$(BUILD_DIR)/firmware.map \
          -Wl,-u,_printf_float \
          --specs=nano.specs \
          --specs=nosys.specs \
          -lc -lm
endif

SRCS = src/startup_lm3s6965.c \
       src/uart.c \
       src/cycles.c \
       src/main.c \
       $(CODE_DIR)/src/core/ht_memory.c \
       $(CODE_DIR)/src/core/ht_tree.c \
       $(CODE_DIR)/src/core/ht_criterion.c \
       $(CODE_DIR)/src/core/ht_observer.c \
       $(CODE_DIR)/src/core/ht_drift.c \
       $(CODE_DIR)/src/core/ht_prep.c

ifeq ($(FIXED_POINT),1)
SRCS += $(CODE_DIR)/src/core/ht_math_fixed.c
endif

OBJS = $(patsubst %.c,$(BUILD_DIR)/%.o,$(notdir $(SRCS)))

vpath %.c src $(CODE_DIR)/src/core $(CODE_DIR)/src/ensemble $(CODE_DIR)/src/federated

.PHONY: all clean size audit

ifeq ($(FIXED_POINT),1)
all: $(TARGET_BIN) size audit
else
all: $(TARGET_BIN) size
endif

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: %.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET_ELF): $(OBJS) lm3s6965.ld
	$(CC) $(OBJS) $(LDFLAGS) -o $@

$(TARGET_BIN): $(TARGET_ELF)
	$(OBJCOPY) -O binary $< $@

size: $(TARGET_ELF)
	@echo "=== Firmware Memory Layout (FIXED_POINT=$(FIXED_POINT)) ==="
	$(SIZE) $<
	@echo "==========================================================="

audit: $(TARGET_ELF)
	@echo "=== Zero-Libm & Zero-SoftFP ELF Audit ($(TARGET_ELF)) ==="
	@if $(NM) $< | grep -E '__aeabi_(f|d|i2f|ui2f|l2f|ul2f|i2d|ui2d|l2d|ul2d)|_printf_float|__addsf|__adddf|__mulsf|__muldf|__divsf|__divdf|\b(sqrt|sqrtf|log|logf|log2|log2f|exp|expf|erf|erff|erfc|erfcf|pow|powf|strtod|atof)\b'; then \
		echo "ERROR: Floating-point / libm symbols detected in Q16.16 firmware ELF!"; \
		exit 1; \
	else \
		echo "PASS: Zero __aeabi_f*/__aeabi_d*, zero _printf_float, zero libm symbols in ELF."; \
	fi
	@echo "==========================================================="

clean:
	rm -rf build build_fixed
