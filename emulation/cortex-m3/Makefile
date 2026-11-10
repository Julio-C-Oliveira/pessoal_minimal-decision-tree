# ==============================================================================
# Makefile — Bare-Metal ARM Cortex-M3 (TI Stellaris LM3S6965) libht Firmware
# ==============================================================================

CC      = arm-none-eabi-gcc
OBJCOPY = arm-none-eabi-objcopy
SIZE    = arm-none-eabi-size

PROJECT_ROOT = ../..
CODE_DIR     = $(PROJECT_ROOT)/code
BUILD_DIR    = build

TARGET_ELF = $(BUILD_DIR)/cortex_m3_hoeffding_tree.elf
TARGET_BIN = $(BUILD_DIR)/cortex_m3_hoeffding_tree.bin

# Architecture & Target Flags
ARCH_FLAGS = -mcpu=cortex-m3 -mthumb -mfloat-abi=soft

# Memory & Capacity Configuration: Tier 4 Universal (fits strictly in 64 KB SRAM)
CONFIG_DEFS = -DHT_MAX_ACTIVE_LEAVES=4 \
              -DHT_MAX_FEATURES=64 \
              -DHT_MAX_CLASSES=8 \
              -DHT_MAX_NODES=128 \
              -DHT_MAX_ADWIN_DETECTORS=2

CFLAGS = $(ARCH_FLAGS) \
         -std=c11 -Wall -Wextra -Werror -pedantic -Os -g3 \
         -ffunction-sections -fdata-sections \
         $(CONFIG_DEFS) \
         -Isrc -I$(CODE_DIR)/include

LDFLAGS = $(ARCH_FLAGS) \
          -Tlm3s6965.ld \
          -nostartfiles \
          -Wl,--gc-sections \
          -Wl,-Map=$(BUILD_DIR)/firmware.map \
          -Wl,-u,_printf_float \
          --specs=nano.specs \
          --specs=nosys.specs \
          -lc -lm

SRCS = src/startup_lm3s6965.c \
       src/uart.c \
       src/cycles.c \
       src/main.c \
       $(CODE_DIR)/src/ht_memory.c \
       $(CODE_DIR)/src/ht_tree.c \
       $(CODE_DIR)/src/ht_criterion.c \
       $(CODE_DIR)/src/ht_observer.c \
       $(CODE_DIR)/src/ht_drift.c \
       $(CODE_DIR)/src/ht_prep.c

OBJS = $(patsubst %.c,$(BUILD_DIR)/%.o,$(notdir $(SRCS)))

vpath %.c src $(CODE_DIR)/src

.PHONY: all clean size

all: $(TARGET_BIN) size

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: %.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET_ELF): $(OBJS) lm3s6965.ld
	$(CC) $(OBJS) $(LDFLAGS) -o $@

$(TARGET_BIN): $(TARGET_ELF)
	$(OBJCOPY) -O binary $< $@

size: $(TARGET_ELF)
	@echo "=== Firmware Memory Layout ==="
	$(SIZE) $<
	@echo "=============================="

clean:
	rm -rf $(BUILD_DIR)
