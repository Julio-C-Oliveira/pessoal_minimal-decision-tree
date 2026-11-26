CC ?= gcc
DEPFLAGS = -MMD -MP
INC_FLAGS = -Iinclude -Iinclude/core -Iinclude/ensemble -Iinclude/federated
CFLAGS = -std=c11 -O2 -Wall -Wextra -Werror -pedantic $(INC_FLAGS) $(DEPFLAGS) $(CFLAGS_EXTRA)
LDFLAGS = -lm

SRC_DIR = src
INC_DIR = include
TEST_DIR = tests
BUILD_DIR = build
BUILD_SRC = $(BUILD_DIR)/src
BUILD_TESTS = $(BUILD_DIR)/tests

# Modular source files (core, ensemble, federated)
CORE_SRCS = $(wildcard $(SRC_DIR)/core/*.c)
ENSEMBLE_SRCS = $(wildcard $(SRC_DIR)/ensemble/*.c)
FEDERATED_SRCS = $(wildcard $(SRC_DIR)/federated/*.c)
SRCS = $(CORE_SRCS) $(ENSEMBLE_SRCS) $(FEDERATED_SRCS)
OBJS = $(patsubst $(SRC_DIR)/%.c, $(BUILD_SRC)/%.o, $(SRCS))

# Test source files (root + modular subfolders)
TEST_SRCS = $(wildcard $(TEST_DIR)/*.c) $(wildcard $(TEST_DIR)/core/*.c) $(wildcard $(TEST_DIR)/ensemble/*.c) $(wildcard $(TEST_DIR)/federated/*.c)
TEST_OBJS = $(patsubst $(TEST_DIR)/%.c, $(BUILD_TESTS)/%.o, $(TEST_SRCS))
TEST_RUNNER = $(BUILD_DIR)/test_runner

VERSION ?= $(shell cat ../VERSION 2>/dev/null || cat VERSION 2>/dev/null || echo "0.1.0")

.PHONY: all lib test clean bench bench-triple bench-drift bench-regression bench-memory bench-main bench-main-federated bench-prep bench-esp32 bench-tune bench-federated bench-fed-net directories version

version:
	@echo "kansho (libht) version $(VERSION)"

all: lib test

directories:
	@mkdir -p $(BUILD_DIR) $(BUILD_SRC)/core $(BUILD_SRC)/ensemble $(BUILD_SRC)/federated $(BUILD_TESTS)/core $(BUILD_TESTS)/ensemble $(BUILD_TESTS)/federated

lib: directories $(OBJS)
	@if [ -n "$(OBJS)" ]; then \
		ar rcs $(BUILD_DIR)/libht.a $(OBJS); \
		echo "Built $(BUILD_DIR)/libht.a"; \
	fi

$(BUILD_SRC)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_TESTS)/%.o: $(TEST_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -I$(TEST_DIR) -c $< -o $@

$(TEST_RUNNER): directories $(OBJS) $(TEST_OBJS)
	$(CC) $(CFLAGS) $(OBJS) $(TEST_OBJS) $(LDFLAGS) -o $@

test: $(TEST_RUNNER)
	@echo "Executing Hoeffding Tree Test Suite..."
	./$(TEST_RUNNER)

BENCH_SRC = ../benchmarks/bench_c_runner.c
BENCH_RUNNER = $(BUILD_DIR)/bench_c_runner
BENCH_RUNNER_DOUBLE = $(BUILD_DIR)/bench_c_runner_double
BENCH_RUNNER_128 = $(BUILD_DIR)/bench_c_runner_128
BENCH_RUNNER_64 = $(BUILD_DIR)/bench_c_runner_64
BENCH_RUNNER_32 = $(BUILD_DIR)/bench_c_runner_32
BENCH_RUNNER_16 = $(BUILD_DIR)/bench_c_runner_16
BENCH_RUNNER_8 = $(BUILD_DIR)/bench_c_runner_8
BENCH_RUNNER_4 = $(BUILD_DIR)/bench_c_runner_4
BENCH_MEMORY_RUNNERS = $(BENCH_RUNNER) $(BENCH_RUNNER_128) $(BENCH_RUNNER_64) $(BENCH_RUNNER_32) $(BENCH_RUNNER_16) $(BENCH_RUNNER_8) $(BENCH_RUNNER_4)
BENCH_MNIST_RUNNER = $(BUILD_DIR)/bench_c_runner_mnist
BENCH_MNIST_DOUBLE = $(BUILD_DIR)/bench_c_runner_mnist_double
PYTHON ?= ../.venv/bin/python

$(BENCH_RUNNER): directories $(CORE_SRCS) $(BENCH_SRC)
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror -pedantic $(INC_FLAGS) $(CORE_SRCS) $(BENCH_SRC) $(LDFLAGS) -o $@

$(BENCH_RUNNER_DOUBLE): directories $(CORE_SRCS) $(BENCH_SRC)
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror -pedantic $(INC_FLAGS) -DHT_USE_DOUBLE $(CORE_SRCS) $(BENCH_SRC) $(LDFLAGS) -o $@

$(BENCH_RUNNER_128): directories $(CORE_SRCS) $(BENCH_SRC)
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror -pedantic $(INC_FLAGS) -DHT_MAX_ACTIVE_LEAVES=128 -DHT_MAX_NODES=2048 $(CORE_SRCS) $(BENCH_SRC) $(LDFLAGS) -o $@

$(BENCH_RUNNER_64): directories $(CORE_SRCS) $(BENCH_SRC)
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror -pedantic $(INC_FLAGS) -DHT_MAX_ACTIVE_LEAVES=64 -DHT_MAX_NODES=1024 $(CORE_SRCS) $(BENCH_SRC) $(LDFLAGS) -o $@

$(BENCH_RUNNER_32): directories $(CORE_SRCS) $(BENCH_SRC)
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror -pedantic $(INC_FLAGS) -DHT_MAX_ACTIVE_LEAVES=32 -DHT_MAX_NODES=512 $(CORE_SRCS) $(BENCH_SRC) $(LDFLAGS) -o $@

$(BENCH_RUNNER_16): directories $(CORE_SRCS) $(BENCH_SRC)
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror -pedantic $(INC_FLAGS) -DHT_MAX_ACTIVE_LEAVES=16 -DHT_MAX_NODES=256 $(CORE_SRCS) $(BENCH_SRC) $(LDFLAGS) -o $@

$(BENCH_RUNNER_8): directories $(CORE_SRCS) $(BENCH_SRC)
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror -pedantic $(INC_FLAGS) -DHT_MAX_ACTIVE_LEAVES=8 -DHT_MAX_NODES=128 $(CORE_SRCS) $(BENCH_SRC) $(LDFLAGS) -o $@

$(BENCH_RUNNER_4): directories $(CORE_SRCS) $(BENCH_SRC)
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror -pedantic $(INC_FLAGS) -DHT_MAX_ACTIVE_LEAVES=4 -DHT_MAX_NODES=64 $(CORE_SRCS) $(BENCH_SRC) $(LDFLAGS) -o $@

$(BENCH_MNIST_RUNNER): directories $(CORE_SRCS) $(BENCH_SRC)
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror -pedantic $(INC_FLAGS) -DHT_MAX_CLASSES=10 -DHT_MAX_ACTIVE_LEAVES=128 -DHT_MAX_NODES=2048 $(CORE_SRCS) $(BENCH_SRC) $(LDFLAGS) -o $@

$(BENCH_MNIST_DOUBLE): directories $(CORE_SRCS) $(BENCH_SRC)
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror -pedantic $(INC_FLAGS) -DHT_USE_DOUBLE -DHT_MAX_CLASSES=10 -DHT_MAX_ACTIVE_LEAVES=128 -DHT_MAX_NODES=2048 $(CORE_SRCS) $(BENCH_SRC) $(LDFLAGS) -o $@

BENCH_MAIN_RUNNER = $(BUILD_DIR)/bench_c_runner_main
BENCH_MAIN_DOUBLE = $(BUILD_DIR)/bench_c_runner_main_double
BENCH_RUNNER_8_MAIN = $(BUILD_DIR)/bench_c_runner_8_main
BENCH_RUNNER_4_MAIN = $(BUILD_DIR)/bench_c_runner_4_main

$(BENCH_MAIN_RUNNER): directories $(CORE_SRCS) $(BENCH_SRC)
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror -pedantic $(INC_FLAGS) -DHT_MAX_FEATURES=64 -DHT_MAX_CLASSES=8 -DHT_MAX_ACTIVE_LEAVES=40 -DHT_MAX_NODES=1024 $(CORE_SRCS) $(BENCH_SRC) $(LDFLAGS) -o $@

$(BENCH_MAIN_DOUBLE): directories $(CORE_SRCS) $(BENCH_SRC)
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror -pedantic $(INC_FLAGS) -DHT_USE_DOUBLE -DHT_MAX_FEATURES=64 -DHT_MAX_CLASSES=8 -DHT_MAX_ACTIVE_LEAVES=40 -DHT_MAX_NODES=1024 $(CORE_SRCS) $(BENCH_SRC) $(LDFLAGS) -o $@

$(BENCH_RUNNER_8_MAIN): directories $(CORE_SRCS) $(BENCH_SRC)
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror -pedantic $(INC_FLAGS) -DHT_MAX_FEATURES=64 -DHT_MAX_CLASSES=8 -DHT_MAX_ACTIVE_LEAVES=8 -DHT_MAX_NODES=128 $(CORE_SRCS) $(BENCH_SRC) $(LDFLAGS) -o $@

$(BENCH_RUNNER_4_MAIN): directories $(CORE_SRCS) $(BENCH_SRC)
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror -pedantic $(INC_FLAGS) -DHT_MAX_FEATURES=64 -DHT_MAX_CLASSES=8 -DHT_MAX_ACTIVE_LEAVES=4 -DHT_MAX_NODES=64 $(CORE_SRCS) $(BENCH_SRC) $(LDFLAGS) -o $@

bench: $(BENCH_RUNNER) $(BENCH_RUNNER_DOUBLE)
	@echo "Executing Hoeffding Tree Parity & Memory Benchmark vs River..."
	@if [ -x "$(PYTHON)" ]; then \
		$(PYTHON) ../benchmarks/compare_river.py --bin-float $(BENCH_RUNNER) --bin-double $(BENCH_RUNNER_DOUBLE) --data-dir ../benchmarks/data --report ../benchmarks/benchmark_report.md; \
	elif command -v python3 >/dev/null 2>&1; then \
		python3 ../benchmarks/compare_river.py --bin-float $(BENCH_RUNNER) --bin-double $(BENCH_RUNNER_DOUBLE) --data-dir ../benchmarks/data --report ../benchmarks/benchmark_report.md; \
	else \
		echo "Error: Python 3 with river not found!"; exit 1; \
	fi

bench-triple: $(BENCH_RUNNER) $(BENCH_RUNNER_DOUBLE) $(BENCH_MNIST_RUNNER) $(BENCH_MNIST_DOUBLE)
	@echo "Executing Triple Benchmark: C11 vs River vs Scikit-Learn on MNIST..."
	@if [ -x "$(PYTHON)" ]; then \
		$(PYTHON) ../benchmarks/compare_triple.py --bin-float $(BENCH_RUNNER) --bin-double $(BENCH_RUNNER_DOUBLE) --bin-mnist-float $(BENCH_MNIST_RUNNER) --bin-mnist-double $(BENCH_MNIST_DOUBLE) --data-dir ../benchmarks/data --report ../benchmarks/triple_benchmark_report.md; \
	elif command -v python3 >/dev/null 2>&1; then \
		python3 ../benchmarks/compare_triple.py --bin-float $(BENCH_RUNNER) --bin-double $(BENCH_RUNNER_DOUBLE) --bin-mnist-float $(BENCH_MNIST_RUNNER) --bin-mnist-double $(BENCH_MNIST_DOUBLE) --data-dir ../benchmarks/data --report ../benchmarks/triple_benchmark_report.md; \
	else \
		echo "Error: Python 3 not found!"; exit 1; \
	fi

bench-drift: $(BENCH_RUNNER) $(BENCH_RUNNER_DOUBLE)
	@echo "Executing Concept Drift & HAT Parity Benchmark vs River (Task 3.3)..."
	@if [ -x "$(PYTHON)" ]; then \
		$(PYTHON) ../benchmarks/compare_river.py --mode drift --bin-float $(BENCH_RUNNER) --bin-double $(BENCH_RUNNER_DOUBLE) --data-dir ../benchmarks/data --report-drift ../benchmarks/drift_benchmark_report.md; \
	elif command -v python3 >/dev/null 2>&1; then \
		python3 ../benchmarks/compare_river.py --mode drift --bin-float $(BENCH_RUNNER) --bin-double $(BENCH_RUNNER_DOUBLE) --data-dir ../benchmarks/data --report-drift ../benchmarks/drift_benchmark_report.md; \
	else \
		echo "Error: Python 3 with river not found!"; exit 1; \
	fi

bench-regression: $(BENCH_RUNNER) $(BENCH_RUNNER_DOUBLE)
	@echo "Executing Triple Benchmark: C11 vs River HTR/HATR vs Scikit-Learn in Regression (Task 4.5)..."
	@if [ -x "$(PYTHON)" ]; then \
		$(PYTHON) ../benchmarks/compare_river.py --mode regression --bin-float $(BENCH_RUNNER) --bin-double $(BENCH_RUNNER_DOUBLE) --data-dir ../benchmarks/data --report-regression ../benchmarks/regression_benchmark_report.md; \
	elif command -v python3 >/dev/null 2>&1; then \
		python3 ../benchmarks/compare_river.py --mode regression --bin-float $(BENCH_RUNNER) --bin-double $(BENCH_RUNNER_DOUBLE) --data-dir ../benchmarks/data --report-regression ../benchmarks/regression_benchmark_report.md; \
	else \
		echo "Error: Python 3 with river not found!"; exit 1; \
	fi

bench-memory: $(BENCH_MEMORY_RUNNERS) $(BENCH_RUNNER_DOUBLE)
	@echo "Executing Memory Scaling Benchmark across capacity limits (256, 128, 64, 32, 16, 8, 4 leaves)..."
	@if [ -x "$(PYTHON)" ]; then \
		$(PYTHON) ../benchmarks/compare_river.py --mode memory --bin-float $(BENCH_RUNNER) --bin-double $(BENCH_RUNNER_DOUBLE) --data-dir ../benchmarks/data --report-memory ../benchmarks/memory_benchmark_report.md; \
	elif command -v python3 >/dev/null 2>&1; then \
		python3 ../benchmarks/compare_river.py --mode memory --bin-float $(BENCH_RUNNER) --bin-double $(BENCH_RUNNER_DOUBLE) --data-dir ../benchmarks/data --report-memory ../benchmarks/memory_benchmark_report.md; \
	else \
		echo "Error: Python 3 with river not found!"; exit 1; \
	fi

bench-main: $(BENCH_RUNNER) $(BENCH_RUNNER_8) $(BENCH_RUNNER_4) $(BENCH_MAIN_RUNNER) $(BENCH_RUNNER_8_MAIN) $(BENCH_RUNNER_4_MAIN)
	@echo "Executing Main Performance Benchmark Suite (8 Datasets, Multi-Tier C11)..."
	@if [ -x "$(PYTHON)" ]; then \
		$(PYTHON) ../main_benchmark/main_benchmark.py \
			--bin-tier256 $(BENCH_RUNNER) \
			--bin-tier8 $(BENCH_RUNNER_8) \
			--bin-tier4 $(BENCH_RUNNER_4) \
			--bin-main-tier256 $(BENCH_MAIN_RUNNER) \
			--bin-main-tier8 $(BENCH_RUNNER_8_MAIN) \
			--bin-main-tier4 $(BENCH_RUNNER_4_MAIN) \
			--data-dir ../main_benchmark/data \
			--report ../main_benchmark/main_benchmark_report.md; \
	elif command -v python3 >/dev/null 2>&1; then \
		python3 ../main_benchmark/main_benchmark.py \
			--bin-tier256 $(BENCH_RUNNER) \
			--bin-tier8 $(BENCH_RUNNER_8) \
			--bin-tier4 $(BENCH_RUNNER_4) \
			--bin-main-tier256 $(BENCH_MAIN_RUNNER) \
			--bin-main-tier8 $(BENCH_RUNNER_8_MAIN) \
			--bin-main-tier4 $(BENCH_RUNNER_4_MAIN) \
			--data-dir ../main_benchmark/data \
			--report ../main_benchmark/main_benchmark_report.md; \
	else \
		echo "Error: Python 3 with river not found!"; exit 1; \
	fi

bench-prep: $(BENCH_RUNNER) $(BENCH_RUNNER_DOUBLE)
	@echo "Executing Preprocessing & Non-Parametric Benchmark: C11 vs River (Task 6.6)..."
	@if [ -x "$(PYTHON)" ]; then \
		$(PYTHON) ../benchmarks/compare_prep.py --bin-float $(BENCH_RUNNER) --bin-double $(BENCH_RUNNER_DOUBLE) --data-dir ../benchmarks/data --report ../benchmarks/prep_benchmark_report.md; \
	elif command -v python3 >/dev/null 2>&1; then \
		python3 ../benchmarks/compare_prep.py --bin-float $(BENCH_RUNNER) --bin-double $(BENCH_RUNNER_DOUBLE) --data-dir ../benchmarks/data --report ../benchmarks/prep_benchmark_report.md; \
	else \
		echo "Error: Python 3 with river not found!"; exit 1; \
	fi

bench-tune: $(BENCH_RUNNER)
	@echo "Executing Hyperparameter Tuning & Sensitivity Benchmark..."
	@if [ -x "$(PYTHON)" ]; then \
		$(PYTHON) ../main_benchmark/tune_hyperparams.py \
			--data-dir ../main_benchmark/data \
			--report ../main_benchmark/hyperparam_report.md \
			--json ../main_benchmark/hyperparam_results.json \
			$(TUNE_ARGS); \
	elif command -v python3 >/dev/null 2>&1; then \
		python3 ../main_benchmark/tune_hyperparams.py \
			--data-dir ../main_benchmark/data \
			--report ../main_benchmark/hyperparam_report.md \
			--json ../main_benchmark/hyperparam_results.json \
			$(TUNE_ARGS); \
	else \
		echo "Error: Python 3 not found!"; exit 1; \
	fi

bench-esp32:
	@echo "Executing ESP32 QEMU Emulation Benchmark..."
	@bash ../emulation/esp32/run_qemu.sh bench

FED_SIM_SRC = ../benchmarks/fed_simulator.c
FED_SIM_BIN = $(BUILD_DIR)/fed_simulator

$(FED_SIM_BIN): directories $(SRCS) $(FED_SIM_SRC)
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror -pedantic $(INC_FLAGS) \
		-DHT_MAX_NODES=128 -DHT_MAX_ACTIVE_LEAVES=4 -DHT_COMPACT_MAX_NODES=64 \
		-DHT_FED_FOREST_MAX_TREES=8 -DHT_FED_SERVER_MAX_TREES=16 \
		$(SRCS) $(FED_SIM_SRC) $(LDFLAGS) -o $@

bench-federated: $(FED_SIM_BIN)
	@echo "Executing Multi-Client Federated Learning Simulator (MCU Tier 4 Profile)..."
	./$(FED_SIM_BIN)

FED_NET_SRV_SRC = ../benchmarks/fed_net/fed_net_server.c
FED_NET_CLI_SRC = ../benchmarks/fed_net/fed_net_client.c
FED_NET_SRV_BIN = $(BUILD_DIR)/fed_net_server
FED_NET_CLI_BIN = $(BUILD_DIR)/fed_net_client
FED_NET_DEFS    = -DHT_MAX_NODES=128 -DHT_MAX_ACTIVE_LEAVES=8 -DHT_COMPACT_MAX_NODES=64 \
                  -DHT_FED_FOREST_MAX_TREES=8 -DHT_FED_SERVER_MAX_TREES=16 -DHT_FED_SERVER_MAX_PROPOSALS=128

$(FED_NET_SRV_BIN): directories $(SRCS) $(FED_NET_SRV_SRC) ../benchmarks/fed_net/fed_net_common.h
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror -pedantic $(INC_FLAGS) -I../benchmarks/fed_net $(FED_NET_DEFS) \
		$(SRCS) $(FED_NET_SRV_SRC) $(LDFLAGS) -o $@

$(FED_NET_CLI_BIN): directories $(SRCS) $(FED_NET_CLI_SRC) ../benchmarks/fed_net/fed_net_common.h
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror -pedantic $(INC_FLAGS) -I../benchmarks/fed_net $(FED_NET_DEFS) \
		$(SRCS) $(FED_NET_CLI_SRC) $(LDFLAGS) -o $@

bench-fed-net: $(FED_NET_SRV_BIN) $(FED_NET_CLI_BIN)
	@echo "Executing Real POSIX TCP Multi-Process Federated Validation Suite (Gate A — 11.10)..."
	@if command -v python3 >/dev/null 2>&1; then \
		python3 ../benchmarks/fed_net/run_fed_net_tests.py --server-bin ./$(FED_NET_SRV_BIN) --client-bin ./$(FED_NET_CLI_BIN); \
	elif [ -x "$(PYTHON)" ]; then \
		$(PYTHON) ../benchmarks/fed_net/run_fed_net_tests.py --server-bin ./$(FED_NET_SRV_BIN) --client-bin ./$(FED_NET_CLI_BIN); \
	else \
		echo "Error: Python 3 not found!"; exit 1; \
	fi

bench-main-federated: directories
	@echo "Executing Real-Dataset Multi-Client Federated Benchmark (8 Datasets x Non-IID & IID)..."
	@if [ -x "$(PYTHON)" ]; then \
		$(PYTHON) ../main_benchmark/federated_benchmark.py --data-dir ../main_benchmark/data --report ../main_benchmark/federated_benchmark_report.md --json ../main_benchmark/federated_benchmark_results.json $(FED_ARGS); \
	elif command -v python3 >/dev/null 2>&1; then \
		python3 ../main_benchmark/federated_benchmark.py --data-dir ../main_benchmark/data --report ../main_benchmark/federated_benchmark_report.md --json ../main_benchmark/federated_benchmark_results.json $(FED_ARGS); \
	else \
		echo "Error: Python 3 not found!"; exit 1; \
	fi

clean:
	rm -rf $(BUILD_DIR)

-include $(wildcard $(BUILD_SRC)/*.d) $(wildcard $(BUILD_SRC)/*/*.d) $(wildcard $(BUILD_TESTS)/*.d) $(wildcard $(BUILD_TESTS)/*/*.d)
