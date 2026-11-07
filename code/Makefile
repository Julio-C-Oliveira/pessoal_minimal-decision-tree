CC ?= gcc
DEPFLAGS = -MMD -MP
CFLAGS = -std=c11 -O2 -Wall -Wextra -Werror -pedantic -Iinclude $(DEPFLAGS) $(CFLAGS_EXTRA)
LDFLAGS = -lm

SRC_DIR = src
INC_DIR = include
TEST_DIR = tests
BUILD_DIR = build
BUILD_SRC = $(BUILD_DIR)/src
BUILD_TESTS = $(BUILD_DIR)/tests

# Source files (empty in Milestone 1.1, populated in subsequent milestones)
SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(patsubst $(SRC_DIR)/%.c, $(BUILD_SRC)/%.o, $(SRCS))

# Test source files
TEST_SRCS = $(wildcard $(TEST_DIR)/*.c)
TEST_OBJS = $(patsubst $(TEST_DIR)/%.c, $(BUILD_TESTS)/%.o, $(TEST_SRCS))
TEST_RUNNER = $(BUILD_DIR)/test_runner

.PHONY: all lib test clean bench directories

all: lib test

directories:
	@mkdir -p $(BUILD_DIR) $(BUILD_SRC) $(BUILD_TESTS)

lib: directories $(OBJS)
	@if [ -n "$(OBJS)" ]; then \
		ar rcs $(BUILD_DIR)/libht.a $(OBJS); \
		echo "Built $(BUILD_DIR)/libht.a"; \
	fi

$(BUILD_SRC)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(BUILD_SRC)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_TESTS)/%.o: $(TEST_DIR)/%.c
	@mkdir -p $(BUILD_TESTS)
	$(CC) $(CFLAGS) -c $< -o $@

$(TEST_RUNNER): directories $(OBJS) $(TEST_OBJS)
	$(CC) $(CFLAGS) $(OBJS) $(TEST_OBJS) $(LDFLAGS) -o $@

test: $(TEST_RUNNER)
	@echo "Executing Hoeffding Tree Test Suite..."
	./$(TEST_RUNNER)

BENCH_SRC = ../benchmarks/bench_c_runner.c
BENCH_RUNNER = $(BUILD_DIR)/bench_c_runner
BENCH_RUNNER_DOUBLE = $(BUILD_DIR)/bench_c_runner_double
PYTHON ?= ../.venv/bin/python

$(BENCH_RUNNER): directories $(SRCS) $(BENCH_SRC)
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror -pedantic -Iinclude $(SRCS) $(BENCH_SRC) $(LDFLAGS) -o $@

$(BENCH_RUNNER_DOUBLE): directories $(SRCS) $(BENCH_SRC)
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror -pedantic -Iinclude -DHT_USE_DOUBLE $(SRCS) $(BENCH_SRC) $(LDFLAGS) -o $@

bench: $(BENCH_RUNNER) $(BENCH_RUNNER_DOUBLE)
	@echo "Executing Hoeffding Tree Parity & Memory Benchmark vs River..."
	@if [ -x "$(PYTHON)" ]; then \
		$(PYTHON) ../benchmarks/compare_river.py --bin-float $(BENCH_RUNNER) --bin-double $(BENCH_RUNNER_DOUBLE) --data-dir ../benchmarks/data --report ../benchmarks/benchmark_report.md; \
	elif command -v python3 >/dev/null 2>&1; then \
		python3 ../benchmarks/compare_river.py --bin-float $(BENCH_RUNNER) --bin-double $(BENCH_RUNNER_DOUBLE) --data-dir ../benchmarks/data --report ../benchmarks/benchmark_report.md; \
	else \
		echo "Error: Python 3 with river not found!"; exit 1; \
	fi

clean:
	rm -rf $(BUILD_DIR)

-include $(wildcard $(BUILD_SRC)/*.d) $(wildcard $(BUILD_TESTS)/*.d)
