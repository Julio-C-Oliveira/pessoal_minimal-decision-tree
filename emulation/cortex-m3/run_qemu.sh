#!/usr/bin/env bash
# ==============================================================================
# run_qemu.sh — Build and Emulate ARM Cortex-M3 (TI Stellaris LM3S6965) Firmware
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
ELF_IMG="${BUILD_DIR}/cortex_m3_hoeffding_tree.elf"
BIN_IMG="${BUILD_DIR}/cortex_m3_hoeffding_tree.bin"
PORT="${QEMU_SERIAL_PORT:-4001}"
DOCKER_IMAGE="kansho-cortex-m3:latest"
CONTAINER_NAME="cortex_m3_qemu_bench"

# ANSI Colors
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() {
    echo -e "${CYAN}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# ------------------------------------------------------------------------------
# Environment Detection (Native ARM GCC & QEMU vs Docker)
# ------------------------------------------------------------------------------
setup_env() {
    if command -v arm-none-eabi-gcc >/dev/null 2>&1 && command -v qemu-system-arm >/dev/null 2>&1; then
        log_info "Found native arm-none-eabi-gcc and qemu-system-arm in PATH."
        USE_DOCKER=0
        return 0
    fi

    if command -v docker >/dev/null 2>&1; then
        log_info "Native ARM toolchain not found. Using Docker container (${DOCKER_IMAGE})..."
        USE_DOCKER=1
        ensure_docker_image
        return 0
    fi

    log_error "Neither native ARM toolchain nor Docker were found in PATH."
    exit 1
}

ensure_docker_image() {
    if ! docker image inspect "${DOCKER_IMAGE}" >/dev/null 2>&1; then
        log_info "Building local Docker toolchain image (${DOCKER_IMAGE})..."
        docker build -t "${DOCKER_IMAGE}" "${SCRIPT_DIR}"
        log_success "Docker image ${DOCKER_IMAGE} built successfully."
    fi
}

USE_DOCKER=0
setup_env

# ------------------------------------------------------------------------------
# Subcommands
# ------------------------------------------------------------------------------
cmd_build() {
    log_info "Building ARM Cortex-M3 firmware in ${SCRIPT_DIR}..."
    cd "${SCRIPT_DIR}"

    if [ "${USE_DOCKER}" -eq 1 ]; then
        docker run --rm -v "${PROJECT_ROOT}:/project" -w "/project/emulation/cortex-m3" \
            "${DOCKER_IMAGE}" make
        docker run --rm -v "${PROJECT_ROOT}:/project" \
            "${DOCKER_IMAGE}" chmod -R 777 "/project/emulation/cortex-m3/build"
    else
        make
    fi
    log_success "Cortex-M3 Firmware built successfully: ${ELF_IMG}"
}

cmd_qemu() {
    local target_port="${1:-${PORT}}"
    if [ ! -f "${ELF_IMG}" ]; then
        cmd_build
    fi

    if [ "${USE_DOCKER}" -eq 1 ]; then
        log_info "Starting ARM Cortex-M3 QEMU via Docker container on port ${target_port}..."
        docker run --rm -it \
            -v "${PROJECT_ROOT}:/project" \
            -w "/project/emulation/cortex-m3" \
            -p "${target_port}:${target_port}" \
            "${DOCKER_IMAGE}" \
            qemu-system-arm \
                -M lm3s6965evb \
                -nographic \
                -kernel build/cortex_m3_hoeffding_tree.elf \
                -serial "tcp:0.0.0.0:${target_port},server,nowait"
    else
        log_info "Starting ARM Cortex-M3 QEMU on serial TCP port ${target_port}..."
        qemu-system-arm \
            -M lm3s6965evb \
            -nographic \
            -kernel "${ELF_IMG}" \
            -serial "tcp:127.0.0.1:${target_port},server,nowait"
    fi
}

cmd_bench() {
    local target_port="${1:-${PORT}}"
    if [ ! -f "${ELF_IMG}" ]; then
        cmd_build
    fi

    local max_samp="${MAX_SAMPLES:-0}"
    local extra_args=()
    if [ "${max_samp}" != "0" ]; then
        extra_args+=(--max-samples "${max_samp}")
    fi

    if [ "${USE_DOCKER}" -eq 1 ]; then
        log_info "Launching background QEMU container (${CONTAINER_NAME}) on port ${target_port}..."
        docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
        docker run -d --rm \
            --name "${CONTAINER_NAME}" \
            -v "${PROJECT_ROOT}:/project" \
            -w "/project/emulation/cortex-m3" \
            -p "${target_port}:${target_port}" \
            "${DOCKER_IMAGE}" \
            qemu-system-arm \
                -M lm3s6965evb \
                -nographic \
                -kernel build/cortex_m3_hoeffding_tree.elf \
                -serial "tcp:0.0.0.0:${target_port},server,nowait" > "${BUILD_DIR}/qemu.log" 2>&1

        trap 'docker rm -f ${CONTAINER_NAME} >/dev/null 2>&1 || true' EXIT

        log_info "Waiting for QEMU socket to open on port ${target_port}..."
        sleep 2

        log_info "Running ARM Cortex-M3 Stream Benchmark Runner via Docker..."
        docker exec "${CONTAINER_NAME}" python3 "/project/emulation/cortex-m3/stream_runner.py" \
            --host 127.0.0.1 \
            --port "${target_port}" \
            --suite main \
            --report "/project/emulation/cortex-m3/cortex_m3_emulation_report.md" \
            ${extra_args[@]+"${extra_args[@]}"}
    else
        log_info "Launching background QEMU instance on port ${target_port}..."
        qemu-system-arm \
            -M lm3s6965evb \
            -nographic \
            -kernel "${ELF_IMG}" \
            -serial "tcp:127.0.0.1:${target_port},server,nowait" > "${BUILD_DIR}/qemu.log" 2>&1 &
        local QEMU_PID=$!
        trap 'kill -9 ${QEMU_PID} 2>/dev/null || true' EXIT

        log_info "Waiting for QEMU socket to open on port ${target_port}..."
        sleep 2

        PYTHON_BIN="python3"
        if [ -x "${PROJECT_ROOT}/.venv/bin/python" ]; then
            PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
        fi

        log_info "Running ARM Cortex-M3 Stream Benchmark Runner via ${PYTHON_BIN}..."
        "${PYTHON_BIN}" "${SCRIPT_DIR}/stream_runner.py" \
            --host 127.0.0.1 \
            --port "${target_port}" \
            --suite main \
            --report "${SCRIPT_DIR}/cortex_m3_emulation_report.md" \
            ${extra_args[@]+"${extra_args[@]}"}
    fi

    log_success "Benchmark completed! Report saved in ${SCRIPT_DIR}/cortex_m3_emulation_report.md"
}

cmd_clean() {
    log_info "Cleaning ARM Cortex-M3 build artifacts..."
    rm -rf "${BUILD_DIR}"
    log_success "Clean complete."
}

# ------------------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------------------
ACTION="${1:-help}"
case "${ACTION}" in
    build)
        cmd_build
        ;;
    qemu)
        shift || true
        cmd_qemu "${1:-${PORT}}"
        ;;
    bench)
        shift || true
        cmd_bench "${1:-${PORT}}"
        ;;
    clean)
        cmd_clean
        ;;
    *)
        echo "Usage: $0 {build|qemu [PORT]|bench [PORT]|clean}"
        echo "Examples:"
        echo "  $0 build          # Compiles bare-metal Cortex-M3 firmware"
        echo "  $0 qemu 4001      # Runs QEMU exposing serial at tcp://127.0.0.1:4001"
        echo "  $0 bench 4001     # Runs automated 8-dataset benchmark suite"
        echo "  $0 clean          # Removes build directory"
        exit 1
        ;;
esac
