#!/usr/bin/env bash
# ==============================================================================
# run_qemu.sh — Build and Emulate ESP32 Hoeffding Tree Firmware in QEMU
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
FLASH_IMG="${BUILD_DIR}/flash_image.bin"
PORT="${QEMU_SERIAL_PORT:-4000}"

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
# Environment Detection (Native ESP-IDF vs Docker)
# ------------------------------------------------------------------------------
setup_env() {
    if command -v idf.py >/dev/null 2>&1; then
        log_info "Found idf.py in PATH: $(which idf.py)"
        USE_DOCKER=0
        return 0
    fi

    # Check common IDF install locations
    for export_script in \
        "${HOME}/esp/esp-idf/export.sh" \
        "${HOME}/esp-idf/export.sh" \
        "/opt/esp-idf/export.sh" \
        "${HOME}/.espressif/esp-idf/export.sh"; do
        if [ -f "${export_script}" ]; then
            log_info "Sourcing ESP-IDF environment from ${export_script}..."
            # shellcheck disable=SC1090
            . "${export_script}" >/dev/null 2>&1 || true
            if command -v idf.py >/dev/null 2>&1; then
                USE_DOCKER=0
                return 0
            fi
        fi
    done

    # Fallback: Docker
    if command -v docker >/dev/null 2>&1; then
        log_warn "idf.py not found in local environment. Using Docker fallback (espressif/idf)..."
        USE_DOCKER=1
        return 0
    fi

    log_warn "Neither native ESP-IDF nor Docker were found in PATH."
    USE_DOCKER=0
}

USE_DOCKER=0
setup_env

# ------------------------------------------------------------------------------
# Subcommands
# ------------------------------------------------------------------------------
cmd_build() {
    log_info "Building ESP32 firmware in ${SCRIPT_DIR}..."
    cd "${SCRIPT_DIR}"

    if [ "${USE_DOCKER}" -eq 1 ]; then
        docker run --rm -v "${PROJECT_ROOT}:/project" -w "/project/emulation/esp32" \
            espressif/idf:v5.1.2 idf.py build
    elif command -v idf.py >/dev/null 2>&1; then
        idf.py build
    else
        log_error "Cannot build: idf.py not found. Please install ESP-IDF or run with Docker."
        log_info "Quick local bootstrap: git clone --recursive https://github.com/espressif/esp-idf.git ~/esp/esp-idf && cd ~/esp/esp-idf && ./install.sh esp32 && . ./export.sh"
        exit 1
    fi
    log_success "ESP32 Firmware built successfully."
}

cmd_flash_image() {
    cd "${SCRIPT_DIR}"
    if [ ! -f "${BUILD_DIR}/esp32_hoeffding_tree.bin" ]; then
        log_warn "Binary not found. Triggering build first..."
        cmd_build
    fi

    log_info "Merging bootloader, partition table, and app into 4MB flash image (${FLASH_IMG})..."
    if [ "${USE_DOCKER}" -eq 1 ]; then
        docker run --rm -v "${PROJECT_ROOT}:/project" -w "/project/emulation/esp32" \
            espressif/idf:v5.1.2 esptool.py --chip esp32 merge_bin \
            --fill-flash-size 4MB \
            -o build/flash_image.bin \
            0x1000 build/bootloader/bootloader.bin \
            0x8000 build/partition_table/partition-table.bin \
            0x10000 build/esp32_hoeffding_tree.bin
    elif command -v esptool.py >/dev/null 2>&1; then
        esptool.py --chip esp32 merge_bin \
            --fill-flash-size 4MB \
            -o "${FLASH_IMG}" \
            0x1000 "${BUILD_DIR}/bootloader/bootloader.bin" \
            0x8000 "${BUILD_DIR}/partition_table/partition-table.bin" \
            0x10000 "${BUILD_DIR}/esp32_hoeffding_tree.bin"
    elif command -v python3 >/dev/null 2>&1; then
        python3 -m esptool --chip esp32 merge_bin \
            --fill-flash-size 4MB \
            -o "${FLASH_IMG}" \
            0x1000 "${BUILD_DIR}/bootloader/bootloader.bin" \
            0x8000 "${BUILD_DIR}/partition_table/partition-table.bin" \
            0x10000 "${BUILD_DIR}/esp32_hoeffding_tree.bin"
    else
        log_error "esptool.py not found to merge binary!"
        exit 1
    fi
    log_success "Flash image created: ${FLASH_IMG} ($(du -h "${FLASH_IMG}" | cut -f1))"
}

cmd_qemu() {
    local target_port="${1:-${PORT}}"
    if [ ! -f "${FLASH_IMG}" ]; then
        cmd_flash_image
    fi

    if [ "${USE_DOCKER}" -eq 1 ]; then
        log_info "Starting ESP32 QEMU via Docker container on port ${target_port}..."
        docker run --rm -it \
            -v "${PROJECT_ROOT}:/project" \
            -w "/project/emulation/esp32" \
            -p "${target_port}:${target_port}" \
            espressif/idf:v5.1.2 \
            /opt/qemu/bin/qemu-system-xtensa \
                -nographic \
                -machine esp32 \
                -drive file=/project/emulation/esp32/build/flash_image.bin,if=mtd,format=raw \
                -nic none \
                -serial "tcp:0.0.0.0:${target_port},server,nowait"
        return 0
    fi

    local qemu_bin=""
    if command -v qemu-system-xtensa >/dev/null 2>&1; then
        qemu_bin="qemu-system-xtensa"
    elif [ -f "${HOME}/.espressif/tools/qemu-xtensa/bin/qemu-system-xtensa" ]; then
        qemu_bin="${HOME}/.espressif/tools/qemu-xtensa/bin/qemu-system-xtensa"
    fi

    if [ -z "${qemu_bin}" ]; then
        log_error "qemu-system-xtensa not found!"
        log_info "To install Espressif's QEMU: esp-idf installation includes it or download from https://github.com/espressif/qemu/releases"
        exit 1
    fi

    log_info "Starting ESP32 QEMU on serial TCP port ${target_port}..."
    "${qemu_bin}" \
        -nographic \
        -machine esp32 \
        -drive file="${FLASH_IMG}",if=mtd,format=raw \
        -nic none \
        -serial "tcp:127.0.0.1:${target_port},server,nowait"
}

cmd_bench() {
    local target_port="${1:-${PORT}}"
    if [ ! -f "${FLASH_IMG}" ]; then
        cmd_flash_image
    fi

    local QEMU_PID=""
    if [ "${USE_DOCKER}" -eq 1 ]; then
        log_info "Launching background QEMU container (esp32_qemu_bench) on port ${target_port}..."
        docker rm -f esp32_qemu_bench >/dev/null 2>&1 || true
        docker run -d --rm \
            --name esp32_qemu_bench \
            -v "${PROJECT_ROOT}:/project" \
            -w "/project/emulation/esp32" \
            -p "${target_port}:${target_port}" \
            espressif/idf:v5.1.2 \
            /opt/qemu/bin/qemu-system-xtensa \
                -nographic \
                -machine esp32 \
                -drive file=/project/emulation/esp32/build/flash_image.bin,if=mtd,format=raw \
                -nic none \
                -serial "tcp:0.0.0.0:${target_port},server,nowait" > "${BUILD_DIR}/qemu.log" 2>&1
        trap 'docker rm -f esp32_qemu_bench >/dev/null 2>&1 || true' EXIT
    else
        local qemu_bin=""
        if command -v qemu-system-xtensa >/dev/null 2>&1; then
            qemu_bin="qemu-system-xtensa"
        elif [ -f "${HOME}/.espressif/tools/qemu-xtensa/bin/qemu-system-xtensa" ]; then
            qemu_bin="${HOME}/.espressif/tools/qemu-xtensa/bin/qemu-system-xtensa"
        fi

        if [ -z "${qemu_bin}" ]; then
            log_error "qemu-system-xtensa not found in PATH or ~/.espressif!"
            exit 1
        fi

        log_info "Launching background QEMU instance on port ${target_port}..."
        "${qemu_bin}" \
            -nographic \
            -machine esp32 \
            -drive file="${FLASH_IMG}",if=mtd,format=raw \
            -nic none \
            -serial "tcp:127.0.0.1:${target_port},server,nowait" > "${BUILD_DIR}/qemu.log" 2>&1 &
        QEMU_PID=$!
        trap 'kill -9 ${QEMU_PID} 2>/dev/null || true' EXIT
    fi

    log_info "Waiting for QEMU socket to open on port ${target_port}..."
    sleep 6

    PYTHON_BIN="python3"
    if [ -x "${PROJECT_ROOT}/.venv/bin/python" ]; then
        PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
    fi

    local max_samp="${MAX_SAMPLES:-20000}"
    local extra_args=()
    if [ "${max_samp}" != "0" ]; then
        extra_args+=(--max-samples "${max_samp}")
    fi

    log_info "Running ESP32 Stream Benchmark Runner via ${PYTHON_BIN}..."
    "${PYTHON_BIN}" "${SCRIPT_DIR}/stream_runner.py" \
        --host 127.0.0.1 \
        --port "${target_port}" \
        --suite main \
        --report "${SCRIPT_DIR}/esp32_emulation_report.md" \
        ${extra_args[@]+"${extra_args[@]}"}

    log_success "Benchmark completed! Report saved in ${SCRIPT_DIR}/esp32_emulation_report.md"
}

cmd_clean() {
    log_info "Cleaning ESP32 build artifacts..."
    rm -rf "${BUILD_DIR}" "${SCRIPT_DIR}/sdkconfig"
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
    flash-image)
        cmd_flash_image
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
        echo "Usage: $0 {build|flash-image|qemu [PORT]|bench [PORT]|clean}"
        echo "Examples:"
        echo "  $0 build          # Compiles the firmware with ESP-IDF"
        echo "  $0 flash-image    # Merges bootloader, partition table & app into 4MB image"
        echo "  $0 qemu 4000      # Runs QEMU exposing serial at tcp://127.0.0.1:4000"
        echo "  $0 bench 4000     # Runs automated 8-dataset benchmark suite"
        echo "  $0 clean          # Removes build directory and sdkconfig"
        exit 1
        ;;
esac
