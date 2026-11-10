# ==============================================================================
# Dockerfile — ARM Cortex-M3 (TI Stellaris LM3S6965) Emulation Toolchain
# ==============================================================================
FROM debian:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc-arm-none-eabi \
    libnewlib-arm-none-eabi \
    binutils-arm-none-eabi \
    qemu-system-arm \
    make \
    python3 \
    procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /project
