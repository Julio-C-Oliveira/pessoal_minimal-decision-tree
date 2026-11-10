cmake_minimum_required(VERSION 3.16)

# Inject Tier 8 Universal compile definitions for libht
add_compile_definitions(
    HT_MAX_NODES=128
    HT_MAX_ACTIVE_LEAVES=8
    HT_MAX_FEATURES=64
    HT_MAX_CLASSES=10
    HT_MAX_ADWIN_DETECTORS=32
    HT_REG_NUM_SLOTS=8
)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(esp32_hoeffding_tree)

