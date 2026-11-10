idf_component_register(
    SRCS
        "main.c"
        "../../../code/src/ht_memory.c"
        "../../../code/src/ht_observer.c"
        "../../../code/src/ht_criterion.c"
        "../../../code/src/ht_tree.c"
        "../../../code/src/ht_drift.c"
        "../../../code/src/ht_prep.c"
    INCLUDE_DIRS
        "."
        "../../../code/include"
    REQUIRES
        esp_hw_support
        esp_system
        freertos
        driver
)

target_compile_options(${COMPONENT_LIB} PRIVATE
    -std=gnu11
    -Wall
    -Wextra
    -Werror
)

