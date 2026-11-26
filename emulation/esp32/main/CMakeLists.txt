idf_component_register(
    SRCS
        "main.c"
        "../../../code/src/core/ht_memory.c"
        "../../../code/src/core/ht_observer.c"
        "../../../code/src/core/ht_criterion.c"
        "../../../code/src/core/ht_tree.c"
        "../../../code/src/core/ht_drift.c"
        "../../../code/src/core/ht_prep.c"
    INCLUDE_DIRS
        "."
        "../../../code/include"
        "../../../code/include/core"
        "../../../code/include/ensemble"
        "../../../code/include/federated"
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

