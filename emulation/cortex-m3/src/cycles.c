/**
 * @file cycles.c
 * @brief High-resolution cycle timer implementation using ARM SysTick.
 */

#include "cycles.h"

#define SYSTICK_BASE     0xE000E010UL
#define SYSTICK_CSR      (*(volatile uint32_t *)(SYSTICK_BASE + 0x00))
#define SYSTICK_RVR      (*(volatile uint32_t *)(SYSTICK_BASE + 0x04))
#define SYSTICK_CVR      (*(volatile uint32_t *)(SYSTICK_BASE + 0x08))

#define SYSTICK_RELOAD   0x00FFFFFFUL

static volatile uint32_t systick_overflows = 0;

void cycles_init(void) {
    systick_overflows = 0;
    SYSTICK_CSR = 0;                          /* Disable timer */
    SYSTICK_RVR = SYSTICK_RELOAD;             /* 24-bit max value */
    SYSTICK_CVR = 0;                          /* Clear counter */
    SYSTICK_CSR = (1 << 0) | (1 << 1) | (1 << 2); /* ENABLE | TICKINT | CLKSOURCE (CPU clock) */
}

void systick_isr(void) {
    systick_overflows++;
}

uint64_t get_cpu_cycles(void) {
    uint32_t high1, low, high2;
    do {
        high1 = systick_overflows;
        low   = SYSTICK_CVR;
        high2 = systick_overflows;
    } while (high1 != high2);

    /* SysTick is a countdown timer: elapsed within current period is (RELOAD - CVR) */
    uint64_t cycles = ((uint64_t)high1 * (uint64_t)(SYSTICK_RELOAD + 1)) +
                      (uint64_t)(SYSTICK_RELOAD - low);
    return cycles;
}

double cycles_to_us(uint64_t cycles) {
    return (double)cycles / (double)CPU_FREQ_MHZ;
}
