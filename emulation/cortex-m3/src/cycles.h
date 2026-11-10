/**
 * @file cycles.h
 * @brief High-resolution cycle timer based on ARM Cortex-M3 SysTick.
 */

#ifndef CYCLES_H
#define CYCLES_H

#include <stdint.h>

#define CPU_FREQ_HZ      50000000ULL
#define CPU_FREQ_MHZ     50ULL

void     cycles_init(void);
void     systick_isr(void);
uint64_t get_cpu_cycles(void);
double   cycles_to_us(uint64_t cycles);

#endif /* CYCLES_H */
