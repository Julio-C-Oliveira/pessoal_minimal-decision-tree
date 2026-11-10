/**
 * @file startup_lm3s6965.c
 * @brief Bare-metal startup code and ARMv7-M vector table for TI Stellaris LM3S6965.
 */

#include <stdint.h>
#include "uart.h"
#include "cycles.h"

/* Symbols defined in linker script lm3s6965.ld */
extern uint32_t _estack;
extern uint32_t _sidata;
extern uint32_t _sdata;
extern uint32_t _edata;
extern uint32_t _sbss;
extern uint32_t _ebss;

/* Application main */
extern int main(void);

/* Interrupt Handlers */
void Reset_Handler(void);
void Default_Handler(void);
void SysTick_Handler(void);

/* Exception Handler Aliases */
void NMI_Handler(void)        __attribute__((weak, alias("Default_Handler")));
void HardFault_Handler(void)  __attribute__((weak, alias("Default_Handler")));
void MemManage_Handler(void)  __attribute__((weak, alias("Default_Handler")));
void BusFault_Handler(void)   __attribute__((weak, alias("Default_Handler")));
void UsageFault_Handler(void) __attribute__((weak, alias("Default_Handler")));
void SVC_Handler(void)        __attribute__((weak, alias("Default_Handler")));
void DebugMon_Handler(void)   __attribute__((weak, alias("Default_Handler")));
void PendSV_Handler(void)     __attribute__((weak, alias("Default_Handler")));

/* Vector table entry union for strict ISO C11 pedantic compliance */
typedef union {
    void *stack_ptr;
    void (*handler)(void);
} isr_vector_t;

/* Vector Table placed in .vectors section at 0x00000000 */
__attribute__((section(".vectors"), used))
const isr_vector_t g_pfnVectors[] = {
    { .stack_ptr = &_estack },        /* Top of Stack (0x20010000) */
    { .handler = Reset_Handler },     /* Reset Handler */
    { .handler = NMI_Handler },       /* NMI Handler */
    { .handler = HardFault_Handler }, /* Hard Fault Handler */
    { .handler = MemManage_Handler }, /* MPU Fault Handler */
    { .handler = BusFault_Handler },  /* Bus Fault Handler */
    { .handler = UsageFault_Handler },/* Usage Fault Handler */
    { .handler = 0 }, { .handler = 0 }, { .handler = 0 }, { .handler = 0 },
    { .handler = SVC_Handler },       /* SVCall Handler */
    { .handler = DebugMon_Handler },  /* Debug Monitor Handler */
    { .handler = 0 },                 /* Reserved */
    { .handler = PendSV_Handler },    /* PendSV Handler */
    { .handler = SysTick_Handler },   /* SysTick Handler */
};

void Reset_Handler(void) {
    /* 1. Copy initialized data (.data) from Flash to SRAM */
    uint32_t *p_src = &_sidata;
    uint32_t *p_dst = &_sdata;
    while (p_dst < &_edata) {
        *p_dst++ = *p_src++;
    }

    /* 2. Zero-fill uninitialized data (.bss) in SRAM */
    p_dst = &_sbss;
    while (p_dst < &_ebss) {
        *p_dst++ = 0;
    }

    /* 3. Initialize hardware subsystems */
    uart_init();
    cycles_init();

    /* 4. Branch to main application loop */
    main();

    /* 5. Trap if main ever terminates */
    while (1) {
        __asm__ volatile ("wfi");
    }
}

void SysTick_Handler(void) {
    systick_isr();
}

void Default_Handler(void) {
    uart_send_str("\n[PANIC] Unhandled Hardware Fault/Interrupt!\n");
    while (1) {
        __asm__ volatile ("wfi");
    }
}
