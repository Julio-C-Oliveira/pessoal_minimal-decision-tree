/**
 * @file uart.h
 * @brief Bare-metal UART0 driver for TI Stellaris LM3S6965 (ARM Cortex-M3).
 */

#ifndef UART_H
#define UART_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

void uart_init(void);
void uart_putc(char c);
char uart_getc(void);
void uart_send_str(const char *str);
int  uart_read_line(char *buf, size_t max_len);

#endif /* UART_H */
