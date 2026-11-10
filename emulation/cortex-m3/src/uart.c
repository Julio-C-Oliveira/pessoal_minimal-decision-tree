/**
 * @file uart.c
 * @brief Bare-metal UART0 driver implementation for TI Stellaris LM3S6965.
 *
 * Hardware peripheral base: 0x4000C000 (UART0)
 */

#include "uart.h"

#define UART0_BASE       0x4000C000UL

#define UART0_DR         (*(volatile uint32_t *)(UART0_BASE + 0x000))
#define UART0_RSR        (*(volatile uint32_t *)(UART0_BASE + 0x004))
#define UART0_FR         (*(volatile uint32_t *)(UART0_BASE + 0x018))
#define UART0_ILPR       (*(volatile uint32_t *)(UART0_BASE + 0x020))
#define UART0_IBRD       (*(volatile uint32_t *)(UART0_BASE + 0x024))
#define UART0_FBRD       (*(volatile uint32_t *)(UART0_BASE + 0x028))
#define UART0_LCRH       (*(volatile uint32_t *)(UART0_BASE + 0x02C))
#define UART0_CTL        (*(volatile uint32_t *)(UART0_BASE + 0x030))

/* Flag Register (FR) Bits */
#define UART_FR_TXFF     (1UL << 5)  /* Transmit FIFO Full */
#define UART_FR_RXFE     (1UL << 4)  /* Receive FIFO Empty */
#define UART_FR_BUSY     (1UL << 3)  /* UART Busy */

/* Line Control Register (LCRH) Bits */
#define UART_LCRH_WLEN_8 (3UL << 5)  /* 8 bits word length */
#define UART_LCRH_FEN    (1UL << 4)  /* Enable FIFOs */

/* Control Register (CTL) Bits */
#define UART_CTL_UARTEN  (1UL << 0)  /* UART Enable */
#define UART_CTL_TXE     (1UL << 8)  /* Transmit Enable */
#define UART_CTL_RXE     (1UL << 9)  /* Receive Enable */

void uart_init(void) {
    /* Disable UART0 before configuration */
    UART0_CTL = 0;

    /* Baud Rate configuration for 115200 @ 50 MHz CPU clock:
     * BRD = 50,000,000 / (16 * 115,200) = 27.126736
     * IBRD = 27
     * FBRD = round(0.126736 * 64) = 8
     */
    UART0_IBRD = 27;
    UART0_FBRD = 8;

    /* 8 bits, no parity, 1 stop bit, FIFOs enabled */
    UART0_LCRH = UART_LCRH_WLEN_8 | UART_LCRH_FEN;

    /* Enable UART0, TX, and RX */
    UART0_CTL = UART_CTL_UARTEN | UART_CTL_TXE | UART_CTL_RXE;
}

void uart_putc(char c) {
    /* Wait until transmit FIFO is not full */
    while (UART0_FR & UART_FR_TXFF) {
        /* Busy wait */
    }
    UART0_DR = (uint32_t)(uint8_t)c;
}

char uart_getc(void) {
    /* Wait until receive FIFO is not empty */
    while (UART0_FR & UART_FR_RXFE) {
        /* Busy wait */
    }
    return (char)(UART0_DR & 0xFF);
}

void uart_send_str(const char *str) {
    if (!str) return;
    while (*str) {
        if (*str == '\n') {
            uart_putc('\r');
        }
        uart_putc(*str++);
    }
}

int uart_read_line(char *buf, size_t max_len) {
    if (!buf || max_len == 0) return 0;
    size_t idx = 0;
    while (idx < max_len - 1) {
        char c = uart_getc();
        if (c == '\n') {
            break;
        } else if (c == '\r') {
            /* Skip carriage return */
            continue;
        }
        buf[idx++] = c;
    }
    buf[idx] = '\0';
    return (int)idx;
}
