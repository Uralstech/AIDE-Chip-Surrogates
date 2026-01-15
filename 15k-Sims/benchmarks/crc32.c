/*
 * CRC32 Checksum Calculation (1MB data)
 * Tests: Bitwise operations, table lookups, sequential memory access
 * Expected runtime: ~1.5 hours on O3CPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define POLY 0xEDB88320
#define DATA_SIZE 1048576  // 1MB

static uint32_t crc32_table[256];
static uint8_t data[DATA_SIZE];

// Initialize CRC32 lookup table
void init_crc32_table(void) {
    uint32_t i, j;
    uint32_t crc;
    
    for (i = 0; i < 256; i++) {
        crc = i;
        for (j = 0; j < 8; j++) {
            if (crc & 1) {
                crc = (crc >> 1) ^ POLY;
            } else {
                crc >>= 1;
            }
        }
        crc32_table[i] = crc;
    }
}

// Calculate CRC32
uint32_t calculate_crc32(const uint8_t *buf, size_t length) {
    uint32_t crc = 0xFFFFFFFF;
    size_t i;
    uint8_t index;
    
    for (i = 0; i < length; i++) {
        index = (crc ^ buf[i]) & 0xFF;
        crc = (crc >> 8) ^ crc32_table[index];
    }
    
    return crc ^ 0xFFFFFFFF;
}

int main(void) {
    size_t i;
    uint32_t result;
    
    // Initialize data with pattern
    for (i = 0; i < DATA_SIZE; i++) {
        data[i] = (uint8_t)(i & 0xFF);
    }
    
    // Initialize CRC table
    init_crc32_table();
    
    // Calculate CRC32
    result = calculate_crc32(data, DATA_SIZE);
    
    printf("CRC32 calculation complete\n");
    printf("Data size: %i bytes\n", DATA_SIZE);
    printf("CRC32: 0x%08X\n", result);
    
    return 0;
}