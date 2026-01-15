/*
 * Quicksort Benchmark (100,000 elements)
 * Tests: Recursive calls, branch prediction, memory access patterns
 * Expected runtime: ~1.5 hours on O3CPU
 * 
 * FIX for Issue #25: Proper stdlib qsort usage with error checking
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define ARRAY_SIZE 100000

static int array[ARRAY_SIZE];

// Comparison function for qsort
int compare_int(const void *a, const void *b) {
    int arg1 = *(const int*)a;
    int arg2 = *(const int*)b;
    
    if (arg1 < arg2) return -1;
    if (arg1 > arg2) return 1;
    return 0;
}

// Verify array is sorted
int verify_sorted(void) {
    int i;
    for (i = 0; i < ARRAY_SIZE - 1; i++) {
        if (array[i] > array[i + 1]) {
            return 0;  // Not sorted
        }
    }
    return 1;  // Sorted
}

int main(void) {
    int i;
    uint32_t seed = 12345;
    int64_t checksum = 0;
    
    // Initialize with pseudo-random values (deterministic)
    for (i = 0; i < ARRAY_SIZE; i++) {
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF;
        array[i] = (int)(seed % 1000000);
    }
    
    printf("Starting quicksort of %d elements\n", ARRAY_SIZE);
    
    // Sort using stdlib qsort
    qsort(array, ARRAY_SIZE, sizeof(int), compare_int);
    
    // Verify sorting
    if (!verify_sorted()) {
        fprintf(stderr, "ERROR: Array not properly sorted!\n");
        return 1;
    }
    
    // Compute checksum
    for (i = 0; i < ARRAY_SIZE; i++) {
        checksum += array[i];
    }
    
    printf("Quicksort complete\n");
    printf("Array sorted: YES\n");
    printf("First element: %d\n", array[0]);
    printf("Last element: %d\n", array[ARRAY_SIZE - 1]);
    printf("Median: %d\n", array[ARRAY_SIZE / 2]);
    printf("Checksum: %lld\n", (long long)checksum);
    
    return 0;
}

