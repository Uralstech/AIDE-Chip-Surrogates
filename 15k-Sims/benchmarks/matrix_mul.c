/*
 * Matrix Multiplication Benchmark (256x256)
 * Tests: Memory hierarchy, cache performance, compute intensity
 * Expected runtime: ~2 hours on O3CPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define N 256

// Use static allocation to ensure data is in .data section
static int A[N][N];
static int B[N][N];
static int C[N][N];

int main(void) {
    int i, j, k;
    int checksum = 0;
    
    // Initialize matrices with deterministic pattern
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[i][j] = (i * N + j) % 100;
            B[i][j] = (j * N + i) % 100;
            C[i][j] = 0;
        }
    }
    
    // Matrix multiplication: C = A * B
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            int sum = 0;
            for (k = 0; k < N; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    
    // Compute checksum to prevent optimization
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            checksum += C[i][j];
        }
    }
    
    // Print result (prevents dead code elimination)
    printf("Matrix multiplication complete\n");
    printf("Checksum: %d\n", checksum);
    printf("Sample C[128][128]: %d\n", C[128][128]);
    
    return 0;
}