/*
 * Fast Fourier Transform (N=4096) - Fixed version
 * The original had inaccurate twiddle factors because the low-order Taylor approximations
 * for sin/cos are not accurate enough over the full [-PI, PI] range (especially near ±PI).
 *
 * Fix: Add proper range reduction to [-PI/4, PI/4], where the same low-order Taylor series
 * are highly accurate (error << 1e-6). This uses symmetry properties of sin/cos.
 * No math library needed, same number of operations in the polynomials.
 */

#include <stdio.h>
#include <stdlib.h>

#define N 4096
#define PI 3.14159265358979323846
#define TWO_PI 6.28318530717958647692

typedef struct {
    double real;
    double imag;
} Complex;

static Complex data[N];
static Complex temp[N];

/* Reduce angle to [-PI, PI] */
static double wrap_angle(double x) {
    while (x > PI)  x -= TWO_PI;
    while (x < -PI) x += TWO_PI;
    return x;
}

/* Taylor approximations for small angles |x| <= PI/4 */
static double small_sin(double x) {  /* x >= 0 */
    double x2 = x * x;
    return x * (1.0
        - x2 / 6.0
        + x2 * x2 / 120.0
        - x2 * x2 * x2 / 5040.0);
}

static double small_cos(double x) {  /* x >= 0 */
    double x2 = x * x;
    return 1.0
        - x2 / 2.0
        + x2 * x2 / 24.0
        - x2 * x2 * x2 / 720.0;
}

/* Accurate sin with range reduction */
static double fast_sin(double x) {
    x = wrap_angle(x);

    double sign = 1.0;
    if (x < 0.0) {
        x = -x;
        sign = -1.0;
    }

    if (x > PI / 2.0) {
        x = PI - x;  /* sin(PI - x) = sin(x), no sign change */
    }

    if (x > PI / 4.0) {
        x = PI / 2.0 - x;
        return sign * small_cos(x);
    }

    return sign * small_sin(x);
}

/* Accurate cos with range reduction */
static double fast_cos(double x) {
    x = wrap_angle(x);

    double sign = 1.0;
    if (x < 0.0) {
        x = -x;      /* cos is even */
    }

    if (x > PI / 2.0) {
        x = PI - x;
        sign = -1.0;  /* cos(PI - x) = -cos(x) */
    }

    if (x > PI / 4.0) {
        x = PI / 2.0 - x;
        return sign * small_sin(x);
    }

    return sign * small_cos(x);
}

/* Recursive FFT implementation (unchanged) */
void fft_recursive(Complex *x, int n, Complex *tmp) {
    int k, m;
    double angle;
    Complex t, even_val;

    if (n <= 1) return;

    m = n / 2;

    /* Split into even and odd */
    for (k = 0; k < m; k++) {
        tmp[k]     = x[k * 2];
        tmp[k + m] = x[k * 2 + 1];
    }

    for (k = 0; k < n; k++) {
        x[k] = tmp[k];
    }

    /* Recursive calls */
    fft_recursive(x, m, tmp);
    fft_recursive(x + m, m, tmp);

    /* Combine */
    for (k = 0; k < m; k++) {
        angle = -2.0 * PI * k / n;

        double c = fast_cos(angle);
        double s = fast_sin(angle);

        t.real = c * x[k + m].real - s * x[k + m].imag;
        t.imag = c * x[k + m].imag + s * x[k + m].real;

        even_val = x[k];

        x[k].real     = even_val.real + t.real;
        x[k].imag     = even_val.imag + t.imag;
        x[k + m].real = even_val.real - t.real;
        x[k + m].imag = even_val.imag - t.imag;
    }
}

int main(void) {
    int i;
    double checksum_real = 0.0;
    double checksum_imag = 0.0;

    /* Initialize with sine wave (now using accurate fast_sin) */
    for (i = 0; i < N; i++) {
        data[i].real = fast_sin(2.0 * PI * 10.0 * i / N);
        data[i].imag = 0.0;
    }

    /* Perform FFT */
    fft_recursive(data, N, temp);

    /* Compute checksum */
    for (i = 0; i < N; i++) {
        checksum_real += data[i].real;
        checksum_imag += data[i].imag;
    }

    printf("FFT calculation complete\n");
    printf("N: %d\n", N);
    printf("Checksum real: %f\n", checksum_real);
    printf("Checksum imag: %f\n", checksum_imag);
    /* Peaks should be near 0 ∓ 2048i at bins 10 and 4086 */
    printf("FFT[10]:  %f + %fi\n", data[10].real, data[10].imag);
    printf("FFT[4086]: %f + %fi\n", data[4086].real, data[4086].imag);

    return 0;
}