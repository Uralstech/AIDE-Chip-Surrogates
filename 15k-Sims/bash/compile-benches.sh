export BENCH_SHA="sha"
export BENCH_QSORT="qsort"
export BENCH_MATRIXMUL="matrix_mul"
export BENCH_FFT="fft"
export BENCH_DIJKSTRA="dijkstra"
export BENCH_CRC32="crc32"

mkdir /home/ubuntu/compiled
riscv64-linux-gnu-gcc -O3 -static -march=rv64gc /home/ubuntu/benchmarks/$BENCH_SHA.c -o /home/ubuntu/compiled/$BENCH_SHA.riscv
riscv64-linux-gnu-gcc -O3 -static -march=rv64gc /home/ubuntu/benchmarks/$BENCH_QSORT.c -o /home/ubuntu/compiled/$BENCH_QSORT.riscv
riscv64-linux-gnu-gcc -O3 -static -march=rv64gc /home/ubuntu/benchmarks/$BENCH_MATRIXMUL.c -o /home/ubuntu/compiled/$BENCH_MATRIXMUL.riscv
riscv64-linux-gnu-gcc -O3 -static -march=rv64gc /home/ubuntu/benchmarks/$BENCH_FFT.c -o /home/ubuntu/compiled/$BENCH_FFT.riscv
riscv64-linux-gnu-gcc -O3 -static -march=rv64gc /home/ubuntu/benchmarks/$BENCH_DIJKSTRA.c -o /home/ubuntu/compiled/$BENCH_DIJKSTRA.riscv
riscv64-linux-gnu-gcc -O3 -static -march=rv64gc /home/ubuntu/benchmarks/$BENCH_CRC32.c -o /home/ubuntu/compiled/$BENCH_CRC32.riscv
