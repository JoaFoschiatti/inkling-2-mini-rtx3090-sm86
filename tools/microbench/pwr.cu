// pwr: bandwidth and board power of read-bound kernels at several occupancies.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

struct __align__(4) block_q8_1 { half2 ds; int8_t qs[32]; };
struct __align__(2) block_q8_0 { half d; int8_t qs[32]; };

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static bool gpu_read(float & temp, float & power) {
    FILE * p = popen("nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv,noheader,nounits", "r");
    if (!p) { return false; }
    const int n = fscanf(p, "%f, %f", &temp, &power);
    pclose(p);
    return n == 2;
}

static void cool_to(float t0) {
    for (int i = 0; i < 300; i++) {
        float t = 0, p = 0;
        if (gpu_read(t, p) && t <= t0) { return; }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

// pure read, 16 B loads, each warp streams its own contiguous span
template <int SMEM_PAD>
__global__ void k_read16(const uint4 * __restrict__ p, const size_t n16, float * __restrict__ dst) {
    __shared__ uint32_t pad[SMEM_PAD > 0 ? SMEM_PAD : 1];
    const size_t n_warps = (size_t) gridDim.x * (blockDim.x / 32);
    const size_t warp    = (size_t) blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const size_t span    = n16 / n_warps;
    const size_t i0      = warp * span;
    const int    lane    = threadIdx.x & 31;
    uint32_t acc = 0;
    for (size_t i = i0 + lane; i < i0 + span; i += 32) {
        const uint4 v = __ldg(&p[i]);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (SMEM_PAD > 0 && threadIdx.x == 0) {
        pad[0] = acc;
        acc ^= pad[0];
    }
    if (acc == 0xdeadbeefu) {
        dst[0] = (float) acc;
    }
}

static __device__ __forceinline__ int get_int_b2(const void * x, const int i32) {
    const uint16_t * x16 = (const uint16_t *) x;
    return (int) __ldg(&x16[2 * i32 + 0]) | ((int) __ldg(&x16[2 * i32 + 1]) << 16);
}

// same row kernel as inkling_q8_0_mmv_f32<NWARPS, 72>
template <int NWARPS, int SMEM_PAD>
__global__ void k_q8_mmv(const block_q8_0 * __restrict__ w, const block_q8_1 * __restrict__ yq, float * __restrict__ dst, const int nrows) {
    __shared__ uint32_t pad[SMEM_PAD > 0 ? SMEM_PAD : 1];
    const int r = blockIdx.x * NWARPS + threadIdx.y;
    if (r >= nrows) {
        return;
    }
    const int tid = threadIdx.x;
    const block_q8_0 * xrow = w + (size_t) r * 72;
    const int iqs = (tid & 3) * 2;
    float sum = 0.0f;
#pragma unroll
    for (int k = tid >> 2; k < 72; k += 8) {
        const block_q8_0 * bx = &xrow[k];
        const block_q8_1 * by = &yq[k];
        const int * uy = (const int *) by->qs;
        int s = __dp4a(get_int_b2(bx->qs, iqs + 0), __ldg(uy + iqs + 0), 0);
        s     = __dp4a(get_int_b2(bx->qs, iqs + 1), __ldg(uy + iqs + 1), s);
        sum += (float) s * __half2float(__ldg(&bx->d)) * __low2float(by->ds);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, o, 32);
    }
    if (SMEM_PAD > 0 && tid == 0 && threadIdx.y == 0) {
        pad[0] = 1;
    }
    if (tid == 0) {
        dst[r] = sum;
    }
}

template <typename F>
static void run_cfg(const char * name, double bytes, F fn, double secs, float t0) {
    cool_to(t0);
    for (int i = 0; i < 20; i++) { fn(); }
    cudaDeviceSynchronize();
    const double ts = now_s();
    long iters = 0;
    float temp = 0, power = 0, psum = 0;
    int pn = 0;
    double tnext = ts + 1.5;
    while (true) {
        for (int i = 0; i < 50; i++) { fn(); }
        iters += 50;
        cudaDeviceSynchronize();
        const double t = now_s();
        if (t >= tnext) {
            if (gpu_read(temp, power)) { psum += power; pn++; }
            tnext = now_s() + 0.5;
        }
        if (t - ts >= secs) { break; }
    }
    const double te = now_s();
    // the power readback pauses the load for a moment, so time only gives a lower bound of GB/s
    const double us = (te - ts) * 1e6 / iters;
    printf("  %-34s %8.2f us  %6.1f GB/s  %5.0f W  T=%2.0fC  %5.2f GB/s/W\n",
           name, us, bytes / (us * 1e3), pn ? psum / pn : 0.0f, temp, pn ? (bytes / (us * 1e3)) / (psum / pn) : 0.0);
    fflush(stdout);
}

int main(int argc, char ** argv) {
    const double secs = argc > 1 ? atof(argv[1]) : 4.0;
    const float  t0   = argc > 2 ? (float) atof(argv[2]) : 55.0f;

    const size_t bytes = 512ull << 20;
    uint4 * buf; float * dst;
    cudaMalloc(&buf, bytes);
    cudaMalloc(&dst, 300000 * sizeof(float));
    cudaMemset(buf, 1, bytes);
    const size_t n16 = bytes / 16;

    printf("=== lectura pura 512 MiB, bloques de 256 hilos (warps residentes por SM variando) ===\n");
    // smem pad per block limits resident blocks per SM: 100 KB / pad
    run_cfg("read16 48 w/SM (6 blk/SM)", (double) bytes, [&]() { k_read16<0><<<4920, 256>>>(buf, n16, dst); }, secs, t0);
    run_cfg("read16 24 w/SM (3 blk/SM)", (double) bytes, [&]() { k_read16<7000><<<4920, 256>>>(buf, n16, dst); }, secs, t0);
    run_cfg("read16 16 w/SM (2 blk/SM)", (double) bytes, [&]() { k_read16<10000><<<4920, 256>>>(buf, n16, dst); }, secs, t0);
    run_cfg("read16  8 w/SM (1 blk/SM)", (double) bytes, [&]() { k_read16<16000><<<4920, 256>>>(buf, n16, dst); }, secs, t0);
    run_cfg("read16  4 w/SM (1 blk/SM,128t)", (double) bytes, [&]() { k_read16<16000><<<4920, 128>>>(buf, n16, dst); }, secs, t0);

    const int nrows = (int) (bytes / (72 * sizeof(block_q8_0)));
    block_q8_1 * yq;
    cudaMalloc(&yq, 72 * sizeof(block_q8_1));
    cudaMemset(yq, 1, 72 * sizeof(block_q8_1));
    const double qbytes = (double) nrows * 72 * sizeof(block_q8_0);
    printf("=== GEMV Q8_0 real (1 warp/fila), %d filas ===\n", nrows);
    run_cfg("q8 mmv 8w/blk 48 w/SM", qbytes, [&]() { k_q8_mmv<8, 0><<<(nrows + 7) / 8, dim3(32, 8)>>>((const block_q8_0 *) buf, yq, dst, nrows); }, secs, t0);
    run_cfg("q8 mmv 8w/blk 24 w/SM", qbytes, [&]() { k_q8_mmv<8, 7000><<<(nrows + 7) / 8, dim3(32, 8)>>>((const block_q8_0 *) buf, yq, dst, nrows); }, secs, t0);
    run_cfg("q8 mmv 8w/blk 16 w/SM", qbytes, [&]() { k_q8_mmv<8, 10000><<<(nrows + 7) / 8, dim3(32, 8)>>>((const block_q8_0 *) buf, yq, dst, nrows); }, secs, t0);
    run_cfg("q8 mmv 8w/blk  8 w/SM", qbytes, [&]() { k_q8_mmv<8, 16000><<<(nrows + 7) / 8, dim3(32, 8)>>>((const block_q8_0 *) buf, yq, dst, nrows); }, secs, t0);
    return 0;
}
