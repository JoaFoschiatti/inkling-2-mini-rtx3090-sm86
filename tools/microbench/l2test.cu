// l2test: does a read-only prefetch of the weights into L2 speed up the GEMV that follows?
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

struct __align__(4) block_q8_1 { half2 ds; int8_t qs[32]; };
struct __align__(2) block_q8_0 { half d; int8_t qs[32]; };

__global__ void k_read16(const uint4 * __restrict__ p, const size_t n16, float * __restrict__ dst) {
    const size_t n_warps = (size_t) gridDim.x * (blockDim.x / 32);
    const size_t warp    = (size_t) blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const size_t span    = (n16 + n_warps - 1) / n_warps;
    const size_t i0      = warp * span;
    const size_t i1      = (i0 + span < n16) ? i0 + span : n16;
    const int    lane    = threadIdx.x & 31;
    uint32_t acc = 0;
    for (size_t i = i0 + lane; i < i1; i += 32) {
        const uint4 v = __ldg(&p[i]);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0xdeadbeefu) {
        dst[0] = (float) acc;
    }
}

static __device__ __forceinline__ int get_int_b2(const void * x, const int i32) {
    const uint16_t * x16 = (const uint16_t *) x;
    return (int) __ldg(&x16[2 * i32 + 0]) | ((int) __ldg(&x16[2 * i32 + 1]) << 16);
}

template <int NWARPS>
__global__ void k_q8_mmv(const block_q8_0 * __restrict__ w, const block_q8_1 * __restrict__ yq, float * __restrict__ dst, const int nrows, const int nb) {
    const int r = blockIdx.x * NWARPS + threadIdx.y;
    if (r >= nrows) {
        return;
    }
    const int tid = threadIdx.x;
    const block_q8_0 * xrow = w + (size_t) r * nb;
    const int iqs = (tid & 3) * 2;
    float sum = 0.0f;
    for (int k = tid >> 2; k < nb; k += 8) {
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
    if (tid == 0) {
        dst[r] = sum;
    }
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s | SMs=%d | L2=%d KB | smem/SM=%zu KB\n", prop.name, prop.multiProcessorCount, prop.l2CacheSize / 1024, prop.sharedMemPerMultiprocessor / 1024);

    const size_t flush_bytes = 96ull << 20;
    uint4 * flush; float * dst; block_q8_1 * yq;
    cudaMalloc(&flush, flush_bytes);
    cudaMemset(flush, 1, flush_bytes);
    cudaMalloc(&dst, 16384 * sizeof(float));
    cudaMalloc(&yq, 96 * sizeof(block_q8_1));
    cudaMemset(yq, 1, 96 * sizeof(block_q8_1));

    struct mat { const char * name; int nrows; int nb; int nwarps; };
    const mat mats[] = {
        {"wo    [2304 x 3072] 7.5 MB", 2304, 96, 4},
        {"horiz [8600 x 2304] 21 MB ", 8600, 72, 8},
    };
    const double pre_mb[] = {0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0};

    for (const mat & m : mats) {
        const size_t wbytes = (size_t) m.nrows * m.nb * sizeof(block_q8_0);
        block_q8_0 * w;
        cudaMalloc(&w, wbytes + 64);
        cudaMemset(w, 1, wbytes);
        printf("=== %s ===\n", m.name);
        for (double pmb : pre_mb) {
            const size_t pbytes = (size_t) (pmb * 1e6) < wbytes ? (size_t) (pmb * 1e6) : wbytes;
            const int iters = 300;
            std::vector<cudaEvent_t> e0(iters), e1(iters), ep(iters);
            for (int i = 0; i < iters; i++) { cudaEventCreate(&e0[i]); cudaEventCreate(&e1[i]); cudaEventCreate(&ep[i]); }
            for (int i = 0; i < iters; i++) {
                // evict L2 with unrelated data
                k_read16<<<1968, 256>>>(flush, flush_bytes / 16, dst);
                cudaEventRecord(ep[i]);
                if (pbytes > 0) {
                    k_read16<<<328, 128>>>((const uint4 *) w, pbytes / 16, dst);
                }
                cudaEventRecord(e0[i]);
                if (m.nwarps == 4) {
                    k_q8_mmv<4><<<(m.nrows + 3) / 4, dim3(32, 4)>>>(w, yq, dst, m.nrows, m.nb);
                } else {
                    k_q8_mmv<8><<<(m.nrows + 7) / 8, dim3(32, 8)>>>(w, yq, dst, m.nrows, m.nb);
                }
                cudaEventRecord(e1[i]);
            }
            cudaDeviceSynchronize();
            double sum = 0.0, sump = 0.0;
            for (int i = 50; i < iters; i++) {
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, e0[i], e1[i]); sum += ms;
                cudaEventElapsedTime(&ms, ep[i], e0[i]); sump += ms;
            }
            const double us = sum * 1000.0 / (iters - 50), usp = sump * 1000.0 / (iters - 50);
            printf("  prefetch %5.1f MB (%6.2f us)  ->  GEMV %6.2f us  (%6.1f GB/s efectivos)\n", pbytes / 1e6, usp, us, wbytes / (us * 1e3));
            for (int i = 0; i < iters; i++) { cudaEventDestroy(e0[i]); cudaEventDestroy(e1[i]); cudaEventDestroy(ep[i]); }
        }
        cudaFree(w);
    }
    return 0;
}
