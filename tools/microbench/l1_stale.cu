// Prueba de coherencia de la cache L1 entre bloques con el patron "ultimo bloque" de inkling-fused.cuh:
//   cada bloque escribe sus 2 floats en scratch -> __threadfence() -> atomicAdd(cnt) -> el ultimo bloque lee los 128.
// Variante 0 (FIEL): igual que el router de 64 bloques, lecturas normales, mismo scratch reutilizado en cada lanzamiento.
// Variante 1 (SIEMBRA): antes de escribir, TODOS los bloques leen el scratch entero (deja lineas viejas en la L1 de cada SM).
// Variante 2 (SIEMBRA + __ldcg): como la 1 pero el ultimo bloque lee con __ldcg (salta la L1).
// Variante 3 (FIEL + trafico): como la 0 pero con un kernel intermedio que lee 64 MiB (como los GEMV entre routers).
// Cada lanzamiento usa valores distintos (seed), asi una lectura obsoleta se detecta exactamente.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

static __device__ __forceinline__ float val_of(const int seed, const int ex) {
    return (float) ((seed * 131 + ex * 7) & 0xFFFFF);
}

template <int MODE>
__global__ void k_last_block(float * scratch, int * cnt, int * n_bad, float * sink, const int seed) {
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    float pre = 0.0f;
    if (MODE == 1 || MODE == 2) {
        // siembra: cada hilo lee una entrada del scratch (valores del lanzamiento anterior)
        for (int i = warp * 32 + lane; i < 256; i += 128) {
            pre += scratch[i];
        }
    }
    __shared__ float s_pre[4];
    pre = pre + __shfl_xor_sync(0xffffffff, pre, 16);
    if (lane == 0) s_pre[warp] = pre;
    __syncthreads();

    if (warp < 2 && lane == 0) {
        const int ex = blockIdx.x * 2 + warp;
        scratch[ex]       = val_of(seed, ex);
        scratch[128 + ex] = val_of(seed, ex) + 0.5f;
    }
    __syncthreads();

    __shared__ bool is_last;
    if (warp == 0 && lane == 0) {
        __threadfence();
        const int done = atomicAdd(cnt, 1) + 1;
        is_last = (done == 64);
        if (is_last) {
            *cnt = 0;
        }
    }
    __syncthreads();
    if (!is_last || warp != 0) {
        if (lane == 0 && warp == 1) sink[blockIdx.x] = s_pre[0] + s_pre[1] + s_pre[2] + s_pre[3];
        return;
    }
    int bad = 0;
    for (int i = 0; i < 4; ++i) {
        const int ex = i * 32 + lane;
        float a, b;
        if (MODE == 2) {
            a = __ldcg(&scratch[ex]);
            b = __ldcg(&scratch[128 + ex]);
        } else {
            a = scratch[ex];
            b = scratch[128 + ex];
        }
        if (a != val_of(seed, ex))        bad++;
        if (b != val_of(seed, ex) + 0.5f) bad++;
    }
    if (bad) atomicAdd(n_bad, bad);
}

__global__ void k_traffic(const float * buf, const size_t n, float * out) {
    const size_t per = n / gridDim.x;
    const float * p = buf + (size_t) blockIdx.x * per;
    float acc = 0.0f;
    for (size_t i = threadIdx.x; i < per; i += blockDim.x) acc += p[i];
    if (threadIdx.x == 0) out[blockIdx.x] = acc;
}

int main(int argc, char ** argv) {
    const int n_launch = argc > 1 ? atoi(argv[1]) : 300000;
    float * d_scratch; int * d_cnt; int * d_bad; float * d_sink; float * d_big; float * d_tout;
    cudaMalloc(&d_scratch, 256 * sizeof(float));
    cudaMalloc(&d_cnt, sizeof(int));
    cudaMalloc(&d_bad, sizeof(int));
    cudaMalloc(&d_sink, 64 * sizeof(float));
    const size_t n_big = (size_t) 16 * 1024 * 1024; // 64 MiB de floats
    cudaMalloc(&d_big, n_big * sizeof(float));
    cudaMalloc(&d_tout, 1024 * sizeof(float));
    cudaMemset(d_big, 0, n_big * sizeof(float));
    printf("scratch=%p (alineacion mod 128 = %d)\n", (void *) d_scratch, (int) ((size_t) d_scratch & 127));

    const char * names[4] = { "0 FIEL (patron real, lectura normal)", "1 SIEMBRA (lectura previa + lectura normal)",
                              "2 SIEMBRA + __ldcg", "3 FIEL + 64 MiB de trafico entre lanzamientos" };
    for (int mode = 0; mode < 4; ++mode) {
        cudaMemset(d_scratch, 0, 256 * sizeof(float));
        cudaMemset(d_cnt, 0, sizeof(int));
        cudaMemset(d_bad, 0, sizeof(int));
        const int n = (mode == 3) ? n_launch / 20 : n_launch;
        for (int it = 1; it <= n; ++it) {
            switch (mode) {
                case 0: k_last_block<0><<<64, dim3(32, 4, 1)>>>(d_scratch, d_cnt, d_bad, d_sink, it); break;
                case 1: k_last_block<1><<<64, dim3(32, 4, 1)>>>(d_scratch, d_cnt, d_bad, d_sink, it); break;
                case 2: k_last_block<2><<<64, dim3(32, 4, 1)>>>(d_scratch, d_cnt, d_bad, d_sink, it); break;
                case 3:
                    k_traffic<<<1024, 256>>>(d_big, n_big, d_tout);
                    k_last_block<0><<<64, dim3(32, 4, 1)>>>(d_scratch, d_cnt, d_bad, d_sink, it);
                    break;
            }
        }
        if (cudaDeviceSynchronize() != cudaSuccess) { printf("error CUDA\n"); return 1; }
        int bad = 0, cnt = -1;
        cudaMemcpy(&bad, d_bad, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&cnt, d_cnt, sizeof(int), cudaMemcpyDeviceToHost);
        printf("variante %-48s lanzamientos=%7d  lecturas_obsoletas=%d  (contador final=%d)\n", names[mode], n, bad, cnt);
    }
    return 0;
}
