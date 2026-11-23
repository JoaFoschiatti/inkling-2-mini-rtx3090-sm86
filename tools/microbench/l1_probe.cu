// Sonda directa de coherencia de la L1 (RTX 3090, SM 8.6):
//   - todos los bloques lectores leen una linea global (queda en la L1 de su SM),
//   - el bloque 0 espera un poco y la sobreescribe (store normal),
//   - los lectores esperan mucho mas (bucle sin accesos a memoria) y vuelven a leer.
// Modos de la segunda lectura:
//   0: lectura normal (LDG)                      -> si la L1 no es coherente, deberia verse el valor VIEJO
//   1: __threadfence() y luego lectura normal    (lo que hace el hilo 0 del "ultimo bloque")
//   2: atomicAdd(dummy, 0) y luego lectura normal
//   3: __ldcg (salta la L1)                      -> control: debe ver siempre el valor NUEVO
//   4: como 1, pero el __threadfence() lo hace OTRO warp del mismo bloque (como en los kernels: fence solo en hilo 0)
//   5: sin lectura previa (no hay linea en L1) + lectura normal -> control
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

static __device__ __forceinline__ unsigned spin(unsigned x, const int iters) {
    for (int i = 0; i < iters; ++i) {
        x = x * 1664525u + 1013904223u;
        x ^= x >> 13;
    }
    return x;
}

template <int MODE>
__global__ void k_probe(float * line, int * dummy, int * n_old, int * n_new, unsigned * sink,
                        const float v_old, const float v_new, const int spin_w, const int spin_r) {
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    unsigned h = blockIdx.x * 977u + lane;
    if (blockIdx.x == 0) {
        // escritor
        h = spin(h, spin_w);
        if (warp == 0 && lane < 32) {
            line[lane] = v_new;
        }
        if (lane == 0 && warp == 0) sink[0] = h;
        return;
    }
    float first = 0.0f;
    if (MODE != 5) {
        first = line[lane]; // deja la linea en la L1 de este SM
    }
    h = spin(h + (unsigned) first, spin_r);
    __shared__ int s_dummy;
    if (MODE == 1) {
        if (lane == 0 && warp == 0) __threadfence();
    } else if (MODE == 2) {
        if (lane == 0 && warp == 0) atomicAdd(dummy, 0);
    } else if (MODE == 4) {
        if (lane == 0 && warp == 1) __threadfence();
    }
    if (lane == 0) s_dummy = (int) h;
    __syncthreads();
    if (warp != 0) {
        return;
    }
    const float second = (MODE == 3) ? __ldcg(&line[lane]) : line[lane];
    const unsigned m_old = __ballot_sync(0xffffffff, second == v_old);
    const unsigned m_new = __ballot_sync(0xffffffff, second == v_new);
    if (lane == 0) {
        if (m_old) atomicAdd(n_old, __popc(m_old));
        if (m_new) atomicAdd(n_new, __popc(m_new));
        sink[blockIdx.x] = h + s_dummy;
    }
}

int main(int argc, char ** argv) {
    const int n_iter  = argc > 1 ? atoi(argv[1]) : 300;
    const int spin_w  = argc > 2 ? atoi(argv[2]) : 20000;
    const int spin_r  = argc > 3 ? atoi(argv[3]) : 400000;
    const int n_blocks = 164;
    float * d_line; int * d_dummy; int * d_old; int * d_new; unsigned * d_sink;
    cudaMalloc(&d_line, 4096);
    cudaMalloc(&d_dummy, sizeof(int));
    cudaMalloc(&d_old, sizeof(int));
    cudaMalloc(&d_new, sizeof(int));
    cudaMalloc(&d_sink, n_blocks * sizeof(unsigned));
    cudaMemset(d_dummy, 0, sizeof(int));
    const char * names[6] = { "0 lectura normal", "1 __threadfence (hilo 0) + lectura normal", "2 atomicAdd + lectura normal",
                              "3 __ldcg (control)", "4 __threadfence en otro warp + lectura normal", "5 sin lectura previa (control)" };
    for (int mode = 0; mode < 6; ++mode) {
        long tot_old = 0, tot_new = 0, tot_other = 0;
        for (int it = 0; it < n_iter; ++it) {
            const float v_old = (float) (2 * it + 1);
            const float v_new = (float) (2 * it + 2);
            float init[32];
            for (int i = 0; i < 32; ++i) init[i] = v_old;
            cudaMemcpy(d_line, init, sizeof(init), cudaMemcpyHostToDevice);
            cudaMemset(d_old, 0, sizeof(int));
            cudaMemset(d_new, 0, sizeof(int));
            const dim3 bd(32, 2, 1);
            switch (mode) {
                case 0: k_probe<0><<<n_blocks, bd>>>(d_line, d_dummy, d_old, d_new, d_sink, v_old, v_new, spin_w, spin_r); break;
                case 1: k_probe<1><<<n_blocks, bd>>>(d_line, d_dummy, d_old, d_new, d_sink, v_old, v_new, spin_w, spin_r); break;
                case 2: k_probe<2><<<n_blocks, bd>>>(d_line, d_dummy, d_old, d_new, d_sink, v_old, v_new, spin_w, spin_r); break;
                case 3: k_probe<3><<<n_blocks, bd>>>(d_line, d_dummy, d_old, d_new, d_sink, v_old, v_new, spin_w, spin_r); break;
                case 4: k_probe<4><<<n_blocks, bd>>>(d_line, d_dummy, d_old, d_new, d_sink, v_old, v_new, spin_w, spin_r); break;
                case 5: k_probe<5><<<n_blocks, bd>>>(d_line, d_dummy, d_old, d_new, d_sink, v_old, v_new, spin_w, spin_r); break;
            }
            if (cudaDeviceSynchronize() != cudaSuccess) { printf("error CUDA\n"); return 1; }
            int o = 0, n = 0;
            cudaMemcpy(&o, d_old, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&n, d_new, sizeof(int), cudaMemcpyDeviceToHost);
            tot_old += o; tot_new += n; tot_other += (long) (n_blocks - 1) * 32 - o - n;
        }
        const double tot = (double) (tot_old + tot_new + tot_other);
        printf("modo %-46s VIEJO=%6.2f%%  NUEVO=%6.2f%%  otro=%ld\n", names[mode], 100.0 * tot_old / tot, 100.0 * tot_new / tot, tot_other);
    }
    return 0;
}
