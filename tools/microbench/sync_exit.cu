// Prueba: warps que hacen return antes de __syncthreads() (patron de los kernels MoE agrupados).
// Si la barrera esperara a los hilos que ya salieron, el kernel no terminaria nunca.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

template <int NWARPS>
__global__ void k_early_exit(const int n_rows, int * out, int * shared_seen) {
    const int row = blockIdx.x * NWARPS + threadIdx.y;
    if (row >= n_rows) {
        return; // warps enteros salen antes de la barrera
    }
    __shared__ int s_val;
    if (threadIdx.y == 0 && threadIdx.x == 0) {
        s_val = 1000 + blockIdx.x;
    }
    __syncthreads();
    const int v = s_val;
    __syncthreads();
    if (threadIdx.x == 0) {
        out[row] = v;
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(shared_seen, 1);
    }
}

int main() {
    constexpr int NW = 8;
    int bad = 0, runs = 0;
    for (int n_rows : { 2308, 2300, 1153, 1157, 9, 1, 2305 }) {
        const int n_blocks = (n_rows + NW - 1) / NW;
        int * d_out; int * d_seen;
        cudaMalloc(&d_out, sizeof(int) * n_blocks * NW);
        cudaMalloc(&d_seen, sizeof(int));
        for (int it = 0; it < 2000; ++it) {
            cudaMemset(d_out, 0xff, sizeof(int) * n_blocks * NW);
            cudaMemset(d_seen, 0, sizeof(int));
            k_early_exit<NW><<<n_blocks, dim3(32, NW, 1)>>>(n_rows, d_out, d_seen);
            if (cudaDeviceSynchronize() != cudaSuccess) { printf("error CUDA\n"); return 1; }
            runs++;
            if (it % 500 == 0) {
                int * h = (int *) malloc(sizeof(int) * n_blocks * NW);
                int seen = 0;
                cudaMemcpy(h, d_out, sizeof(int) * n_blocks * NW, cudaMemcpyDeviceToHost);
                cudaMemcpy(&seen, d_seen, sizeof(int), cudaMemcpyDeviceToHost);
                for (int r = 0; r < n_blocks * NW; ++r) {
                    const int expect = (r < n_rows) ? 1000 + r / NW : -1;
                    if (h[r] != expect) bad++;
                }
                if (seen != n_blocks) bad++;
                free(h);
            }
        }
        cudaFree(d_out); cudaFree(d_seen);
        printf("n_rows=%5d (resto %d de %d warps en el ultimo bloque): OK, sin cuelgue\n", n_rows, n_rows % NW, NW);
    }
    printf("lanzamientos=%d  resultados_incorrectos=%d\n", runs, bad);
    return bad != 0;
}
