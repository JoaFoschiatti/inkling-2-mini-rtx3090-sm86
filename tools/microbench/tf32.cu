// Prueba: cublasGemmEx(CUBLAS_COMPUTE_32F) con el handle en CUBLAS_TF32_TENSOR_OP_MATH (como common.cuh:1441)
// frente a CUBLAS_DEFAULT_MATH y CUBLAS_COMPUTE_32F_PEDANTIC, con las dimensiones de inkling_dspark_lr_get_or_build:
//   M[R x vc] = H[R x K] * W[K x vc], R=256, K=2304, vc=4096;  y  b[vc x 1] = W^T[vc x K] * bho[K x 1].
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

static double rel_err(const std::vector<float> & a, const std::vector<double> & ref, double * max_rel) {
    double se = 0.0, sr = 0.0, mx = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = (double) a[i] - ref[i];
        se += d * d; sr += ref[i] * ref[i];
        const double r = fabs(d) / (fabs(ref[i]) + 1e-30);
        if (fabs(ref[i]) > 1e-3 && r > mx) mx = r;
    }
    *max_rel = mx;
    return sqrt(se / sr);
}

int main() {
    const int R = 256, K = 2304, VC = 4096;
    std::vector<float> H((size_t) R * K), W((size_t) K * VC), B(K);
    srand(1234);
    auto rnd = []() { return (float) ((rand() / (double) RAND_MAX) * 2.0 - 1.0); };
    for (auto & x : H) x = rnd() * 0.05f;
    for (auto & x : W) x = roundf(rnd() * 127.0f) * 0.0123f; // como Q8_0 descuantizado
    for (auto & x : B) x = rnd();

    // referencia en double (CPU), solo una muestra de columnas para no tardar
    const int n_samp = 64;
    std::vector<double> refM((size_t) R * n_samp), refB(VC);
    for (int s = 0; s < n_samp; ++s) {
        const int v = (s * 61) % VC;
        for (int r = 0; r < R; ++r) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) acc += (double) H[(size_t) k * R + r] * (double) W[(size_t) v * K + k];
            refM[(size_t) s * R + r] = acc;
        }
    }
    for (int v = 0; v < VC; ++v) {
        double acc = 0.0;
        for (int k = 0; k < K; ++k) acc += (double) W[(size_t) v * K + k] * (double) B[k];
        refB[v] = acc;
    }

    float * dH; float * dW; float * dB; float * dM; float * dO;
    cudaMalloc(&dH, H.size() * 4); cudaMalloc(&dW, W.size() * 4); cudaMalloc(&dB, B.size() * 4);
    cudaMalloc(&dM, (size_t) R * VC * 4); cudaMalloc(&dO, (size_t) VC * 4);
    cudaMemcpy(dH, H.data(), H.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dW, W.data(), W.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), B.size() * 4, cudaMemcpyHostToDevice);

    cublasHandle_t h;
    cublasCreate(&h);
    const float alpha = 1.0f, beta = 0.0f;
    struct { const char * name; cublasMath_t math; cublasComputeType_t ct; } cases[] = {
        { "TF32_TENSOR_OP_MATH + COMPUTE_32F (lo que hace el codigo)", CUBLAS_TF32_TENSOR_OP_MATH, CUBLAS_COMPUTE_32F },
        { "DEFAULT_MATH        + COMPUTE_32F",                         CUBLAS_DEFAULT_MATH,        CUBLAS_COMPUTE_32F },
        { "TF32_TENSOR_OP_MATH + COMPUTE_32F_PEDANTIC (arreglo propuesto)", CUBLAS_TF32_TENSOR_OP_MATH, CUBLAS_COMPUTE_32F_PEDANTIC },
    };
    std::vector<float> hM((size_t) R * VC), hO(VC), sM((size_t) R * n_samp);
    for (auto & c : cases) {
        cublasSetMathMode(h, c.math);
        cublasStatus_t st1 = cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, R, VC, K, &alpha,
            dH, CUDA_R_32F, R, dW, CUDA_R_32F, K, &beta, dM, CUDA_R_32F, R, c.ct, CUBLAS_GEMM_DEFAULT);
        cublasStatus_t st2 = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, VC, 1, K, &alpha,
            dW, CUDA_R_32F, K, dB, CUDA_R_32F, K, &beta, dO, CUDA_R_32F, VC, c.ct, CUBLAS_GEMM_DEFAULT);
        cudaDeviceSynchronize();
        cudaMemcpy(hM.data(), dM, hM.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(hO.data(), dO, hO.size() * 4, cudaMemcpyDeviceToHost);
        for (int s = 0; s < n_samp; ++s) {
            const int v = (s * 61) % VC;
            for (int r = 0; r < R; ++r) sM[(size_t) s * R + r] = hM[(size_t) v * R + r];
        }
        double mx1 = 0, mx2 = 0;
        const double e1 = rel_err(sM, refM, &mx1);
        const double e2 = rel_err(hO, refB, &mx2);
        printf("%-64s st=%d/%d  M: err_rel_rms=%.3e  |  b_out: err_rel_rms=%.3e\n", c.name, (int) st1, (int) st2, e1, e2);
    }
    // referencia de escala: error de guardar M en FP16 (lo que hace el codigo despues)
    {
        double se = 0, sr = 0;
        for (size_t i = 0; i < refM.size(); ++i) {
            const float f = (float) refM[i];
            // redondeo a 11 bits de mantisa (half) aproximado
            int e; const float m = frexpf(f, &e);
            const float q = ldexpf(roundf(m * 2048.0f) / 2048.0f, e);
            se += ((double) q - refM[i]) * ((double) q - refM[i]); sr += refM[i] * refM[i];
        }
        printf("%-64s            M: err_rel_rms=%.3e\n", "(escala) redondeo de M exacta a FP16", sqrt(se / sr));
    }
    cublasDestroy(h);
    return 0;
}
