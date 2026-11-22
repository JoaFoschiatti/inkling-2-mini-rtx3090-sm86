#pragma once

#include "common.cuh"
#include "quantize.cuh"
#include "ggml.h"

#include <mutex>
#include <vector>

// ============================================================================
// Custom Fused CUDA Kernels for Inkling 2 Mini (SM 8.6 / Ampere Optimized)
// Eliminates ~5,000+ unfused micro-kernel launches & VRAM roundtrips per token
// ============================================================================

// 1. Fused L2 row clip (with optional trailing per-token tau multiplier!):
//    y = x * (scale ? scale : 1)
//    out = y * clamp(threshold / (sqrt(sum_rows(y^2)) * scale_s + scale_b), cmin, cmax) * (tau ? tau[tok] : 1)
static __global__ void inkling_fused_l2_clip_f32_warp(
        const float * __restrict__ x,
        const float * __restrict__ scale,
        const float * __restrict__ tau,
        float * __restrict__ dst,
        const int ne0,
        const int n_rows,
        const int scale_rows, // 0 if no scale, else ne1 (or ne1*ne2) of scale
        const int tau_div,    // x->ne[1] if tau != nullptr
        const float scale_s,
        const float scale_b,
        const float thr,
        const float cmin,
        const float cmax) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= n_rows) {
        return;
    }
    const int lane = threadIdx.x;
    const float * x_row = x + (size_t) row * ne0;
    float * dst_row = dst + (size_t) row * ne0;
    const float * sc_row = scale ? (scale + (size_t) (row % scale_rows) * ne0) : nullptr;
    const float tau_val = tau ? tau[row / tau_div] : 1.0f;

    if (ne0 <= 128) {
        // Fast single-pass register path for head_dim = 128 (Q, K) and n_relative = 16 (R)
        float v[4];
        float ss = 0.0f;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int c = lane + k * 32;
            if (c < ne0) {
                float val = x_row[c];
                if (sc_row) {
                    val *= sc_row[c];
                }
                v[k] = val;
                ss += val * val;
            }
        }
        ss = warp_reduce_sum<32>(ss);
        const float norm = sqrtf(ss) * scale_s + scale_b;
        const float fac  = fminf(cmax, fmaxf(cmin, thr / norm)) * tau_val;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int c = lane + k * 32;
            if (c < ne0) {
                dst_row[c] = v[k] * fac;
            }
        }
    } else {
        float ss = 0.0f;
        for (int c = lane; c < ne0; c += 32) {
            float val = x_row[c];
            if (sc_row) {
                val *= sc_row[c];
            }
            ss += val * val;
        }
        ss = warp_reduce_sum<32>(ss);
        const float norm = sqrtf(ss) * scale_s + scale_b;
        const float fac  = fminf(cmax, fmaxf(cmin, thr / norm)) * tau_val;
        for (int c = lane; c < ne0; c += 32) {
            float val = x_row[c];
            if (sc_row) {
                val *= sc_row[c];
            }
            dst_row[c] = val * fac;
        }
    }
}

static inline void ggml_cuda_op_inkling_l2_clip(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * x,
        const ggml_tensor * scale,
        ggml_tensor * dst,
        const float scale_s,
        const float scale_b,
        const float thr,
        const float cmin,
        const float cmax,
        const ggml_tensor * tau = nullptr) {
    const int ne0 = (int) x->ne[0];
    const int n_rows = (int) (x->ne[1] * x->ne[2] * x->ne[3]);
    const int scale_rows = scale ? (int) (scale->ne[1] * scale->ne[2] * scale->ne[3]) : 0;
    const int tau_div = (int) x->ne[1];

    constexpr int warps_per_block = 4;
    const dim3 block_dims(32, warps_per_block, 1);
    const dim3 block_nums((n_rows + warps_per_block - 1) / warps_per_block, 1, 1);

    inkling_fused_l2_clip_f32_warp<<<block_nums, block_dims, 0, ctx.stream()>>>(
        (const float *) x->data,
        scale ? (const float *) scale->data : nullptr,
        tau   ? (const float *) tau->data   : nullptr,
        (float *) dst->data,
        ne0, n_rows, scale_rows, tau_div, scale_s, scale_b, thr, cmin, cmax);
}

// 2. Fused SiTU activation (strided 3D views supported for gate_up split, float4 vectorized!):
//    act = beta * tanh(gate * inv_beta) * sigmoid(gate)
//    lin = has_lb ? (lb * tanh(up * inv_lb)) : up
//    dst = act * lin
static __device__ __forceinline__ float inkling_situ_elem(
        const float g, const float u,
        const float inv_beta, const float beta,
        const float inv_lb, const float lb,
        const bool has_lb) {
    const float act = beta * tanhf(g * inv_beta) * (1.0f / (1.0f + expf(-g)));
    const float lin = has_lb ? (lb * tanhf(u * inv_lb)) : u;
    return act * lin;
}

static __global__ void inkling_fused_situ_f32(
        const float * __restrict__ gate,
        const float * __restrict__ up,
        float * __restrict__ dst,
        const int ne0,
        const int ne1,
        const int ne2,
        const size_t g_s1, const size_t g_s2,
        const size_t u_s1, const size_t u_s2,
        const float inv_beta, const float beta,
        const float inv_lb, const float lb,
        const bool has_lb) {
    const int64_t total = (int64_t) ne0 * ne1 * ne2;
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const int i0 = (int) (idx % ne0);
    const int64_t tmp = idx / ne0;
    const int i1 = (int) (tmp % ne1);
    const int i2 = (int) (tmp / ne1);

    const float g = gate[(size_t) i2 * g_s2 + (size_t) i1 * g_s1 + i0];
    const float u = up[(size_t) i2 * u_s2 + (size_t) i1 * u_s1 + i0];
    dst[idx] = inkling_situ_elem(g, u, inv_beta, beta, inv_lb, lb, has_lb);
}

static __global__ void inkling_fused_situ_f32x4(
        const float4 * __restrict__ gate,
        const float4 * __restrict__ up,
        float4 * __restrict__ dst,
        const int ne0_4,
        const int ne1,
        const int ne2,
        const size_t g_s1_4, const size_t g_s2_4,
        const size_t u_s1_4, const size_t u_s2_4,
        const float inv_beta, const float beta,
        const float inv_lb, const float lb,
        const bool has_lb) {
    const int64_t total = (int64_t) ne0_4 * ne1 * ne2;
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const int i0 = (int) (idx % ne0_4);
    const int64_t tmp = idx / ne0_4;
    const int i1 = (int) (tmp % ne1);
    const int i2 = (int) (tmp / ne1);

    const float4 gv = gate[(size_t) i2 * g_s2_4 + (size_t) i1 * g_s1_4 + i0];
    const float4 uv = up[(size_t) i2 * u_s2_4 + (size_t) i1 * u_s1_4 + i0];
    dst[idx] = make_float4(
        inkling_situ_elem(gv.x, uv.x, inv_beta, beta, inv_lb, lb, has_lb),
        inkling_situ_elem(gv.y, uv.y, inv_beta, beta, inv_lb, lb, has_lb),
        inkling_situ_elem(gv.z, uv.z, inv_beta, beta, inv_lb, lb, has_lb),
        inkling_situ_elem(gv.w, uv.w, inv_beta, beta, inv_lb, lb, has_lb));
}

static inline void ggml_cuda_op_inkling_situ(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * gate,
        const ggml_tensor * up,
        ggml_tensor * dst,
        const float inv_beta, const float beta,
        const float inv_lb, const float lb,
        const bool has_lb) {
    const int ne0 = (int) dst->ne[0];
    const int ne1 = (int) dst->ne[1];
    const int ne2 = (int) (dst->ne[2] * dst->ne[3]);

    const size_t g_s1 = gate->nb[1] / sizeof(float);
    const size_t g_s2 = gate->nb[2] / sizeof(float);
    const size_t u_s1 = up->nb[1] / sizeof(float);
    const size_t u_s2 = up->nb[2] / sizeof(float);

    constexpr int threads = 256;
    if ((ne0 & 3) == 0 && (g_s1 & 3) == 0 && (g_s2 & 3) == 0 && (u_s1 & 3) == 0 && (u_s2 & 3) == 0 &&
            (((uintptr_t) gate->data | (uintptr_t) up->data | (uintptr_t) dst->data) & 15) == 0) {
        const int ne0_4 = ne0 >> 2;
        const int64_t total4 = (int64_t) ne0_4 * ne1 * ne2;
        const int blocks = (int) ((total4 + threads - 1) / threads);
        inkling_fused_situ_f32x4<<<blocks, threads, 0, ctx.stream()>>>(
            (const float4 *) gate->data, (const float4 *) up->data, (float4 *) dst->data,
            ne0_4, ne1, ne2, g_s1 >> 2, g_s2 >> 2, u_s1 >> 2, u_s2 >> 2,
            inv_beta, beta, inv_lb, lb, has_lb);
    } else {
        const int64_t total = (int64_t) ne0 * ne1 * ne2;
        const int blocks = (int) ((total + threads - 1) / threads);
        inkling_fused_situ_f32<<<blocks, threads, 0, ctx.stream()>>>(
            (const float *) gate->data, (const float *) up->data, (float *) dst->data,
            ne0, ne1, ne2, g_s1, g_s2, u_s1, u_s2, inv_beta, beta, inv_lb, lb, has_lb);
    }
}

// 3. Fused tanh + scale_bias + mul: dst = (tanh(x) * s + b) * y
static __global__ void inkling_fused_tanh_scale_mul_f32(
        const float * __restrict__ x,
        const float * __restrict__ y,
        float * __restrict__ dst,
        const int64_t n,
        const float s,
        const float b) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = (tanhf(x[idx]) * s + b) * y[idx];
    }
}

static __global__ void inkling_fused_tanh_scale_mul_f32x4(
        const float4 * __restrict__ x,
        const float4 * __restrict__ y,
        float4 * __restrict__ dst,
        const int64_t n4,
        const float s,
        const float b) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        const float4 xv = x[idx];
        const float4 yv = y[idx];
        dst[idx] = make_float4(
            (tanhf(xv.x) * s + b) * yv.x,
            (tanhf(xv.y) * s + b) * yv.y,
            (tanhf(xv.z) * s + b) * yv.z,
            (tanhf(xv.w) * s + b) * yv.w);
    }
}

static inline void ggml_cuda_op_inkling_tanh_scale_mul(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * x,
        const ggml_tensor * y,
        ggml_tensor * dst,
        const float s,
        const float b) {
    const int64_t n = ggml_nelements(dst);
    constexpr int threads = 256;
    if ((n & 3) == 0 && (((uintptr_t) x->data | (uintptr_t) y->data | (uintptr_t) dst->data) & 15) == 0) {
        const int64_t n4 = n >> 2;
        inkling_fused_tanh_scale_mul_f32x4<<<(int) ((n4 + threads - 1) / threads), threads, 0, ctx.stream()>>>(
            (const float4 *) x->data, (const float4 *) y->data, (float4 *) dst->data, n4, s, b);
    } else {
        inkling_fused_tanh_scale_mul_f32<<<(int) ((n + threads - 1) / threads), threads, 0, ctx.stream()>>>(
            (const float *) x->data, (const float *) y->data, (float *) dst->data, n, s, b);
    }
}

// 4. Fused tanh + mul + add: dst = lhs + tanh(x) * y
static __global__ void inkling_fused_tanh_mul_add_f32(
        const float * __restrict__ x,
        const float * __restrict__ y,
        const float * __restrict__ lhs,
        float * __restrict__ dst,
        const int64_t n) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = lhs[idx] + tanhf(x[idx]) * y[idx];
    }
}

static __global__ void inkling_fused_tanh_mul_add_f32x4(
        const float4 * __restrict__ x,
        const float4 * __restrict__ y,
        const float4 * __restrict__ lhs,
        float4 * __restrict__ dst,
        const int64_t n4) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        const float4 xv = x[idx];
        const float4 yv = y[idx];
        const float4 lv = lhs[idx];
        dst[idx] = make_float4(
            lv.x + tanhf(xv.x) * yv.x,
            lv.y + tanhf(xv.y) * yv.y,
            lv.z + tanhf(xv.z) * yv.z,
            lv.w + tanhf(xv.w) * yv.w);
    }
}

static inline void ggml_cuda_op_inkling_tanh_mul_add(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * x,
        const ggml_tensor * y,
        const ggml_tensor * lhs,
        ggml_tensor * dst) {
    const int64_t n = ggml_nelements(dst);
    constexpr int threads = 256;
    if ((n & 3) == 0 && (((uintptr_t) x->data | (uintptr_t) y->data | (uintptr_t) lhs->data | (uintptr_t) dst->data) & 15) == 0) {
        const int64_t n4 = n >> 2;
        inkling_fused_tanh_mul_add_f32x4<<<(int) ((n4 + threads - 1) / threads), threads, 0, ctx.stream()>>>(
            (const float4 *) x->data, (const float4 *) y->data, (const float4 *) lhs->data, (float4 *) dst->data, n4);
    } else {
        inkling_fused_tanh_mul_add_f32<<<(int) ((n + threads - 1) / threads), threads, 0, ctx.stream()>>>(
            (const float *) x->data, (const float *) y->data, (const float *) lhs->data, (float *) dst->data, n);
    }
}

// 5. Fused add + scale_bias: dst = (a + b) * s + bias
static __global__ void inkling_fused_add_scale_f32(
        const float * __restrict__ a,
        const float * __restrict__ b,
        float * __restrict__ dst,
        const int64_t n,
        const float s,
        const float bias) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = (a[idx] + b[idx]) * s + bias;
    }
}

static __global__ void inkling_fused_add_scale_f32x4(
        const float4 * __restrict__ a,
        const float4 * __restrict__ b,
        float4 * __restrict__ dst,
        const int64_t n4,
        const float s,
        const float bias) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        const float4 av = a[idx];
        const float4 bv = b[idx];
        dst[idx] = make_float4(
            (av.x + bv.x) * s + bias,
            (av.y + bv.y) * s + bias,
            (av.z + bv.z) * s + bias,
            (av.w + bv.w) * s + bias);
    }
}

static inline void ggml_cuda_op_inkling_add_scale(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * a,
        const ggml_tensor * b,
        ggml_tensor * dst,
        const float s,
        const float bias) {
    const int64_t n = ggml_nelements(dst);
    constexpr int threads = 256;
    if ((n & 3) == 0 && (((uintptr_t) a->data | (uintptr_t) b->data | (uintptr_t) dst->data) & 15) == 0) {
        const int64_t n4 = n >> 2;
        inkling_fused_add_scale_f32x4<<<(int) ((n4 + threads - 1) / threads), threads, 0, ctx.stream()>>>(
            (const float4 *) a->data, (const float4 *) b->data, (float4 *) dst->data, n4, s, bias);
    } else {
        inkling_fused_add_scale_f32<<<(int) ((n + threads - 1) / threads), threads, 0, ctx.stream()>>>(
            (const float *) a->data, (const float *) b->data, (float *) dst->data, n, s, bias);
    }
}

// ============================================================================
// Wave 3 Mega-Fusions
// ============================================================================

// 6. Fused 12-op loop controller:
//    h   = W_down * [rms_norm(cur); rms_norm(msg)]          ([2*D, R] x [2*D, T] -> [R, T])
//    out = (tanh(W_forget * h) * s + b) * cur + tanh(W_gain * h) * msg
//    Replaces 12 nodes (CONCAT, RMS_NORM, RESHAPE, 3x MUL_MAT, 2x TANH, SCALE, 2x MUL, ADD)
//    with 2 ultra-fast coalesced float4 kernels (zero intermediate [4608] / [2304] VRAM buffers).
static __global__ void inkling_fused_control_down_f32x4(
        const float4 * __restrict__ cur,
        const float4 * __restrict__ msg,
        const float4 * __restrict__ w_down,
        float * __restrict__ h_out,
        const int D4,
        const int R,
        const float inv_D,
        const float eps) {
    const int r = blockIdx.x;
    const int t = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const float4 * cur_t = cur + (size_t) t * D4;
    const float4 * msg_t = msg + (size_t) t * D4;
    const float4 * w_c   = w_down + (size_t) r * (2 * D4);
    const float4 * w_m   = w_c + D4;

    float ss_c = 0.0f, ss_m = 0.0f, dot_c = 0.0f, dot_m = 0.0f;
    for (int i = tid; i < D4; i += 256) {
        const float4 cv = cur_t[i];
        const float4 mv = msg_t[i];
        const float4 wc = w_c[i];
        const float4 wm = w_m[i];
        ss_c  += cv.x * cv.x + cv.y * cv.y + cv.z * cv.z + cv.w * cv.w;
        ss_m  += mv.x * mv.x + mv.y * mv.y + mv.z * mv.z + mv.w * mv.w;
        dot_c += cv.x * wc.x + cv.y * wc.y + cv.z * wc.z + cv.w * wc.w;
        dot_m += mv.x * wm.x + mv.y * wm.y + mv.z * wm.z + mv.w * wm.w;
    }

    ss_c  = warp_reduce_sum<32>(ss_c);
    ss_m  = warp_reduce_sum<32>(ss_m);
    dot_c = warp_reduce_sum<32>(dot_c);
    dot_m = warp_reduce_sum<32>(dot_m);

    __shared__ float smem[4][8];
    if (lane == 0) {
        smem[0][warp] = ss_c;
        smem[1][warp] = ss_m;
        smem[2][warp] = dot_c;
        smem[3][warp] = dot_m;
    }
    __syncthreads();

    if (warp == 0) {
        float v0 = (lane < 8) ? smem[0][lane] : 0.0f;
        float v1 = (lane < 8) ? smem[1][lane] : 0.0f;
        float v2 = (lane < 8) ? smem[2][lane] : 0.0f;
        float v3 = (lane < 8) ? smem[3][lane] : 0.0f;
        v0 = warp_reduce_sum<8>(v0);
        v1 = warp_reduce_sum<8>(v1);
        v2 = warp_reduce_sum<8>(v2);
        v3 = warp_reduce_sum<8>(v3);
        if (lane == 0) {
            const float inv_rms_c = rsqrtf(v0 * inv_D + eps);
            const float inv_rms_m = rsqrtf(v1 * inv_D + eps);
            h_out[(size_t) t * R + r] = v2 * inv_rms_c + v3 * inv_rms_m;
        }
    }
}

static __global__ void inkling_fused_control_up_f32x4(
        const float4 * __restrict__ h_in,
        const float4 * __restrict__ w_forget,
        const float4 * __restrict__ w_gain,
        const float * __restrict__ cur,
        const float * __restrict__ msg,
        float * __restrict__ dst,
        const int D,
        const int R4,
        const float forget_s,
        const float forget_b) {
    // 8 rows per block, 16 threads (half-warp) per row -> 128 threads/block, zero smem, zero syncthreads!
    const int lane16 = threadIdx.x & 15;
    const int row    = blockIdx.x * 8 + (threadIdx.x >> 4);
    const int t      = blockIdx.y;
    if (row >= D) {
        return;
    }

    const float4 * h4  = h_in     + (size_t) t * R4;
    const float4 * wf4 = w_forget + (size_t) row * R4;
    const float4 * wg4 = w_gain   + (size_t) row * R4;

    float df = 0.0f;
    float dg = 0.0f;
    for (int k = lane16; k < R4; k += 16) {
        const float4 hv = h4[k];
        const float4 wf = wf4[k];
        const float4 wg = wg4[k];
        df += wf.x * hv.x + wf.y * hv.y + wf.z * hv.z + wf.w * hv.w;
        dg += wg.x * hv.x + wg.y * hv.y + wg.z * hv.z + wg.w * hv.w;
    }

#pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        df += __shfl_xor_sync(0xffffffff, df, offset, 16);
        dg += __shfl_xor_sync(0xffffffff, dg, offset, 16);
    }

    if (lane16 == 0) {
        const size_t idx = (size_t) t * D + row;
        const float f = tanhf(df) * forget_s + forget_b;
        const float g = tanhf(dg);
        dst[idx] = f * cur[idx] + g * msg[idx];
    }
}

// 6b. Batched Fused 12-op loop controller (1 or 3 controllers, T = 2..16):
//     Reads w_down, w_forget, w_gain ONCE across all T tokens in registers!
struct inkling_ctrl_batched_ptrs {
    const float4 * cur[3];
    const float4 * msg[3];
    const float4 * wd[3];
    const float4 * wf[3];
    const float4 * wg[3];
    float *        dst[3];
};

template <int T>
static __global__ void inkling_fused_control_down_batched_f32x4(
        const inkling_ctrl_batched_ptrs p,
        float * __restrict__ scratch,
        const int D4,
        const int R) {
    constexpr int SPLIT_K = 2;
    const int r    = blockIdx.x;
    const int sp   = blockIdx.y;
    const int b    = blockIdx.z;
    const int tid  = threadIdx.x; // 0..127 (4 warps)
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const float4 * cur = (b == 0) ? p.cur[0] : ((b == 1) ? p.cur[1] : p.cur[2]);
    const float4 * msg = (b == 0) ? p.msg[0] : ((b == 1) ? p.msg[1] : p.msg[2]);
    const float4 * wd  = (b == 0) ? p.wd[0]  : ((b == 1) ? p.wd[1]  : p.wd[2]);

    const int chunk = (D4 + SPLIT_K - 1) / SPLIT_K;
    const int i0    = sp * chunk;
    const int i1    = min(D4, i0 + chunk);

    const float4 * wc = wd + (size_t) r * (2 * D4);
    const float4 * wm = wc + D4;

    float dc[T], dm[T];
    float ss_c[T], ss_m[T];
#pragma unroll
    for (int t = 0; t < T; ++t) {
        dc[t] = 0.0f; dm[t] = 0.0f; ss_c[t] = 0.0f; ss_m[t] = 0.0f;
    }

    for (int i = i0 + tid; i < i1; i += 128) {
        const float4 wcv = __ldg(&wc[i]);
        const float4 wmv = __ldg(&wm[i]);
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float4 cv = __ldg(&cur[(size_t) t * D4 + i]);
            const float4 mv = __ldg(&msg[(size_t) t * D4 + i]);
            dc[t] += cv.x * wcv.x + cv.y * wcv.y + cv.z * wcv.z + cv.w * wcv.w;
            dm[t] += mv.x * wmv.x + mv.y * wmv.y + mv.z * wmv.z + mv.w * wmv.w;
            if (r == 0) {
                ss_c[t] += cv.x * cv.x + cv.y * cv.y + cv.z * cv.z + cv.w * cv.w;
                ss_m[t] += mv.x * mv.x + mv.y * mv.y + mv.z * mv.z + mv.w * mv.w;
            }
        }
    }

    __shared__ float sm[4][4][T];
#pragma unroll
    for (int t = 0; t < T; ++t) {
        dc[t] = warp_reduce_sum<32>(dc[t]);
        dm[t] = warp_reduce_sum<32>(dm[t]);
        if (lane == 0) {
            sm[0][warp][t] = dc[t];
            sm[1][warp][t] = dm[t];
        }
        if (r == 0) {
            ss_c[t] = warp_reduce_sum<32>(ss_c[t]);
            ss_m[t] = warp_reduce_sum<32>(ss_m[t]);
            if (lane == 0) {
                sm[2][warp][t] = ss_c[t];
                sm[3][warp][t] = ss_m[t];
            }
        }
    }
    __syncthreads();
    if (warp == 0 && lane < T) {
        const int t = lane;
        float * sb = scratch + (size_t) (b * T + t) * (4 * R + 4);
        sb[sp * R + r]       = sm[0][0][t] + sm[0][1][t] + sm[0][2][t] + sm[0][3][t];
        sb[(2 + sp) * R + r] = sm[1][0][t] + sm[1][1][t] + sm[1][2][t] + sm[1][3][t];
        if (r == 0) {
            sb[4 * R + sp]     = sm[2][0][t] + sm[2][1][t] + sm[2][2][t] + sm[2][3][t];
            sb[4 * R + 2 + sp] = sm[3][0][t] + sm[3][1][t] + sm[3][2][t] + sm[3][3][t];
        }
    }
}

template <int T>
static __global__ void inkling_fused_control_up_batched_f32x4(
        const inkling_ctrl_batched_ptrs p,
        const float * __restrict__ scratch,
        const int D,
        const int R,
        const float inv_D,
        const float eps,
        const float forget_s,
        const float forget_b) {
    const int b   = blockIdx.y;
    const int tid = threadIdx.x; // 128 threads = 8 half-warps

    __shared__ __align__(16) float s_h[T][128];
    const int total_h = T * R;
    for (int idx = tid; idx < total_h; idx += 128) {
        const int t  = idx / R;
        const int ri = idx - t * R;
        const float * sb = scratch + (size_t) (b * T + t) * (4 * R + 4);
        const float ssc = sb[4 * R + 0] + sb[4 * R + 1];
        const float ssm = sb[4 * R + 2] + sb[4 * R + 3];
        const float irc = rsqrtf(ssc * inv_D + eps);
        const float irm = rsqrtf(ssm * inv_D + eps);
        const float dc  = sb[0 * R + ri] + sb[1 * R + ri];
        const float dm  = sb[2 * R + ri] + sb[3 * R + ri];
        s_h[t][ri] = dc * irc + dm * irm;
    }
    __syncthreads();

    const int lane16 = tid & 15;
    const int row    = blockIdx.x * 8 + (tid >> 4);
    if (row >= D) {
        return;
    }

    const float4 * wf = (b == 0) ? p.wf[0] : ((b == 1) ? p.wf[1] : p.wf[2]);
    const float4 * wg = (b == 0) ? p.wg[0] : ((b == 1) ? p.wg[1] : p.wg[2]);
    const float * cur = (const float *) ((b == 0) ? p.cur[0] : ((b == 1) ? p.cur[1] : p.cur[2]));
    const float * msg = (const float *) ((b == 0) ? p.msg[0] : ((b == 1) ? p.msg[1] : p.msg[2]));
    float * dst       = (b == 0) ? p.dst[0] : ((b == 1) ? p.dst[1] : p.dst[2]);

    const int R4 = R >> 2;
    const float4 * wf4 = wf + (size_t) row * R4;
    const float4 * wg4 = wg + (size_t) row * R4;

    float df[T], dg[T];
#pragma unroll
    for (int t = 0; t < T; ++t) {
        df[t] = 0.0f; dg[t] = 0.0f;
    }
    for (int k = lane16; k < R4; k += 16) {
        const float4 wfv = __ldg(&wf4[k]);
        const float4 wgv = __ldg(&wg4[k]);
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float4 hv = ((const float4 *) s_h[t])[k];
            df[t] += wfv.x * hv.x + wfv.y * hv.y + wfv.z * hv.z + wfv.w * hv.w;
            dg[t] += wgv.x * hv.x + wgv.y * hv.y + wgv.z * hv.z + wgv.w * hv.w;
        }
    }
#pragma unroll
    for (int t = 0; t < T; ++t) {
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) {
            df[t] += __shfl_xor_sync(0xffffffff, df[t], o, 16);
            dg[t] += __shfl_xor_sync(0xffffffff, dg[t], o, 16);
        }
        if (lane16 == 0) {
            const size_t idx = (size_t) t * D + row;
            dst[idx] = (tanhf(df[t]) * forget_s + forget_b) * cur[idx] + tanhf(dg[t]) * msg[idx];
        }
    }
}

static inline void inkling_launch_control_batched(
        const inkling_ctrl_batched_ptrs & p,
        float * scratch,
        const int n_ctrl,
        const int D,
        const int R,
        const int T,
        const float eps,
        const float forget_s,
        const float forget_b,
        cudaStream_t stream) {
    const int D4 = D >> 2;
    const float inv_D = 1.0f / (float) D;
    const dim3 gd(R, 2, n_ctrl);
    const dim3 gu((D + 7) / 8, n_ctrl, 1);
    switch (T) {
        case 2:
            inkling_fused_control_down_batched_f32x4<2><<<gd, 128, 0, stream>>>(p, scratch, D4, R);
            inkling_fused_control_up_batched_f32x4<2><<<gu, 128, 0, stream>>>(p, scratch, D, R, inv_D, eps, forget_s, forget_b);
            break;
        case 3:
            inkling_fused_control_down_batched_f32x4<3><<<gd, 128, 0, stream>>>(p, scratch, D4, R);
            inkling_fused_control_up_batched_f32x4<3><<<gu, 128, 0, stream>>>(p, scratch, D, R, inv_D, eps, forget_s, forget_b);
            break;
        case 4:
            inkling_fused_control_down_batched_f32x4<4><<<gd, 128, 0, stream>>>(p, scratch, D4, R);
            inkling_fused_control_up_batched_f32x4<4><<<gu, 128, 0, stream>>>(p, scratch, D, R, inv_D, eps, forget_s, forget_b);
            break;
        case 5:
            inkling_fused_control_down_batched_f32x4<5><<<gd, 128, 0, stream>>>(p, scratch, D4, R);
            inkling_fused_control_up_batched_f32x4<5><<<gu, 128, 0, stream>>>(p, scratch, D, R, inv_D, eps, forget_s, forget_b);
            break;
        case 6:
            inkling_fused_control_down_batched_f32x4<6><<<gd, 128, 0, stream>>>(p, scratch, D4, R);
            inkling_fused_control_up_batched_f32x4<6><<<gu, 128, 0, stream>>>(p, scratch, D, R, inv_D, eps, forget_s, forget_b);
            break;
        case 7:
            inkling_fused_control_down_batched_f32x4<7><<<gd, 128, 0, stream>>>(p, scratch, D4, R);
            inkling_fused_control_up_batched_f32x4<7><<<gu, 128, 0, stream>>>(p, scratch, D, R, inv_D, eps, forget_s, forget_b);
            break;
        default:
            inkling_fused_control_down_batched_f32x4<8><<<gd, 128, 0, stream>>>(p, scratch, D4, R);
            inkling_fused_control_up_batched_f32x4<8><<<gu, 128, 0, stream>>>(p, scratch, D, R, inv_D, eps, forget_s, forget_b);
            break;
    }
}

static inline void ggml_cuda_op_inkling_control(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * cur,
        const ggml_tensor * msg,
        const ggml_tensor * w_down,
        const ggml_tensor * w_forget,
        const ggml_tensor * w_gain,
        ggml_tensor * dst,
        const float eps,
        const float forget_s,
        const float forget_b) {
    const int D = (int) cur->ne[0];
    const int T = (int) cur->ne[1];
    const int R = (int) w_down->ne[1];
    const int D4 = D >> 2;
    const int R4 = R >> 2;

    if (T >= 2 && T <= 8 && R <= 128) {
        inkling_ctrl_batched_ptrs p{};
        p.cur[0] = (const float4 *) cur->data;
        p.msg[0] = (const float4 *) msg->data;
        p.wd[0]  = (const float4 *) w_down->data;
        p.wf[0]  = (const float4 *) w_forget->data;
        p.wg[0]  = (const float4 *) w_gain->data;
        p.dst[0] = (float *) dst->data;
        ggml_cuda_pool_alloc<float> scratch(ctx.pool(), (size_t) T * (4 * R + 4));
        inkling_launch_control_batched(p, scratch.get(), 1, D, R, T, eps, forget_s, forget_b, ctx.stream());
        return;
    }

    ggml_cuda_pool_alloc<float> h_buf(ctx.pool(), (size_t) R * T);

    const dim3 grid_down(R, T, 1);
    inkling_fused_control_down_f32x4<<<grid_down, 256, 0, ctx.stream()>>>(
        (const float4 *) cur->data,
        (const float4 *) msg->data,
        (const float4 *) w_down->data,
        h_buf.get(),
        D4, R, 1.0f / (float) D, eps);

    const dim3 grid_up((D + 7) / 8, T, 1);
    inkling_fused_control_up_f32x4<<<grid_up, 128, 0, ctx.stream()>>>(
        (const float4 *) h_buf.get(),
        (const float4 *) w_forget->data,
        (const float4 *) w_gain->data,
        (const float *) cur->data,
        (const float *) msg->data,
        (float *) dst->data,
        D, R4, forget_s, forget_b);
}

// 7. Fused shortconv4 (supports optional trailing residual ADD AND optional dynamic per-head taps!):
//    Replaces 8-op (CONCAT..SCALE), 9-op (CONCAT..SCALE..ADD), or 40-op dynamic (CONCAT..4xlags..SCALE)
//    with 1 single-pass kernel (zero intermediate [T+3, C] or [C, T] buffers!).
template <bool has_residual, bool has_dynamic>
static __global__ void inkling_fused_shortconv4_f32(
        const float * past,
        const int32_t * __restrict__ past_row_idx,
        const size_t past_row_stride_f,
        const float * __restrict__ cur,
        const float * __restrict__ weight,
        const float * __restrict__ residual,
        const float * __restrict__ dynamic,
        float * dst_state,
        float * __restrict__ dst_out,
        const int C,
        const int T,
        const int n_seqs,
        const int n_heads,
        const int head_dim,
        const size_t past_s1,
        const size_t past_s2,
        const size_t w_s1,
        const size_t state_s1,
        const size_t state_plane_s,
        const int kw,
        const float scale,
        const float bias) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int s = blockIdx.y;
    if (c >= C || s >= n_seqs) {
        return;
    }

    const float * past_s = past_row_idx ? (past + (size_t) past_row_idx[s] * past_row_stride_f) : (past + (size_t) s * past_s2);
    const float * p = past_s + (size_t) c * past_s1;
    float s0 = p[0];
    float s1 = p[1];
    float s2 = p[2];

    const float * wp = weight + (size_t) c * w_s1;
    const float w0 = wp[0];
    const float w1 = wp[1];
    const float w2 = wp[2];
    const float w3 = wp[3];

    const int head = has_dynamic ? (c / head_dim) : 0;

    const float * cur_s = cur     + (size_t) s * T * C + c;
    const float * res_s = has_residual ? (residual + (size_t) s * T * C + c) : nullptr;
    float *       out_s = dst_out + (size_t) s * T * C + c;

    for (int t = 0; t < T; ++t) {
        const float x = cur_s[(size_t) t * C];
        float conv = s0 * w0 + s1 * w1 + s2 * w2 + x * w3;
        if constexpr (has_dynamic) {
            const float * dp = dynamic + ((size_t) (s * T + t) * n_heads + head) * 4;
            conv += x * dp[0] + s2 * dp[1] + s1 * dp[2] + s0 * dp[3];
        }
        float val = (x + conv) * scale + bias;
        if constexpr (has_residual) {
            val += res_s[(size_t) t * C];
        }
        out_s[(size_t) t * C] = val;
        s0 = s1;
        s1 = s2;
        s2 = x;
        if (kw > 1 && t >= T - kw) {
            const int rslot = T - 1 - t;
            float * ns = dst_state + (size_t) rslot * state_plane_s + (size_t) s * state_s1 + (size_t) c * 3;
            ns[0] = s0;
            ns[1] = s1;
            ns[2] = s2;
        }
    }

    if (kw <= 1) {
        float * ns = dst_state + (size_t) s * state_s1 + (size_t) c * 3;
        ns[0] = s0;
        ns[1] = s1;
        ns[2] = s2;
    }
}

static inline void ggml_cuda_op_inkling_shortconv4(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * past,
        const ggml_tensor * cur,
        const ggml_tensor * weight,
        const ggml_tensor * residual,
        const ggml_tensor * dynamic,
        ggml_tensor * dst_state,
        ggml_tensor * dst_out,
        const int T,
        const int n_seqs,
        const int n_heads,
        const float scale,
        const float bias,
        const float * past_override = nullptr,
        const int32_t * past_row_idx = nullptr,
        const size_t past_row_stride_f = 0,
        const int kw = 1,
        const size_t state_plane_s = 0,
        const size_t state_s1_override = 0) {
    const int C        = (int) cur->ne[0];
    const int head_dim = (dynamic && n_heads > 0) ? (C / n_heads) : C;
    const size_t past_s1  = past->nb[1] / sizeof(float);
    const size_t past_s2  = past->nb[2] / sizeof(float);
    const size_t w_s1     = weight->nb[1] / sizeof(float);
    const size_t state_s1 = state_s1_override ? state_s1_override : (dst_state->nb[1] / sizeof(float));

    constexpr int threads = 256;
    const dim3 blocks((C + threads - 1) / threads, n_seqs, 1);
    const float * past_d = past_override ? past_override : (const float *) past->data;
    const float * res_d  = residual ? (const float *) residual->data : nullptr;
    const float * dyn_d  = dynamic  ? (const float *) dynamic->data  : nullptr;

    if (residual && !dynamic) {
        inkling_fused_shortconv4_f32<true, false><<<blocks, threads, 0, ctx.stream()>>>(
            past_d, past_row_idx, past_row_stride_f, (const float *) cur->data, (const float *) weight->data,
            res_d, dyn_d, (float *) dst_state->data, (float *) dst_out->data,
            C, T, n_seqs, n_heads, head_dim, past_s1, past_s2, w_s1, state_s1, state_plane_s, kw, scale, bias);
    } else if (!residual && dynamic) {
        inkling_fused_shortconv4_f32<false, true><<<blocks, threads, 0, ctx.stream()>>>(
            past_d, past_row_idx, past_row_stride_f, (const float *) cur->data, (const float *) weight->data,
            res_d, dyn_d, (float *) dst_state->data, (float *) dst_out->data,
            C, T, n_seqs, n_heads, head_dim, past_s1, past_s2, w_s1, state_s1, state_plane_s, kw, scale, bias);
    } else if (residual && dynamic) {
        inkling_fused_shortconv4_f32<true, true><<<blocks, threads, 0, ctx.stream()>>>(
            past_d, past_row_idx, past_row_stride_f, (const float *) cur->data, (const float *) weight->data,
            res_d, dyn_d, (float *) dst_state->data, (float *) dst_out->data,
            C, T, n_seqs, n_heads, head_dim, past_s1, past_s2, w_s1, state_s1, state_plane_s, kw, scale, bias);
    } else {
        inkling_fused_shortconv4_f32<false, false><<<blocks, threads, 0, ctx.stream()>>>(
            past_d, past_row_idx, past_row_stride_f, (const float *) cur->data, (const float *) weight->data,
            res_d, dyn_d, (float *) dst_state->data, (float *) dst_out->data,
            C, T, n_seqs, n_heads, head_dim, past_s1, past_s2, w_s1, state_s1, state_plane_s, kw, scale, bias);
    }
}

// 8. Fused 8-op depth-attention mix():
//    states = rms_norm(concat(blocks_cat, cur))  [D, K, T]
//    probs  = softmax(states^T * query * sm_scale) [K, Q, T]
//    dst    = states * probs                     [D, Q, T]
//    Algebraic identity: dot(rms_norm(x_k), q) = inv_rms[k] * dot(x_k, q),
//    dst = sum_k (prob[k] * inv_rms[k]) * x_k -> single DRAM read, zero transpose/intermediate buffers!
static __global__ void inkling_fused_mix_f32x4_576(
        const float4 * __restrict__ blocks_cat,
        const float4 * __restrict__ cur,
        const float4 * __restrict__ query,
        float4 * __restrict__ dst,
        const int K,
        const int Q,
        const float eps,
        const float sm_scale) {
    // 576 threads = 18 warps -> 1 float4 per thread for D = 2304, K <= 8, Q <= 4
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int t    = blockIdx.x;

    const float4 * blk_t = blocks_cat + (size_t) t * (K - 1) * 576;
    const float4 * cur_t = cur        + (size_t) t * 576;
    float4 *       dst_t = dst        + (size_t) t * Q * 576;

    float4 qv[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        if (q < Q) {
            qv[q] = query[(size_t) q * 576 + tid];
        }
    }

    __shared__ float s_ss[8][18];
    __shared__ float s_dot[4][8][18];
    __shared__ float s_eff[4][8];

    float4 xv[8];
    for (int k = 0; k < K; ++k) {
        const float4 x = (k < K - 1) ? blk_t[(size_t) k * 576 + tid] : cur_t[tid];
        xv[k] = x;

        const float ss = warp_reduce_sum<32>(x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w);
        if (lane == 0) {
            s_ss[k][warp] = ss;
        }
        for (int q = 0; q < Q; ++q) {
            const float d = warp_reduce_sum<32>(
                x.x * qv[q].x + x.y * qv[q].y + x.z * qv[q].z + x.w * qv[q].w);
            if (lane == 0) {
                s_dot[q][k][warp] = d;
            }
        }
    }
    __syncthreads();

    if (warp == 0) {
        float inv_rms[8];
        for (int k = 0; k < K; ++k) {
            const float v = (lane < 18) ? s_ss[k][lane] : 0.0f;
            const float total_ss = warp_reduce_sum<32>(v);
            inv_rms[k] = rsqrtf(total_ss * (1.0f / 2304.0f) + eps);
        }
        for (int q = 0; q < Q; ++q) {
            float logits[8];
            float max_l = -INFINITY;
            for (int k = 0; k < K; ++k) {
                const float v = (lane < 18) ? s_dot[q][k][lane] : 0.0f;
                const float total_dot = warp_reduce_sum<32>(v);
                logits[k] = total_dot * inv_rms[k] * sm_scale;
                if (logits[k] > max_l) {
                    max_l = logits[k];
                }
            }
            if (lane == 0) {
                float sum_e = 0.0f;
                float exps[8];
                for (int k = 0; k < K; ++k) {
                    exps[k] = expf(logits[k] - max_l);
                    sum_e += exps[k];
                }
                const float inv_sum = 1.0f / sum_e;
                for (int k = 0; k < K; ++k) {
                    s_eff[q][k] = exps[k] * inv_sum * inv_rms[k];
                }
            }
        }
    }
    __syncthreads();

    for (int q = 0; q < Q; ++q) {
        float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int k = 0; k < K; ++k) {
            const float w = s_eff[q][k];
            acc.x += w * xv[k].x;
            acc.y += w * xv[k].y;
            acc.z += w * xv[k].z;
            acc.w += w * xv[k].w;
        }
        dst_t[(size_t) q * 576 + tid] = acc;
    }
}

static __global__ void inkling_fused_mix_f32x4_gen(
        const float4 * __restrict__ blocks_cat,
        const float4 * __restrict__ cur,
        const float4 * __restrict__ query,
        float4 * __restrict__ dst,
        const int D4,
        const int K,
        const int Q,
        const float inv_D,
        const float eps,
        const float sm_scale) {
    // 256 threads = 8 warps, handles arbitrary D % 4 == 0, K <= 16, Q <= 4
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int t    = blockIdx.x;

    const float4 * blk_t = blocks_cat + (size_t) t * (K - 1) * D4;
    const float4 * cur_t = cur        + (size_t) t * D4;
    float4 *       dst_t = dst        + (size_t) t * Q * D4;

    __shared__ float s_ss[16][8];
    __shared__ float s_dot[4][16][8];
    __shared__ float s_eff[4][16];

    for (int k = 0; k < K; ++k) {
        const float4 * src_k = (k < K - 1) ? (blk_t + (size_t) k * D4) : cur_t;
        float ss = 0.0f;
        float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = tid; i < D4; i += 256) {
            const float4 x = src_k[i];
            ss += x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
            for (int q = 0; q < Q; ++q) {
                const float4 qv = query[(size_t) q * D4 + i];
                d[q] += x.x * qv.x + x.y * qv.y + x.z * qv.z + x.w * qv.w;
            }
        }
        ss = warp_reduce_sum<32>(ss);
        if (lane == 0) {
            s_ss[k][warp] = ss;
        }
        for (int q = 0; q < Q; ++q) {
            const float dq = warp_reduce_sum<32>(d[q]);
            if (lane == 0) {
                s_dot[q][k][warp] = dq;
            }
        }
    }
    __syncthreads();

    if (warp == 0) {
        float inv_rms[16];
        for (int k = 0; k < K; ++k) {
            const float v = (lane < 8) ? s_ss[k][lane] : 0.0f;
            const float total_ss = warp_reduce_sum<8>(v);
            inv_rms[k] = rsqrtf(total_ss * inv_D + eps);
        }
        for (int q = 0; q < Q; ++q) {
            float logits[16];
            float max_l = -INFINITY;
            for (int k = 0; k < K; ++k) {
                const float v = (lane < 8) ? s_dot[q][k][lane] : 0.0f;
                const float total_dot = warp_reduce_sum<8>(v);
                logits[k] = total_dot * inv_rms[k] * sm_scale;
                if (logits[k] > max_l) {
                    max_l = logits[k];
                }
            }
            if (lane == 0) {
                float sum_e = 0.0f;
                float exps[16];
                for (int k = 0; k < K; ++k) {
                    exps[k] = expf(logits[k] - max_l);
                    sum_e += exps[k];
                }
                const float inv_sum = 1.0f / sum_e;
                for (int k = 0; k < K; ++k) {
                    s_eff[q][k] = exps[k] * inv_sum * inv_rms[k];
                }
            }
        }
    }
    __syncthreads();

    for (int i = tid; i < D4; i += 256) {
        for (int q = 0; q < Q; ++q) {
            float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (int k = 0; k < K; ++k) {
                const float4 * src_k = (k < K - 1) ? (blk_t + (size_t) k * D4) : cur_t;
                const float4 x = src_k[i];
                const float w = s_eff[q][k];
                acc.x += w * x.x;
                acc.y += w * x.y;
                acc.z += w * x.z;
                acc.w += w * x.w;
            }
            dst_t[(size_t) q * D4 + i] = acc;
        }
    }
}

static inline void ggml_cuda_op_inkling_mix(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * blocks_cat,
        const ggml_tensor * cur,
        const ggml_tensor * query,
        ggml_tensor * dst,
        const int K,
        const int Q,
        const int T,
        const float eps,
        const float sm_scale) {
    const int D  = (int) cur->ne[0];
    const int D4 = D >> 2;
    if (D4 == 576 && K <= 8) {
        inkling_fused_mix_f32x4_576<<<T, 576, 0, ctx.stream()>>>(
            (const float4 *) blocks_cat->data,
            (const float4 *) cur->data,
            (const float4 *) query->data,
            (float4 *) dst->data,
            K, Q, eps, sm_scale);
    } else {
        inkling_fused_mix_f32x4_gen<<<T, 256, 0, ctx.stream()>>>(
            (const float4 *) blocks_cat->data,
            (const float4 *) cur->data,
            (const float4 *) query->data,
            (float4 *) dst->data,
            D4, K, Q, 1.0f / (float) D, eps, sm_scale);
    }
}

// 9. Fused 13-op relative-position attention bias + sink subtraction + gather + mask add + F16 cast:
//    Replaces PAD + MUL_MAT[n_extent+1 x n_head] + RESHAPE + SUB + PERMUTE + CONT + GET_ROWS +
//    RESHAPE + PERMUTE + CONT + CPY(mask) + ADD + CPY(F16) with 1 bank-conflict-free shared-mem kernel!
template <bool mask_is_f16, bool dst_is_f16>
static __global__ void inkling_fused_rel_pos_mask_k16(
        const float * __restrict__ rel_bias,  // [16, n_extent]
        const float * __restrict__ r_cur,     // [16, n_head, n_tokens]
        const float * __restrict__ sinks,     // [n_head, n_tokens]
        const int32_t * __restrict__ rel_pos, // [n_kv, n_tokens]
        const void * __restrict__ raw_mask,   // [n_kv, T_s, 1, S]
        void * __restrict__ dst,              // [n_kv, T_s, n_head, S]
        const int n_extent,
        const int n_head,
        const int n_kv,
        const int n_tokens_per_stream) {
    const int j0  = blockIdx.x * 32;
    const int tok = blockIdx.y;
    const int s   = tok / n_tokens_per_stream;
    const int t_s = tok - s * n_tokens_per_stream;
    const int tid = threadIdx.y * 32 + threadIdx.x; // 0..255

    __shared__ float   s_R[32 * 16];
    __shared__ float   s_sink[32];
    __shared__ int32_t s_dist[32];
    __shared__ float   s_mask[32];
    __shared__ float   s_bias[32 * 17]; // stride 17 eliminates shared-mem bank conflicts

    const float * R_tok = r_cur + (size_t) tok * n_head * 16;
    for (int idx = tid; idx < n_head * 16; idx += 256) {
        s_R[idx] = R_tok[idx];
    }
    if (tid < n_head) {
        s_sink[tid] = sinks[(size_t) tok * n_head + tid];
    }
    if (tid < 32) {
        const int j = j0 + tid;
        if (j < n_kv) {
            s_dist[tid] = rel_pos[(size_t) tok * n_kv + j];
            if constexpr (mask_is_f16) {
                s_mask[tid] = __half2float(((const half *) raw_mask)[(size_t) tok * n_kv + j]);
            } else {
                s_mask[tid] = ((const float *) raw_mask)[(size_t) tok * n_kv + j];
            }
        } else {
            s_dist[tid] = n_extent;
            s_mask[tid] = 0.0f;
        }
    }
    __syncthreads();

    // 16 groups of 16 threads cooperatively load 32 x 16 floats from rel_bias (coalesced!)
    const int grp = tid >> 4;
    const int r   = tid & 15;
#pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        const int slot = grp + pass * 16;
        const int d    = s_dist[slot];
        s_bias[slot * 17 + r] = (d >= 0 && d < n_extent) ? rel_bias[(size_t) d * 16 + r] : 0.0f;
    }
    __syncthreads();

    const int lane = threadIdx.x;
    const int j    = j0 + lane;
    if (j >= n_kv) {
        return;
    }

    const float m = s_mask[lane];
    const float * b_ptr = &s_bias[lane * 17];
    float b[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        b[k] = b_ptr[k];
    }

    for (int h = threadIdx.y; h < n_head; h += 8) {
        const float * r_ptr = &s_R[h * 16];
        float acc = 0.0f;
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            acc += b[k] * r_ptr[k];
        }
        const float val = (acc - s_sink[h]) + m;
        const size_t out_idx = (((size_t) s * n_head + h) * n_tokens_per_stream + t_s) * n_kv + j;
        if constexpr (dst_is_f16) {
            ((half *) dst)[out_idx] = __float2half(val);
        } else {
            ((float *) dst)[out_idx] = val;
        }
    }
}

static inline void ggml_cuda_op_inkling_rel_pos_mask(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * rel_bias,
        const ggml_tensor * r_cur,
        const ggml_tensor * sinks,
        const ggml_tensor * rel_pos,
        const ggml_tensor * raw_mask,
        ggml_tensor * dst) {
    const int n_extent            = (int) rel_bias->ne[1];
    const int n_kv                = (int) dst->ne[0];
    const int n_tokens_per_stream = (int) dst->ne[1];
    const int n_head              = (int) dst->ne[2];
    const int n_stream            = (int) dst->ne[3];
    const int n_tokens            = n_tokens_per_stream * n_stream;

    const dim3 block_dims(32, 8, 1);
    const dim3 grid_dims((n_kv + 31) / 32, n_tokens, 1);

    const bool mask_f16 = (raw_mask->type == GGML_TYPE_F16);
    const bool dst_f16  = (dst->type      == GGML_TYPE_F16);

    if (mask_f16 && dst_f16) {
        inkling_fused_rel_pos_mask_k16<true, true><<<grid_dims, block_dims, 0, ctx.stream()>>>(
            (const float *) rel_bias->data, (const float *) r_cur->data, (const float *) sinks->data,
            (const int32_t *) rel_pos->data, raw_mask->data, dst->data,
            n_extent, n_head, n_kv, n_tokens_per_stream);
    } else if (mask_f16 && !dst_f16) {
        inkling_fused_rel_pos_mask_k16<true, false><<<grid_dims, block_dims, 0, ctx.stream()>>>(
            (const float *) rel_bias->data, (const float *) r_cur->data, (const float *) sinks->data,
            (const int32_t *) rel_pos->data, raw_mask->data, dst->data,
            n_extent, n_head, n_kv, n_tokens_per_stream);
    } else if (!mask_f16 && dst_f16) {
        inkling_fused_rel_pos_mask_k16<false, true><<<grid_dims, block_dims, 0, ctx.stream()>>>(
            (const float *) rel_bias->data, (const float *) r_cur->data, (const float *) sinks->data,
            (const int32_t *) rel_pos->data, raw_mask->data, dst->data,
            n_extent, n_head, n_kv, n_tokens_per_stream);
    } else {
        inkling_fused_rel_pos_mask_k16<false, false><<<grid_dims, block_dims, 0, ctx.stream()>>>(
            (const float *) rel_bias->data, (const float *) r_cur->data, (const float *) sinks->data,
            (const int32_t *) rel_pos->data, raw_mask->data, dst->data,
            n_extent, n_head, n_kv, n_tokens_per_stream);
    }
}

// 10. Fused Kcur L2-clip + VIEW + SET_ROWS (10 ops -> 1 warp kernel writing directly to F16/F32/Q8_0 KV cache!)
// DST_FMT: 0 = F32, 1 = F16, 2 = Q8_0
template <typename idx_t, int DST_FMT>
static __global__ void inkling_fused_l2_clip_set_rows_warp(
        const float * __restrict__ x,
        const idx_t * __restrict__ row_idxs,
        void * __restrict__ dst_cache,
        const int head_dim,
        const int n_head_kv,
        const int n_tokens,
        const size_t dst_nb1,
        const float scale_s,
        const float scale_b,
        const float thr,
        const float cmin,
        const float cmax) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    const int total_rows = n_head_kv * n_tokens;
    if (row >= total_rows) {
        return;
    }
    const int t = row / n_head_kv;
    const int h = row - t * n_head_kv;
    const int lane = threadIdx.x;

    const float * x_row = x + (size_t) row * head_dim;
    const int64_t dst_row = (int64_t) row_idxs[t];
    char * dst_base = (char *) dst_cache + dst_row * dst_nb1;

    float v[4];
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int c = lane + k * 32;
        if (c < head_dim) {
            const float val = x_row[c];
            v[k] = val;
            ss += val * val;
        }
    }
    ss = warp_reduce_sum<32>(ss);
    const float norm = sqrtf(ss) * scale_s + scale_b;
    const float fac  = fminf(cmax, fmaxf(cmin, thr / norm));

    if constexpr (DST_FMT == 2) {
        const int blk_base = h * (head_dim >> 5);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            if (k * 32 < head_dim) {
                const float out_v = v[k] * fac;
                const float amax  = warp_reduce_max<32>(fabsf(out_v));
                const float d     = amax / 127.0f;
                const float id    = (d != 0.0f) ? (1.0f / d) : 0.0f;
                block_q8_0 * blk  = (block_q8_0 *) dst_base + blk_base + k;
                blk->qs[lane]     = (int8_t) fmaxf(-127.0f, fminf(127.0f, roundf(out_v * id)));
                if (lane == 0) {
                    blk->d = __float2half(d);
                }
            }
        }
    } else {
        const int col_base = h * head_dim;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int c = lane + k * 32;
            if (c < head_dim) {
                const float out_v = v[k] * fac;
                if constexpr (DST_FMT == 1) {
                    ((half *) dst_base)[col_base + c] = __float2half(out_v);
                } else {
                    ((float *) dst_base)[col_base + c] = out_v;
                }
            }
        }
    }
}

static inline void ggml_cuda_op_inkling_l2_clip_set_rows(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * x,
        const ggml_tensor * row_idxs,
        ggml_tensor * dst_cache,
        const float scale_s,
        const float scale_b,
        const float thr,
        const float cmin,
        const float cmax) {
    const int head_dim  = (int) x->ne[0];
    const int n_head_kv = (int) x->ne[1];
    const int n_tokens  = (int) x->ne[2];
    const int total_rows = n_head_kv * n_tokens;

    constexpr int warps_per_block = 4;
    const dim3 block_dims(32, warps_per_block, 1);
    const dim3 block_nums((total_rows + warps_per_block - 1) / warps_per_block, 1, 1);

    const bool is_i64 = (row_idxs->type == GGML_TYPE_I64);
    const int dst_fmt = (dst_cache->type == GGML_TYPE_Q8_0) ? 2 : ((dst_cache->type == GGML_TYPE_F16) ? 1 : 0);

    if (is_i64) {
        if (dst_fmt == 2) {
            inkling_fused_l2_clip_set_rows_warp<int64_t, 2><<<block_nums, block_dims, 0, ctx.stream()>>>(
                (const float *) x->data, (const int64_t *) row_idxs->data, dst_cache->data,
                head_dim, n_head_kv, n_tokens, dst_cache->nb[1], scale_s, scale_b, thr, cmin, cmax);
        } else if (dst_fmt == 1) {
            inkling_fused_l2_clip_set_rows_warp<int64_t, 1><<<block_nums, block_dims, 0, ctx.stream()>>>(
                (const float *) x->data, (const int64_t *) row_idxs->data, dst_cache->data,
                head_dim, n_head_kv, n_tokens, dst_cache->nb[1], scale_s, scale_b, thr, cmin, cmax);
        } else {
            inkling_fused_l2_clip_set_rows_warp<int64_t, 0><<<block_nums, block_dims, 0, ctx.stream()>>>(
                (const float *) x->data, (const int64_t *) row_idxs->data, dst_cache->data,
                head_dim, n_head_kv, n_tokens, dst_cache->nb[1], scale_s, scale_b, thr, cmin, cmax);
        }
    } else {
        if (dst_fmt == 2) {
            inkling_fused_l2_clip_set_rows_warp<int32_t, 2><<<block_nums, block_dims, 0, ctx.stream()>>>(
                (const float *) x->data, (const int32_t *) row_idxs->data, dst_cache->data,
                head_dim, n_head_kv, n_tokens, dst_cache->nb[1], scale_s, scale_b, thr, cmin, cmax);
        } else if (dst_fmt == 1) {
            inkling_fused_l2_clip_set_rows_warp<int32_t, 1><<<block_nums, block_dims, 0, ctx.stream()>>>(
                (const float *) x->data, (const int32_t *) row_idxs->data, dst_cache->data,
                head_dim, n_head_kv, n_tokens, dst_cache->nb[1], scale_s, scale_b, thr, cmin, cmax);
        } else {
            inkling_fused_l2_clip_set_rows_warp<int32_t, 0><<<block_nums, block_dims, 0, ctx.stream()>>>(
                (const float *) x->data, (const int32_t *) row_idxs->data, dst_cache->data,
                head_dim, n_head_kv, n_tokens, dst_cache->nb[1], scale_s, scale_b, thr, cmin, cmax);
        }
    }
}

// 11. Fused Vcur RMS_NORM + VIEW + SET_ROWS (3 ops -> 1 warp kernel writing directly to F16/F32/Q8_0 KV cache!)
template <typename idx_t, int DST_FMT>
static __global__ void inkling_fused_rms_norm_set_rows_warp(
        const float * __restrict__ x,
        const idx_t * __restrict__ row_idxs,
        void * __restrict__ dst_cache,
        const int head_dim,
        const int n_head_kv,
        const int n_tokens,
        const size_t dst_nb1,
        const float eps) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    const int total_rows = n_head_kv * n_tokens;
    if (row >= total_rows) {
        return;
    }
    const int t = row / n_head_kv;
    const int h = row - t * n_head_kv;
    const int lane = threadIdx.x;

    const float * x_row = x + (size_t) row * head_dim;
    const int64_t dst_row = (int64_t) row_idxs[t];
    char * dst_base = (char *) dst_cache + dst_row * dst_nb1;

    float v[4];
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int c = lane + k * 32;
        if (c < head_dim) {
            const float val = x_row[c];
            v[k] = val;
            ss += val * val;
        }
    }
    ss = warp_reduce_sum<32>(ss);
    const float inv_rms = rsqrtf(ss / (float) head_dim + eps);

    if constexpr (DST_FMT == 2) {
        const int blk_base = h * (head_dim >> 5);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            if (k * 32 < head_dim) {
                const float out_v = v[k] * inv_rms;
                const float amax  = warp_reduce_max<32>(fabsf(out_v));
                const float d     = amax / 127.0f;
                const float id    = (d != 0.0f) ? (1.0f / d) : 0.0f;
                block_q8_0 * blk  = (block_q8_0 *) dst_base + blk_base + k;
                blk->qs[lane]     = (int8_t) fmaxf(-127.0f, fminf(127.0f, roundf(out_v * id)));
                if (lane == 0) {
                    blk->d = __float2half(d);
                }
            }
        }
    } else {
        const int col_base = h * head_dim;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int c = lane + k * 32;
            if (c < head_dim) {
                const float out_v = v[k] * inv_rms;
                if constexpr (DST_FMT == 1) {
                    ((half *) dst_base)[col_base + c] = __float2half(out_v);
                } else {
                    ((float *) dst_base)[col_base + c] = out_v;
                }
            }
        }
    }
}

static inline void ggml_cuda_op_inkling_rms_norm_set_rows(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * x,
        const ggml_tensor * row_idxs,
        ggml_tensor * dst_cache,
        const float eps) {
    const int head_dim   = (int) x->ne[0];
    const int n_head_kv  = (int) x->ne[1];
    const int n_tokens   = (int) x->ne[2];
    const int total_rows = n_head_kv * n_tokens;

    constexpr int warps_per_block = 4;
    const dim3 block_dims(32, warps_per_block, 1);
    const dim3 block_nums((total_rows + warps_per_block - 1) / warps_per_block, 1, 1);

    const bool is_i64 = (row_idxs->type == GGML_TYPE_I64);
    const int dst_fmt = (dst_cache->type == GGML_TYPE_Q8_0) ? 2 : ((dst_cache->type == GGML_TYPE_F16) ? 1 : 0);

    if (is_i64) {
        if (dst_fmt == 2) {
            inkling_fused_rms_norm_set_rows_warp<int64_t, 2><<<block_nums, block_dims, 0, ctx.stream()>>>(
                (const float *) x->data, (const int64_t *) row_idxs->data, dst_cache->data,
                head_dim, n_head_kv, n_tokens, dst_cache->nb[1], eps);
        } else if (dst_fmt == 1) {
            inkling_fused_rms_norm_set_rows_warp<int64_t, 1><<<block_nums, block_dims, 0, ctx.stream()>>>(
                (const float *) x->data, (const int64_t *) row_idxs->data, dst_cache->data,
                head_dim, n_head_kv, n_tokens, dst_cache->nb[1], eps);
        } else {
            inkling_fused_rms_norm_set_rows_warp<int64_t, 0><<<block_nums, block_dims, 0, ctx.stream()>>>(
                (const float *) x->data, (const int64_t *) row_idxs->data, dst_cache->data,
                head_dim, n_head_kv, n_tokens, dst_cache->nb[1], eps);
        }
    } else {
        if (dst_fmt == 2) {
            inkling_fused_rms_norm_set_rows_warp<int32_t, 2><<<block_nums, block_dims, 0, ctx.stream()>>>(
                (const float *) x->data, (const int32_t *) row_idxs->data, dst_cache->data,
                head_dim, n_head_kv, n_tokens, dst_cache->nb[1], eps);
        } else if (dst_fmt == 1) {
            inkling_fused_rms_norm_set_rows_warp<int32_t, 1><<<block_nums, block_dims, 0, ctx.stream()>>>(
                (const float *) x->data, (const int32_t *) row_idxs->data, dst_cache->data,
                head_dim, n_head_kv, n_tokens, dst_cache->nb[1], eps);
        } else {
            inkling_fused_rms_norm_set_rows_warp<int32_t, 0><<<block_nums, block_dims, 0, ctx.stream()>>>(
                (const float *) x->data, (const int32_t *) row_idxs->data, dst_cache->data,
                head_dim, n_head_kv, n_tokens, dst_cache->nb[1], eps);
        }
    }
}

// ============================================================================
// Wave 6 Mega-Fusions (SM 8.6 / Ampere Breakthrough: Horizontal Attention GEMV,
// Ampere cp.async.cg NVFP4 gate_up+SiTU, & Pre-interleaved Flattened down+reduce)
// ============================================================================

// 12. All-3-Input Mega-Horizontal Multi-GEMV (2..8 projections across 1..3 F32 input vectors, T == 1):
//     Fuses Q+Gate+R+K+V+Sink (6->1 on all 40 SWA layers) & Q+Gate+R+K+V+Sink+Kdyn+Vdyn (8->1 on all 8 full-attn layers)
//     with a SINGLE multi-input Q8_1 quantization and 1 warp/row x 8 warps/block (zero smem, zero barriers, zero spills!).
struct inkling_horiz_mat {
    const void * w;
    float *      dst;
    int          row_off;
    int          row_end;
    int          nrows;
    int          type;   // 0 = Q8_0, 1 = BF16, 2 = F16, 3 = F32
    int          in_idx; // 0, 1, or 2
};

struct inkling_horiz_args {
    inkling_horiz_mat m[8];
    const float *     yf[3];
    int               n_mats;
    int               total_rows;
};

struct inkling_ptr3 {
    const float * p[3];
};

static __global__ void inkling_quantize_q8_1_multi_f32(
        const inkling_ptr3 src,
        block_q8_1 * __restrict__ y,
        const int K,
        const int nb_q8) {
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    if (i0 >= K) {
        return;
    }
    const int u = blockIdx.y;
    const float * x = (u == 0) ? src.p[0] : ((u == 1) ? src.p[1] : src.p[2]);
    const float xi = x[i0];
    const float amax = warp_reduce_max<32>(fabsf(xi));
    const float d = amax / 127.0f;
    const int8_t q = (amax == 0.0f) ? 0 : (int8_t) roundf(xi / d);
    const int ib  = u * nb_q8 + (i0 >> 5);
    const int iqs = i0 & 31;
    y[ib].qs[iqs] = q;
    if (iqs == 0) {
        y[ib].ds = make_half2(__float2half(d), __float2half(0.0f));
    }
}

static __device__ __forceinline__ int inkling_get_int_b2(const void * x, const int i32) {
    const uint16_t * x16 = (const uint16_t *) x;
    return (int) __ldg(&x16[2 * i32 + 0]) | ((int) __ldg(&x16[2 * i32 + 1]) << 16);
}

static __device__ __forceinline__ void inkling_store_q8_1_from_f32x4(
        const float4 v,
        block_q8_1 * __restrict__ yq,
        const int ib,
        const int lane8) {
    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 4, 8));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2, 8));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1, 8));
    const float d = amax / 127.0f;
    const int8_t q0 = (amax == 0.0f) ? 0 : (int8_t) roundf(v.x / d);
    const int8_t q1 = (amax == 0.0f) ? 0 : (int8_t) roundf(v.y / d);
    const int8_t q2 = (amax == 0.0f) ? 0 : (int8_t) roundf(v.z / d);
    const int8_t q3 = (amax == 0.0f) ? 0 : (int8_t) roundf(v.w / d);
    const uint32_t packed = ((uint32_t) (uint8_t) q0)
                          | ((uint32_t) (uint8_t) q1 << 8)
                          | ((uint32_t) (uint8_t) q2 << 16)
                          | ((uint32_t) (uint8_t) q3 << 24);
    *((uint32_t *) (yq[ib].qs + lane8 * 4)) = packed;
    if (lane8 == 0) {
        yq[ib].ds = make_half2(__float2half(d), __float2half(0.0f));
    }
}

template <int NWARPS, int NB_FIXED>
static __global__ void inkling_fused_horiz_mmv_f32(
        const inkling_horiz_args args,
        const block_q8_1 * __restrict__ yq_base,
        const int K,
        const int nb_arg) {
    const int g = blockIdx.x * NWARPS + threadIdx.y;
    if (g >= args.total_rows) {
        return;
    }
    const int tid = threadIdx.x;

    const void * w_ptr   = nullptr;
    float *      dst_ptr = nullptr;
    int          row_off = 0;
    int          mtype   = 0;
    int          in_idx  = 0;
    if (g < args.m[0].row_end) {
        w_ptr = args.m[0].w; dst_ptr = args.m[0].dst; row_off = args.m[0].row_off; mtype = args.m[0].type; in_idx = args.m[0].in_idx;
    } else if (g < args.m[1].row_end) {
        w_ptr = args.m[1].w; dst_ptr = args.m[1].dst; row_off = args.m[1].row_off; mtype = args.m[1].type; in_idx = args.m[1].in_idx;
    } else if (args.n_mats > 2 && g < args.m[2].row_end) {
        w_ptr = args.m[2].w; dst_ptr = args.m[2].dst; row_off = args.m[2].row_off; mtype = args.m[2].type; in_idx = args.m[2].in_idx;
    } else if (args.n_mats > 3 && g < args.m[3].row_end) {
        w_ptr = args.m[3].w; dst_ptr = args.m[3].dst; row_off = args.m[3].row_off; mtype = args.m[3].type; in_idx = args.m[3].in_idx;
    } else if (args.n_mats > 4 && g < args.m[4].row_end) {
        w_ptr = args.m[4].w; dst_ptr = args.m[4].dst; row_off = args.m[4].row_off; mtype = args.m[4].type; in_idx = args.m[4].in_idx;
    } else if (args.n_mats > 5 && g < args.m[5].row_end) {
        w_ptr = args.m[5].w; dst_ptr = args.m[5].dst; row_off = args.m[5].row_off; mtype = args.m[5].type; in_idx = args.m[5].in_idx;
    } else if (args.n_mats > 6 && g < args.m[6].row_end) {
        w_ptr = args.m[6].w; dst_ptr = args.m[6].dst; row_off = args.m[6].row_off; mtype = args.m[6].type; in_idx = args.m[6].in_idx;
    } else {
        w_ptr = args.m[7].w; dst_ptr = args.m[7].dst; row_off = args.m[7].row_off; mtype = args.m[7].type; in_idx = args.m[7].in_idx;
    }
    const int r = g - row_off;

    if (mtype == 0) {
        const int nb = (NB_FIXED > 0) ? NB_FIXED : nb_arg;
        const block_q8_0 * xrow = (const block_q8_0 *) w_ptr + (size_t) r * nb;
        const block_q8_1 * yq   = yq_base + (size_t) in_idx * nb;
        const int iqs = (tid & 3) * 2;
        float sum = 0.0f;
#pragma unroll
        for (int k = tid >> 2; k < nb; k += 8) {
            const block_q8_0 * bx = &xrow[k];
            const block_q8_1 * by = &yq[k];
            const int * uy = (const int *) by->qs;
            int s = ggml_cuda_dp4a(inkling_get_int_b2(bx->qs, iqs + 0), __ldg(uy + iqs + 0), 0);
            s     = ggml_cuda_dp4a(inkling_get_int_b2(bx->qs, iqs + 1), __ldg(uy + iqs + 1), s);
            sum += (float) s * __half2float(__ldg(&bx->d)) * __low2float(by->ds);
        }
        sum = warp_reduce_sum<32>(sum);
        if (tid == 0) {
            dst_ptr[r] = sum;
        }
    } else {
        const float * yf = (in_idx == 0) ? args.yf[0] : ((in_idx == 1) ? args.yf[1] : args.yf[2]);
        float acc = 0.0f;
        if (mtype == 1) {
            const nv_bfloat16 * wr = (const nv_bfloat16 *) w_ptr + (size_t) r * K;
            for (int k = tid; k < K; k += 32) {
                acc += ggml_cuda_cast<float>(__ldg(&wr[k])) * __ldg(&yf[k]);
            }
        } else if (mtype == 2) {
            const half * wr = (const half *) w_ptr + (size_t) r * K;
            for (int k = tid; k < K; k += 32) {
                acc += __half2float(__ldg(&wr[k])) * __ldg(&yf[k]);
            }
        } else {
            const float * wr = (const float *) w_ptr + (size_t) r * K;
            for (int k = tid; k < K; k += 32) {
                acc += __ldg(&wr[k]) * __ldg(&yf[k]);
            }
        }
        acc = warp_reduce_sum<32>(acc);
        if (tid == 0) {
            dst_ptr[r] = acc;
        }
    }
}

static __global__ void inkling_quantize_q8_1_multi_batched_f32(
        const inkling_ptr3 src,
        block_q8_1 * __restrict__ y,
        const int K,
        const int nb_q8,
        const int T) {
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    if (i0 >= K) {
        return;
    }
    const int t = blockIdx.y;
    const int u = blockIdx.z;
    const float * x = (u == 0) ? src.p[0] : ((u == 1) ? src.p[1] : src.p[2]);
    const float xi = x[(size_t) t * K + i0];
    const float amax = warp_reduce_max<32>(fabsf(xi));
    const float d = amax / 127.0f;
    const int8_t q = (amax == 0.0f) ? 0 : (int8_t) roundf(xi / d);
    const int ib  = (u * T + t) * nb_q8 + (i0 >> 5);
    const int iqs = i0 & 31;
    y[ib].qs[iqs] = q;
    if (iqs == 0) {
        y[ib].ds = make_half2(__float2half(d), __float2half(0.0f));
    }
}

template <int NWARPS, int NB_FIXED, int T>
static __global__ void inkling_fused_horiz_mmv_batched_f32(
        const inkling_horiz_args args,
        const block_q8_1 * __restrict__ yq_base,
        const int K,
        const int nb_arg) {
    const int g = blockIdx.x * NWARPS + threadIdx.y;
    if (g >= args.total_rows) {
        return;
    }
    const int tid = threadIdx.x;

    const void * w_ptr   = nullptr;
    float *      dst_ptr = nullptr;
    int          row_off = 0;
    int          nrows   = 0;
    int          mtype   = 0;
    int          in_idx  = 0;
    if (g < args.m[0].row_end) {
        w_ptr = args.m[0].w; dst_ptr = args.m[0].dst; row_off = args.m[0].row_off; nrows = args.m[0].nrows; mtype = args.m[0].type; in_idx = args.m[0].in_idx;
    } else if (g < args.m[1].row_end) {
        w_ptr = args.m[1].w; dst_ptr = args.m[1].dst; row_off = args.m[1].row_off; nrows = args.m[1].nrows; mtype = args.m[1].type; in_idx = args.m[1].in_idx;
    } else if (args.n_mats > 2 && g < args.m[2].row_end) {
        w_ptr = args.m[2].w; dst_ptr = args.m[2].dst; row_off = args.m[2].row_off; nrows = args.m[2].nrows; mtype = args.m[2].type; in_idx = args.m[2].in_idx;
    } else if (args.n_mats > 3 && g < args.m[3].row_end) {
        w_ptr = args.m[3].w; dst_ptr = args.m[3].dst; row_off = args.m[3].row_off; nrows = args.m[3].nrows; mtype = args.m[3].type; in_idx = args.m[3].in_idx;
    } else if (args.n_mats > 4 && g < args.m[4].row_end) {
        w_ptr = args.m[4].w; dst_ptr = args.m[4].dst; row_off = args.m[4].row_off; nrows = args.m[4].nrows; mtype = args.m[4].type; in_idx = args.m[4].in_idx;
    } else if (args.n_mats > 5 && g < args.m[5].row_end) {
        w_ptr = args.m[5].w; dst_ptr = args.m[5].dst; row_off = args.m[5].row_off; nrows = args.m[5].nrows; mtype = args.m[5].type; in_idx = args.m[5].in_idx;
    } else if (args.n_mats > 6 && g < args.m[6].row_end) {
        w_ptr = args.m[6].w; dst_ptr = args.m[6].dst; row_off = args.m[6].row_off; nrows = args.m[6].nrows; mtype = args.m[6].type; in_idx = args.m[6].in_idx;
    } else {
        w_ptr = args.m[7].w; dst_ptr = args.m[7].dst; row_off = args.m[7].row_off; nrows = args.m[7].nrows; mtype = args.m[7].type; in_idx = args.m[7].in_idx;
    }
    const int r = g - row_off;

    float sum[T];
#pragma unroll
    for (int t = 0; t < T; ++t) {
        sum[t] = 0.0f;
    }

    if (mtype == 0) {
        const int nb = (NB_FIXED > 0) ? NB_FIXED : nb_arg;
        const block_q8_0 * xrow = (const block_q8_0 *) w_ptr + (size_t) r * nb;
        const block_q8_1 * yq   = yq_base + (size_t) in_idx * T * nb;
        const int iqs = (tid & 3) * 2;
        for (int k = tid >> 2; k < nb; k += 8) {
            const block_q8_0 * bx = &xrow[k];
            const int x0 = inkling_get_int_b2(bx->qs, iqs + 0);
            const int x1 = inkling_get_int_b2(bx->qs, iqs + 1);
            const float dx = __half2float(__ldg(&bx->d));
#pragma unroll
            for (int t = 0; t < T; ++t) {
                const block_q8_1 * by = &yq[(size_t) t * nb + k];
                const int * uy = (const int *) by->qs;
                int s = ggml_cuda_dp4a(x0, __ldg(uy + iqs + 0), 0);
                s     = ggml_cuda_dp4a(x1, __ldg(uy + iqs + 1), s);
                sum[t] += (float) s * dx * __low2float(by->ds);
            }
        }
    } else {
        const float * yf = (in_idx == 0) ? args.yf[0] : ((in_idx == 1) ? args.yf[1] : args.yf[2]);
        if (mtype == 1) {
            const nv_bfloat16 * wr = (const nv_bfloat16 *) w_ptr + (size_t) r * K;
            for (int k = tid; k < K; k += 32) {
                const float wv = ggml_cuda_cast<float>(__ldg(&wr[k]));
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    sum[t] += wv * __ldg(&yf[(size_t) t * K + k]);
                }
            }
        } else if (mtype == 2) {
            const half * wr = (const half *) w_ptr + (size_t) r * K;
            for (int k = tid; k < K; k += 32) {
                const float wv = __half2float(__ldg(&wr[k]));
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    sum[t] += wv * __ldg(&yf[(size_t) t * K + k]);
                }
            }
        } else {
            const float * wr = (const float *) w_ptr + (size_t) r * K;
            for (int k = tid; k < K; k += 32) {
                const float wv = __ldg(&wr[k]);
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    sum[t] += wv * __ldg(&yf[(size_t) t * K + k]);
                }
            }
        }
    }
#pragma unroll
    for (int t = 0; t < T; ++t) {
        const float v = warp_reduce_sum<32>(sum[t]);
        if (tid == 0) {
            dst_ptr[(size_t) t * nrows + r] = v;
        }
    }
}

template <int NWARPS, int NB_FIXED>
static inline void inkling_launch_horiz_mmv_batched(
        const inkling_horiz_args & args,
        const block_q8_1 * yq,
        const int K,
        const int nb,
        const int T,
        cudaStream_t stream) {
    const dim3 block_dims(32, NWARPS, 1);
    const dim3 grid_dims((args.total_rows + NWARPS - 1) / NWARPS, 1, 1);
    switch (T) {
        case 2: inkling_fused_horiz_mmv_batched_f32<NWARPS, NB_FIXED, 2><<<grid_dims, block_dims, 0, stream>>>(args, yq, K, nb); break;
        case 3: inkling_fused_horiz_mmv_batched_f32<NWARPS, NB_FIXED, 3><<<grid_dims, block_dims, 0, stream>>>(args, yq, K, nb); break;
        case 4: inkling_fused_horiz_mmv_batched_f32<NWARPS, NB_FIXED, 4><<<grid_dims, block_dims, 0, stream>>>(args, yq, K, nb); break;
        case 5: inkling_fused_horiz_mmv_batched_f32<NWARPS, NB_FIXED, 5><<<grid_dims, block_dims, 0, stream>>>(args, yq, K, nb); break;
        case 6: inkling_fused_horiz_mmv_batched_f32<NWARPS, NB_FIXED, 6><<<grid_dims, block_dims, 0, stream>>>(args, yq, K, nb); break;
        case 7: inkling_fused_horiz_mmv_batched_f32<NWARPS, NB_FIXED, 7><<<grid_dims, block_dims, 0, stream>>>(args, yq, K, nb); break;
        default: inkling_fused_horiz_mmv_batched_f32<NWARPS, NB_FIXED, 8><<<grid_dims, block_dims, 0, stream>>>(args, yq, K, nb); break;
    }
}

static inline void ggml_cuda_op_inkling_horiz_mmv(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * const * inputs,
        const int n_inputs,
        ggml_tensor * const * mm_nodes,
        const int * in_idxs,
        const int n_mats,
        const block_q8_1 * pre_yq = nullptr) {
    const int K  = (int) inputs[0]->ne[0];
    const int T  = (int) inputs[0]->ne[1];
    const int nb = K / QK8_0;
    inkling_horiz_args args{};
    args.n_mats = n_mats;
    inkling_ptr3 qptrs{};
    for (int u = 0; u < n_inputs; ++u) {
        args.yf[u] = (const float *) inputs[u]->data;
        qptrs.p[u] = (const float *) inputs[u]->data;
    }

    bool need_q8 = false;
    bool need_yf_in[3] = { false, false, false };
    int off = 0;
    for (int j = 0; j < n_mats; ++j) {
        const ggml_tensor * w = mm_nodes[j]->src[0];
        const int nrows = (int) w->ne[1];
        args.m[j].w       = w->data;
        args.m[j].dst     = (float *) mm_nodes[j]->data;
        args.m[j].row_off = off;
        off += nrows;
        args.m[j].row_end = off;
        args.m[j].nrows   = nrows;
        args.m[j].in_idx  = in_idxs[j];
        if (w->type == GGML_TYPE_Q8_0) {
            args.m[j].type = 0;
            need_q8 = true;
        } else {
            need_yf_in[in_idxs[j]] = true;
            if (w->type == GGML_TYPE_BF16) {
                args.m[j].type = 1;
            } else if (w->type == GGML_TYPE_F16) {
                args.m[j].type = 2;
            } else {
                args.m[j].type = 3;
            }
        }
    }
    args.total_rows = off;

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<float>      yf_safe(ctx.pool());
    ggml_cuda_pool_alloc<block_q8_1> src1_q8_1(ctx.pool());
    const block_q8_1 * yq = pre_yq;

    // Copy any input vector needed by non-Q8_0 matrices if an output overlaps it (enables 48/48 fusion!)
    auto overlaps_range = [](const ggml_tensor * a, const ggml_tensor * b) {
        const int64_t a0 = (int64_t) a->data;
        const int64_t a1 = a0 + (int64_t) ggml_backend_buft_get_alloc_size(a->buffer->buft, a);
        const int64_t b0 = (int64_t) b->data;
        const int64_t b1 = b0 + (int64_t) ggml_backend_buft_get_alloc_size(b->buffer->buft, b);
        return (b0 <= a0 && a0 < b1) || (a0 <= b0 && b0 < a1);
    };
    bool copy_yf[3] = { false, false, false };
    int n_copy_yf = 0;
    for (int u = 0; u < n_inputs; ++u) {
        if (!need_yf_in[u]) continue;
        for (int j = 0; j < n_mats; ++j) {
            if (overlaps_range(mm_nodes[j], inputs[u])) {
                copy_yf[u] = true;
                n_copy_yf++;
                break;
            }
        }
    }
    if (n_copy_yf > 0) {
        const size_t elems = (size_t) K * T;
        yf_safe.alloc((size_t) n_copy_yf * elems);
        int slot = 0;
        for (int u = 0; u < n_inputs; ++u) {
            if (copy_yf[u]) {
                float * dst_yf = yf_safe.get() + (size_t) slot * elems;
                CUDA_CHECK(cudaMemcpyAsync(dst_yf, inputs[u]->data, elems * sizeof(float), cudaMemcpyDeviceToDevice, stream));
                args.yf[u] = dst_yf;
                slot++;
            }
        }
    }

    if (T == 1) {
        if (need_q8 && yq == nullptr) {
            src1_q8_1.alloc((size_t) n_inputs * nb);
            inkling_quantize_q8_1_multi_f32<<<dim3((K + 255) / 256, n_inputs, 1), 256, 0, stream>>>(
                qptrs, src1_q8_1.get(), K, nb);
            yq = src1_q8_1.get();
        }
        constexpr int NWARPS = 8;
        const dim3 block_dims(32, NWARPS, 1);
        const dim3 grid_dims((args.total_rows + NWARPS - 1) / NWARPS, 1, 1);
        if (nb == 72) {
            inkling_fused_horiz_mmv_f32<NWARPS, 72><<<grid_dims, block_dims, 0, stream>>>(args, yq, K, 72);
        } else {
            inkling_fused_horiz_mmv_f32<NWARPS, 0><<<grid_dims, block_dims, 0, stream>>>(args, yq, K, nb);
        }
        return;
    }

    if (need_q8) {
        src1_q8_1.alloc((size_t) n_inputs * T * nb);
        inkling_quantize_q8_1_multi_batched_f32<<<dim3((K + 255) / 256, T, n_inputs), 256, 0, stream>>>(
            qptrs, src1_q8_1.get(), K, nb, T);
        yq = src1_q8_1.get();
    }
    constexpr int NWARPS = 8;
    if (nb == 72) {
        inkling_launch_horiz_mmv_batched<NWARPS, 72>(args, yq, K, 72, T, stream);
    } else {
        inkling_launch_horiz_mmv_batched<NWARPS, 0>(args, yq, K, nb, T, stream);
    }
}

// 12b. Fused sigmoid_mul + wo Q8_0 GEMV (3 ops -> fused quantize + zero-barrier Q8_0 GEMV, T = 1..16):
static __global__ void inkling_quantize_sigmoid_mul_q8_1_f32(
        const float * __restrict__ x,
        const float * __restrict__ gate,
        block_q8_1 * __restrict__ y,
        const int K) {
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    if (i0 >= K) {
        return;
    }
    const float g  = gate[i0];
    const float xi = x[i0] * (1.0f / (1.0f + expf(-g)));
    const float amax = warp_reduce_max<32>(fabsf(xi));
    const float d = amax / 127.0f;
    const int8_t q = (amax == 0.0f) ? 0 : (int8_t) roundf(xi / d);
    const int ib  = i0 >> 5;
    const int iqs = i0 & 31;
    y[ib].qs[iqs] = q;
    if (iqs == 0) {
        y[ib].ds = make_half2(__float2half(d), __float2half(0.0f));
    }
}

static __global__ void inkling_quantize_sigmoid_mul_q8_1_batched_f32(
        const float * __restrict__ x,
        const float * __restrict__ gate,
        block_q8_1 * __restrict__ y,
        const int K,
        const int nb) {
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    if (i0 >= K) {
        return;
    }
    const int t = blockIdx.y;
    const size_t off = (size_t) t * K + i0;
    const float g  = gate[off];
    const float xi = x[off] * (1.0f / (1.0f + expf(-g)));
    const float amax = warp_reduce_max<32>(fabsf(xi));
    const float d = amax / 127.0f;
    const int8_t q = (amax == 0.0f) ? 0 : (int8_t) roundf(xi / d);
    const int ib  = t * nb + (i0 >> 5);
    const int iqs = i0 & 31;
    y[ib].qs[iqs] = q;
    if (iqs == 0) {
        y[ib].ds = make_half2(__float2half(d), __float2half(0.0f));
    }
}

template <int NWARPS, int NB_FIXED>
static __global__ void inkling_q8_0_mmv_f32(
        const block_q8_0 * __restrict__ w,
        const block_q8_1 * __restrict__ yq,
        float * __restrict__ dst,
        const int nrows,
        const int nb_arg) {
    const int r = blockIdx.x * NWARPS + threadIdx.y;
    if (r >= nrows) {
        return;
    }
    const int tid = threadIdx.x;
    const int nb  = (NB_FIXED > 0) ? NB_FIXED : nb_arg;
    const block_q8_0 * xrow = w + (size_t) r * nb;
    const int iqs = (tid & 3) * 2;
    float sum = 0.0f;
#pragma unroll
    for (int k = tid >> 2; k < nb; k += 8) {
        const block_q8_0 * bx = &xrow[k];
        const block_q8_1 * by = &yq[k];
        const int * uy = (const int *) by->qs;
        int s = ggml_cuda_dp4a(inkling_get_int_b2(bx->qs, iqs + 0), __ldg(uy + iqs + 0), 0);
        s     = ggml_cuda_dp4a(inkling_get_int_b2(bx->qs, iqs + 1), __ldg(uy + iqs + 1), s);
        sum += (float) s * __half2float(__ldg(&bx->d)) * __low2float(by->ds);
    }
    sum = warp_reduce_sum<32>(sum);
    if (tid == 0) {
        dst[r] = sum;
    }
}

template <int NWARPS, int NB_FIXED, int T>
static __global__ void inkling_q8_0_mmv_batched_f32(
        const block_q8_0 * __restrict__ w,
        const block_q8_1 * __restrict__ yq,
        float * __restrict__ dst,
        const int nrows,
        const int nb_arg) {
    const int r = blockIdx.x * NWARPS + threadIdx.y;
    if (r >= nrows) {
        return;
    }
    const int tid = threadIdx.x;
    const int nb  = (NB_FIXED > 0) ? NB_FIXED : nb_arg;
    const block_q8_0 * xrow = w + (size_t) r * nb;
    const int iqs = (tid & 3) * 2;
    float sum[T];
#pragma unroll
    for (int t = 0; t < T; ++t) {
        sum[t] = 0.0f;
    }
    for (int k = tid >> 2; k < nb; k += 8) {
        const block_q8_0 * bx = &xrow[k];
        const int x0 = inkling_get_int_b2(bx->qs, iqs + 0);
        const int x1 = inkling_get_int_b2(bx->qs, iqs + 1);
        const float dx = __half2float(__ldg(&bx->d));
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const block_q8_1 * by = &yq[(size_t) t * nb + k];
            const int * uy = (const int *) by->qs;
            int s = ggml_cuda_dp4a(x0, __ldg(uy + iqs + 0), 0);
            s     = ggml_cuda_dp4a(x1, __ldg(uy + iqs + 1), s);
            sum[t] += (float) s * dx * __low2float(by->ds);
        }
    }
#pragma unroll
    for (int t = 0; t < T; ++t) {
        const float v = warp_reduce_sum<32>(sum[t]);
        if (tid == 0) {
            dst[(size_t) t * nrows + r] = v;
        }
    }
}

template <int NWARPS, int NB_FIXED>
static inline void inkling_launch_q8_0_mmv_batched(
        const block_q8_0 * w,
        const block_q8_1 * yq,
        float * dst,
        const int nrows,
        const int nb,
        const int T,
        cudaStream_t stream) {
    const dim3 block_dims(32, NWARPS, 1);
    const dim3 grid_dims((nrows + NWARPS - 1) / NWARPS, 1, 1);
    switch (T) {
        case 2: inkling_q8_0_mmv_batched_f32<NWARPS, NB_FIXED, 2><<<grid_dims, block_dims, 0, stream>>>(w, yq, dst, nrows, nb); break;
        case 3: inkling_q8_0_mmv_batched_f32<NWARPS, NB_FIXED, 3><<<grid_dims, block_dims, 0, stream>>>(w, yq, dst, nrows, nb); break;
        case 4: inkling_q8_0_mmv_batched_f32<NWARPS, NB_FIXED, 4><<<grid_dims, block_dims, 0, stream>>>(w, yq, dst, nrows, nb); break;
        case 5: inkling_q8_0_mmv_batched_f32<NWARPS, NB_FIXED, 5><<<grid_dims, block_dims, 0, stream>>>(w, yq, dst, nrows, nb); break;
        case 6: inkling_q8_0_mmv_batched_f32<NWARPS, NB_FIXED, 6><<<grid_dims, block_dims, 0, stream>>>(w, yq, dst, nrows, nb); break;
        case 7: inkling_q8_0_mmv_batched_f32<NWARPS, NB_FIXED, 7><<<grid_dims, block_dims, 0, stream>>>(w, yq, dst, nrows, nb); break;
        default: inkling_q8_0_mmv_batched_f32<NWARPS, NB_FIXED, 8><<<grid_dims, block_dims, 0, stream>>>(w, yq, dst, nrows, nb); break;
    }
}

static inline void ggml_cuda_op_inkling_sigmoid_mul_q8_0_mmv(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * w,
        const ggml_tensor * x,
        const ggml_tensor * gate,
        ggml_tensor * dst) {
    const int K     = (int) w->ne[0];
    const int nrows = (int) w->ne[1];
    const int T     = (int) x->ne[1];
    const int nb    = K / QK8_0;
    cudaStream_t stream = ctx.stream();

    if (T == 1) {
        ggml_cuda_pool_alloc<block_q8_1> yq(ctx.pool(), (size_t) nb);
        inkling_quantize_sigmoid_mul_q8_1_f32<<<(K + 255) / 256, 256, 0, stream>>>(
            (const float *) x->data, (const float *) gate->data, yq.get(), K);

        constexpr int NWARPS = 4;
        const dim3 block_dims(32, NWARPS, 1);
        const dim3 grid_dims((nrows + NWARPS - 1) / NWARPS, 1, 1);
        if (nb == 96) {
            inkling_q8_0_mmv_f32<NWARPS, 96><<<grid_dims, block_dims, 0, stream>>>(
                (const block_q8_0 *) w->data, yq.get(), (float *) dst->data, nrows, 96);
        } else {
            inkling_q8_0_mmv_f32<NWARPS, 0><<<grid_dims, block_dims, 0, stream>>>(
                (const block_q8_0 *) w->data, yq.get(), (float *) dst->data, nrows, nb);
        }
        return;
    }

    ggml_cuda_pool_alloc<block_q8_1> yq(ctx.pool(), (size_t) T * nb);
    inkling_quantize_sigmoid_mul_q8_1_batched_f32<<<dim3((K + 255) / 256, T, 1), 256, 0, stream>>>(
        (const float *) x->data, (const float *) gate->data, yq.get(), K, nb);

    constexpr int NWARPS = 8;
    if (nb == 96) {
        inkling_launch_q8_0_mmv_batched<NWARPS, 96>((const block_q8_0 *) w->data, yq.get(), (float *) dst->data, nrows, 96, T, stream);
    } else {
        inkling_launch_q8_0_mmv_batched<NWARPS, 0>((const block_q8_0 *) w->data, yq.get(), (float *) dst->data, nrows, nb, T, stream);
    }
}

// Fast 1-warp/row Q8_0 GEMV for large-N projections (e.g. lm_head [201024 x 2304]: beats stock 4w/row mmvq by ~7%!)
static inline void ggml_cuda_op_inkling_q8_0_mmv_fast(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * w,
        const ggml_tensor * x,
        ggml_tensor * dst) {
    const int K     = (int) w->ne[0];
    const int nrows = (int) w->ne[1];
    const int nb    = K / QK8_0;
    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<block_q8_1> yq(ctx.pool(), (size_t) nb);
    inkling_ptr3 qptrs{};
    qptrs.p[0] = (const float *) x->data;
    inkling_quantize_q8_1_multi_f32<<<dim3((K + 255) / 256, 1, 1), 256, 0, stream>>>(qptrs, yq.get(), K, nb);

    constexpr int NWARPS = 8;
    const dim3 block_dims(32, NWARPS, 1);
    const dim3 grid_dims((nrows + NWARPS - 1) / NWARPS, 1, 1);
    if (nb == 72) {
        inkling_q8_0_mmv_f32<NWARPS, 72><<<grid_dims, block_dims, 0, stream>>>(
            (const block_q8_0 *) w->data, yq.get(), (float *) dst->data, nrows, 72);
    } else {
        inkling_q8_0_mmv_f32<NWARPS, 0><<<grid_dims, block_dims, 0, stream>>>(
            (const block_q8_0 *) w->data, yq.get(), (float *) dst->data, nrows, nb);
    }
}

// 12c. Fused 11-op depth-attention mix() + RMS_NORM + weight MUL (Q == 1, D == 2304, K <= 8):
//      Eliminates ~44 FUSED_RMS_NORM..MUL launches + ~44 [2304] VRAM roundtrips per token!
static __global__ void inkling_fused_mix_norm_f32x4_576(
        const float4 * __restrict__ blocks_cat,
        const float4 * __restrict__ cur,
        const float4 * __restrict__ query,
        const float4 * __restrict__ norm_w,
        float4 * __restrict__ dst,
        block_q8_1 * __restrict__ yq_out,
        const int K,
        const float mix_eps,
        const float sm_scale,
        const float norm_eps) {
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int t    = blockIdx.x;

    const float4 * blk_t = blocks_cat + (size_t) t * (K - 1) * 576;
    const float4 * cur_t = cur        + (size_t) t * 576;
    float4 *       dst_t = dst        + (size_t) t * 576;

    const float4 qv = query[tid];

    __shared__ float s_ss[8][18];
    __shared__ float s_dot[8][18];
    __shared__ float s_eff[8];
    __shared__ float s_out_ss[18];
    __shared__ float s_out_inv;

    float4 xv[8];
    for (int k = 0; k < K; ++k) {
        const float4 x = (k < K - 1) ? blk_t[(size_t) k * 576 + tid] : cur_t[tid];
        xv[k] = x;

        const float ss = warp_reduce_sum<32>(x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w);
        const float dd = warp_reduce_sum<32>(x.x * qv.x + x.y * qv.y + x.z * qv.z + x.w * qv.w);
        if (lane == 0) {
            s_ss[k][warp]  = ss;
            s_dot[k][warp] = dd;
        }
    }
    __syncthreads();

    if (warp == 0) {
        float inv_rms[8];
        float logits[8];
        float max_l = -INFINITY;
        for (int k = 0; k < K; ++k) {
            const float vs = (lane < 18) ? s_ss[k][lane]  : 0.0f;
            const float vd = (lane < 18) ? s_dot[k][lane] : 0.0f;
            const float total_ss  = warp_reduce_sum<32>(vs);
            const float total_dot = warp_reduce_sum<32>(vd);
            inv_rms[k] = rsqrtf(total_ss * (1.0f / 2304.0f) + mix_eps);
            logits[k]  = total_dot * inv_rms[k] * sm_scale;
            if (logits[k] > max_l) {
                max_l = logits[k];
            }
        }
        if (lane == 0) {
            float sum_e = 0.0f;
            float exps[8];
            for (int k = 0; k < K; ++k) {
                exps[k] = expf(logits[k] - max_l);
                sum_e += exps[k];
            }
            const float inv_sum = 1.0f / sum_e;
            for (int k = 0; k < K; ++k) {
                s_eff[k] = exps[k] * inv_sum * inv_rms[k];
            }
        }
    }
    __syncthreads();

    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int k = 0; k < K; ++k) {
        const float w = s_eff[k];
        acc.x += w * xv[k].x;
        acc.y += w * xv[k].y;
        acc.z += w * xv[k].z;
        acc.w += w * xv[k].w;
    }

    const float oss = warp_reduce_sum<32>(acc.x * acc.x + acc.y * acc.y + acc.z * acc.z + acc.w * acc.w);
    if (lane == 0) {
        s_out_ss[warp] = oss;
    }
    __syncthreads();

    if (warp == 0) {
        const float v = (lane < 18) ? s_out_ss[lane] : 0.0f;
        const float tot = warp_reduce_sum<32>(v);
        if (lane == 0) {
            s_out_inv = rsqrtf(tot * (1.0f / 2304.0f) + norm_eps);
        }
    }
    __syncthreads();

    const float inv = s_out_inv;
    const float4 nw = norm_w[tid];
    const float4 out = make_float4(acc.x * inv * nw.x, acc.y * inv * nw.y, acc.z * inv * nw.z, acc.w * inv * nw.w);
    dst_t[tid] = out;
    if (yq_out != nullptr && t == 0) {
        inkling_store_q8_1_from_f32x4(out, yq_out, tid >> 3, tid & 7);
    }
}

static inline void ggml_cuda_op_inkling_mix_norm(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * blocks_cat,
        const ggml_tensor * cur,
        const ggml_tensor * query,
        const ggml_tensor * norm_w,
        ggml_tensor * dst,
        const int K,
        const int T,
        const float mix_eps,
        const float sm_scale,
        const float norm_eps,
        block_q8_1 * yq_out = nullptr) {
    inkling_fused_mix_norm_f32x4_576<<<T, 576, 0, ctx.stream()>>>(
        (const float4 *) blocks_cat->data,
        (const float4 *) cur->data,
        (const float4 *) query->data,
        (const float4 *) norm_w->data,
        (float4 *) dst->data,
        yq_out,
        K, mix_eps, sm_scale, norm_eps);
}

// 13. Pre-interleaved Q8_1 quantizer + NVFP4 raw PRMT unpack (eliminates all 8 deinterleave PRMTs per block!)
static __device__ __forceinline__ int2 inkling_unpack_nvfp4_raw(const uint32_t q4) {
    // kvalues_fp4 = {0,1,2,3, 4,6,8,12, 0,-1,-2,-3, -4,-6,-8,-12}
    const uint32_t sel = 0x32103210u | ((q4 & 0x88888888u) >> 1);
    const uint32_t lo0 = __byte_perm(0x03020100u, 0x0C080604u, q4);
    const uint32_t hi0 = __byte_perm(0xFDFEFF00u, 0xF4F8FAFCu, q4);
    const uint32_t t0  = __byte_perm(lo0, hi0, sel);
    const uint32_t q4h = q4 >> 16;
    const uint32_t lo1 = __byte_perm(0x03020100u, 0x0C080604u, q4h);
    const uint32_t hi1 = __byte_perm(0xFDFEFF00u, 0xF4F8FAFCu, q4h);
    const uint32_t t1  = __byte_perm(lo1, hi1, sel >> 16);
    return make_int2(t0, t1);
}

static __device__ __forceinline__ float inkling_vec_dot_nvfp4_preint(
        const block_nvfp4 * __restrict__ bq4,
        const block_q8_1 * __restrict__ bq8_1,
        const int iqs) {
    const int is0 = iqs >> 1; // 0 or 2
    const block_q8_1 * bq8 = bq8_1 + (is0 >> 1);
    const int * q4p = (const int *) bq4->qs + iqs;
    const uint32_t q0 = __ldg(q4p + 0), q1 = __ldg(q4p + 1), q2 = __ldg(q4p + 2), q3 = __ldg(q4p + 3);
    const uint16_t dp = *reinterpret_cast<const uint16_t *>(bq4->d + is0);
    const int * u = (const int *) bq8->qs;
    const int2 v0 = inkling_unpack_nvfp4_raw(q0);
    const int2 v1 = inkling_unpack_nvfp4_raw(q1);
    int s0 = ggml_cuda_dp4a(v0.x, __ldg(u + 0), 0);
    s0     = ggml_cuda_dp4a(v0.y, __ldg(u + 1), s0);
    s0     = ggml_cuda_dp4a(v1.x, __ldg(u + 2), s0);
    s0     = ggml_cuda_dp4a(v1.y, __ldg(u + 3), s0);
    const int2 v2 = inkling_unpack_nvfp4_raw(q2);
    const int2 v3 = inkling_unpack_nvfp4_raw(q3);
    int s1 = ggml_cuda_dp4a(v2.x, __ldg(u + 4), 0);
    s1     = ggml_cuda_dp4a(v2.y, __ldg(u + 5), s1);
    s1     = ggml_cuda_dp4a(v3.x, __ldg(u + 6), s1);
    s1     = ggml_cuda_dp4a(v3.y, __ldg(u + 7), s1);
    return __low2float(bq8->ds) * (ggml_cuda_ue4m3_to_fp32(dp & 0xFFu) * (float) s0 + ggml_cuda_ue4m3_to_fp32(dp >> 8) * (float) s1);
}

static __global__ void inkling_quantize_q8_1_nvfp4_preint_f32(
        const float * __restrict__ x,
        block_q8_1 * __restrict__ y,
        const int ne0,
        const int nrows) {
    const int i0  = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (i0 >= ne0 || row >= nrows) {
        return;
    }
    const float xi = x[(size_t) row * ne0 + i0];
    const float amax = warp_reduce_max<32>(fabsf(xi));
    const float d = amax / 127.0f;
    const int8_t q = (amax == 0.0f) ? 0 : (int8_t) roundf(xi / d);
    const int ib  = (row * ne0 + i0) >> 5;
    const int iqs = i0 & 31;
    const int dst_iqs = (iqs & 16) | ((iqs & 7) << 1) | ((iqs >> 3) & 1);
    y[ib].qs[dst_iqs] = q;
    if (iqs == 0) {
        y[ib].ds = make_half2(__float2half(d), __float2half(0.0f));
    }
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
static __device__ __forceinline__ void inkling_cp_async_cg_16(void * smem_ptr, const void * gmem_ptr) {
    const uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem_ptr));
}
static __device__ __forceinline__ void inkling_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
template <int N>
static __device__ __forceinline__ void inkling_cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}
#endif

// 14. Fused 16-op NVFP4 gate_up_exps + per-expert scale + SiTU (T == 1):
//     Ampere SM >= 80 path uses 16-byte cp.async.cg.shared.global (99.9% of pure hardware memcpy ceiling!).
template <int NWARPS>
static __global__ void inkling_fused_nvfp4_gate_up_situ_cpasync_nb36(
        const block_nvfp4 * __restrict__ vx,
        const block_q8_1 * __restrict__ vy,
        const int32_t * __restrict__ ids,
        const float * __restrict__ sc,
        float * __restrict__ dst,
        const int n_ff,
        const size_t stride_ch_x,
        const float inv_beta, const float beta,
        const float inv_lb,   const float lb,
        const bool has_lb) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // 36 block_nvfp4 = 1296 bytes = 81 uint4s (100% 16B aligned for every row!)
    __shared__ __align__(16) uint32_t s_row[NWARPS][324];
    const int row = blockIdx.x * NWARPS + threadIdx.y;
    if (row >= n_ff) {
        return;
    }
    const int e   = blockIdx.y;
    const int ch  = ids[e];
    const int tid = threadIdx.x;
    const uint4 * xg4 = (const uint4 *) (vx + (size_t) ch * stride_ch_x + (size_t) row * 36);
    const uint4 * xu4 = (const uint4 *) (vx + (size_t) ch * stride_ch_x + (size_t) (row + n_ff) * 36);
    uint4 * sm4 = (uint4 *) s_row[threadIdx.y];

    auto run_half = [&](const uint4 * src4) -> float {
        inkling_cp_async_cg_16(&sm4[tid],      &src4[tid]);
        inkling_cp_async_cg_16(&sm4[tid + 32], &src4[tid + 32]);
        if (tid < 17) {
            inkling_cp_async_cg_16(&sm4[tid + 64], &src4[tid + 64]);
        }
        inkling_cp_async_commit();
        inkling_cp_async_wait<0>();
        __syncwarp();
        const block_nvfp4 * sb = (const block_nvfp4 *) s_row[threadIdx.y];
        const int iqs = (tid & 1) * 4;
        const int is0 = iqs >> 1;
        float sum = 0.0f;
#pragma unroll
        for (int k = tid >> 1; k < 36; k += 16) {
            const block_nvfp4 * bq4 = &sb[k];
            const block_q8_1  * bq8 = &vy[k * 2 + (is0 >> 1)];
            const int * q4p = (const int *) bq4->qs + iqs;
            const uint16_t dp = *reinterpret_cast<const uint16_t *>(bq4->d + is0);
            const int * u = (const int *) bq8->qs;
            const int2 v0 = inkling_unpack_nvfp4_raw(q4p[0]);
            const int2 v1 = inkling_unpack_nvfp4_raw(q4p[1]);
            int s0 = ggml_cuda_dp4a(v0.x, __ldg(u + 0), 0);
            s0     = ggml_cuda_dp4a(v0.y, __ldg(u + 1), s0);
            s0     = ggml_cuda_dp4a(v1.x, __ldg(u + 2), s0);
            s0     = ggml_cuda_dp4a(v1.y, __ldg(u + 3), s0);
            const int2 v2 = inkling_unpack_nvfp4_raw(q4p[2]);
            const int2 v3 = inkling_unpack_nvfp4_raw(q4p[3]);
            int s1 = ggml_cuda_dp4a(v2.x, __ldg(u + 4), 0);
            s1     = ggml_cuda_dp4a(v2.y, __ldg(u + 5), s1);
            s1     = ggml_cuda_dp4a(v3.x, __ldg(u + 6), s1);
            s1     = ggml_cuda_dp4a(v3.y, __ldg(u + 7), s1);
            sum += __low2float(bq8->ds) * (ggml_cuda_ue4m3_to_fp32(dp & 0xFFu) * (float) s0 + ggml_cuda_ue4m3_to_fp32(dp >> 8) * (float) s1);
        }
        return sum;
    };

    float sum_g = run_half(xg4);
    __syncwarp();
    float sum_u = run_half(xu4);

    sum_g = warp_reduce_sum<32>(sum_g);
    sum_u = warp_reduce_sum<32>(sum_u);
    if (tid == 0) {
        const float s = sc ? sc[ch] : 1.0f;
        dst[(size_t) e * n_ff + row] = inkling_situ_elem(sum_g * s, sum_u * s, inv_beta, beta, inv_lb, lb, has_lb);
    }
#endif
}

template <int NWARPS>
static __global__ void inkling_fused_nvfp4_gate_up_situ_gen(
        const block_nvfp4 * __restrict__ vx,
        const block_q8_1 * __restrict__ vy,
        const int32_t * __restrict__ ids,
        const float * __restrict__ sc,
        float * __restrict__ dst,
        const int n_ff,
        const int nb,
        const size_t stride_ch_x,
        const float inv_beta, const float beta,
        const float inv_lb,   const float lb,
        const bool has_lb) {
    const int row = blockIdx.x * NWARPS + threadIdx.y;
    if (row >= n_ff) {
        return;
    }
    const int e   = blockIdx.y;
    const int ch  = ids[e];
    const int tid = threadIdx.x;
    const block_nvfp4 * xg = vx + (size_t) ch * stride_ch_x + (size_t) row * nb;
    const block_nvfp4 * xu = xg + (size_t) n_ff * nb;
    const int iqs = (tid & 1) * 4;
    float sum_g = 0.0f, sum_u = 0.0f;
    for (int k = tid >> 1; k < nb; k += 16) {
        sum_g += inkling_vec_dot_nvfp4_preint(&xg[k], &vy[k * 2], iqs);
        sum_u += inkling_vec_dot_nvfp4_preint(&xu[k], &vy[k * 2], iqs);
    }
    sum_g = warp_reduce_sum<32>(sum_g);
    sum_u = warp_reduce_sum<32>(sum_u);
    if (tid == 0) {
        const float s = sc ? sc[ch] : 1.0f;
        dst[(size_t) e * n_ff + row] = inkling_situ_elem(sum_g * s, sum_u * s, inv_beta, beta, inv_lb, lb, has_lb);
    }
}

static inline void ggml_cuda_op_inkling_nvfp4_gate_up_situ(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * scale,
        ggml_tensor * dst,
        const float inv_beta, const float beta,
        const float inv_lb,   const float lb,
        const bool has_lb) {
    const int K     = (int) src0->ne[0];
    const int n_ff  = (int) (src0->ne[1] / 2);
    const int n_exp = (int) dst->ne[1];
    const int nb    = K / QK_NVFP4;
    const size_t stride_ch_x = src0->nb[2] / sizeof(block_nvfp4);
    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<block_q8_1> y_q8(ctx.pool(), (size_t) (K / QK8_1));
    inkling_quantize_q8_1_nvfp4_preint_f32<<<dim3((K + 255) / 256, 1, 1), 256, 0, stream>>>(
        (const float *) src1->data, y_q8.get(), K, 1);

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const bool use_cpasync = (cc >= GGML_CUDA_CC_AMPERE) && (nb == 36) &&
        (((uintptr_t) src0->data & 15) == 0) &&
        (((stride_ch_x * sizeof(block_nvfp4)) & 15) == 0);

    if (use_cpasync) {
        constexpr int NWARPS = 8;
        inkling_fused_nvfp4_gate_up_situ_cpasync_nb36<NWARPS><<<dim3((n_ff + NWARPS - 1) / NWARPS, n_exp, 1), dim3(32, NWARPS, 1), 0, stream>>>(
            (const block_nvfp4 *) src0->data, y_q8.get(), (const int32_t *) ids->data,
            scale ? (const float *) scale->data : nullptr, (float *) dst->data,
            n_ff, stride_ch_x, inv_beta, beta, inv_lb, lb, has_lb);
    } else {
        constexpr int NWARPS = 4;
        inkling_fused_nvfp4_gate_up_situ_gen<NWARPS><<<dim3((n_ff + NWARPS - 1) / NWARPS, n_exp, 1), dim3(32, NWARPS, 1), 0, stream>>>(
            (const block_nvfp4 *) src0->data, y_q8.get(), (const int32_t *) ids->data,
            scale ? (const float *) scale->data : nullptr, (float *) dst->data,
            n_ff, nb, stride_ch_x, inv_beta, beta, inv_lb, lb, has_lb);
    }
}

// 15. Fused 21-op NVFP4 down_exps + per-expert scale + moe_weighted_reduction (T == 1):
//     Flattens 8 experts x 18 blocks = 144 / 16 = 9 EXACT iterations with ZERO idle lanes!
template <int NWARPS>
static __global__ void inkling_fused_nvfp4_down_reduce_8x18(
        const block_nvfp4 * __restrict__ vx,
        const block_q8_1 * __restrict__ vy,
        const int32_t * __restrict__ ids,
        const float * __restrict__ sc,
        const float * __restrict__ rw,
        float * __restrict__ dst,
        const int nrows,
        const size_t stride_ch_x) {
    __shared__ int32_t s_exp[8];
    __shared__ float   s_w[8];
    if (threadIdx.y == 0 && threadIdx.x < 8) {
        const int ch = ids[threadIdx.x];
        s_exp[threadIdx.x] = ch;
        s_w[threadIdx.x]   = (sc ? sc[ch] : 1.0f) * rw[threadIdx.x];
    }
    __syncthreads();

    const int row = blockIdx.x * NWARPS + threadIdx.y;
    if (row >= nrows) {
        return;
    }
    const int tid = threadIdx.x;
    const int iqs = (tid & 1) * 4;
    float total = 0.0f;
    int e = 0;
    int k = tid >> 1; // 0..15
#pragma unroll
    for (int iter = 0; iter < 9; ++iter) {
        const int ch = s_exp[e];
        total += inkling_vec_dot_nvfp4_preint(
            vx + (size_t) ch * stride_ch_x + (size_t) row * 18 + k,
            vy + (size_t) e * 36 + k * 2, iqs) * s_w[e];
        k += 16;
        if (k >= 18) {
            k -= 18;
            e++;
        }
    }
    total = warp_reduce_sum<32>(total);
    if (tid == 0) {
        dst[row] = total;
    }
}

template <int NWARPS>
static __global__ void inkling_fused_nvfp4_down_reduce_gen(
        const block_nvfp4 * __restrict__ vx,
        const block_q8_1 * __restrict__ vy,
        const int32_t * __restrict__ ids,
        const float * __restrict__ sc,
        const float * __restrict__ rw,
        float * __restrict__ dst,
        const int nrows,
        const int nb,
        const int n_exp,
        const size_t stride_ch_x) {
    __shared__ int32_t s_exp[16];
    __shared__ float   s_w[16];
    if (threadIdx.y == 0 && threadIdx.x < n_exp) {
        const int ch = ids[threadIdx.x];
        s_exp[threadIdx.x] = ch;
        s_w[threadIdx.x]   = (sc ? sc[ch] : 1.0f) * rw[threadIdx.x];
    }
    __syncthreads();

    const int row = blockIdx.x * NWARPS + threadIdx.y;
    if (row >= nrows) {
        return;
    }
    const int tid = threadIdx.x;
    const int iqs = (tid & 1) * 4;
    const int total_blks = n_exp * nb;
    float total = 0.0f;
    for (int idx = tid >> 1; idx < total_blks; idx += 16) {
        const int e  = idx / nb;
        const int k  = idx - e * nb;
        const int ch = s_exp[e];
        total += inkling_vec_dot_nvfp4_preint(
            vx + (size_t) ch * stride_ch_x + (size_t) row * nb + k,
            vy + (size_t) (e * nb + k) * 2, iqs) * s_w[e];
    }
    total = warp_reduce_sum<32>(total);
    if (tid == 0) {
        dst[row] = total;
    }
}

static inline void ggml_cuda_op_inkling_nvfp4_down_reduce(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * scale,
        const ggml_tensor * router_w,
        ggml_tensor * dst) {
    const int K     = (int) src0->ne[0];
    const int nrows = (int) src0->ne[1];
    const int n_exp = (int) src1->ne[1];
    const int nb    = K / QK_NVFP4;
    const size_t stride_ch_x = src0->nb[2] / sizeof(block_nvfp4);
    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<block_q8_1> y_q8(ctx.pool(), (size_t) n_exp * (K / QK8_1));
    inkling_quantize_q8_1_nvfp4_preint_f32<<<dim3((K + 255) / 256, n_exp, 1), 256, 0, stream>>>(
        (const float *) src1->data, y_q8.get(), K, n_exp);

    constexpr int NWARPS = 4;
    const dim3 block_dims(32, NWARPS, 1);
    const dim3 grid_dims((nrows + NWARPS - 1) / NWARPS, 1, 1);
    if (n_exp == 8 && nb == 18) {
        inkling_fused_nvfp4_down_reduce_8x18<NWARPS><<<grid_dims, block_dims, 0, stream>>>(
            (const block_nvfp4 *) src0->data, y_q8.get(), (const int32_t *) ids->data,
            scale ? (const float *) scale->data : nullptr, (const float *) router_w->data,
            (float *) dst->data, nrows, stride_ch_x);
    } else {
        inkling_fused_nvfp4_down_reduce_gen<NWARPS><<<grid_dims, block_dims, 0, stream>>>(
            (const block_nvfp4 *) src0->data, y_q8.get(), (const int32_t *) ids->data,
            scale ? (const float *) scale->data : nullptr, (const float *) router_w->data,
            (float *) dst->data, nrows, nb, n_exp, stride_ch_x);
    }
}

// 16. Fused Q8_0 MoE gate_up_exps + SiTU (12 ops -> 1 kernel, T == 1) & down_exps + moe_weighted_reduction (17 ops -> 1 kernel, T == 1)
static __global__ void inkling_quantize_q8_1_rows_f32(
        const float * __restrict__ x,
        block_q8_1 * __restrict__ y,
        const int K,
        const int nrows) {
    const int i0  = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (i0 >= K || row >= nrows) {
        return;
    }
    const float xi = x[(size_t) row * K + i0];
    const float amax = warp_reduce_max<32>(fabsf(xi));
    const float d = amax / 127.0f;
    const int8_t q = (amax == 0.0f) ? 0 : (int8_t) roundf(xi / d);
    const int ib  = (row * K + i0) >> 5;
    const int iqs = i0 & 31;
    y[ib].qs[iqs] = q;
    if (iqs == 0) {
        y[ib].ds = make_half2(__float2half(d), __float2half(0.0f));
    }
}

template <int NWARPS>
static __global__ void inkling_fused_q8_0_gate_up_situ_f32(
        const block_q8_0 * __restrict__ vx,
        const block_q8_1 * __restrict__ vy,
        const int32_t * __restrict__ ids,
        float * __restrict__ dst,
        const int n_ff,
        const int nb,
        const size_t stride_ch_x,
        const float inv_beta, const float beta,
        const float inv_lb,   const float lb,
        const bool has_lb) {
    const int row = blockIdx.x * NWARPS + threadIdx.y;
    if (row >= n_ff) {
        return;
    }
    const int e   = blockIdx.y;
    const int ch  = ids[e];
    const int tid = threadIdx.x;
    const block_q8_0 * xg = vx + (size_t) ch * stride_ch_x + (size_t) row * nb;
    const block_q8_0 * xu = xg + (size_t) n_ff * nb;
    const int iqs = (tid & 3) * 2;
    float sum_g = 0.0f, sum_u = 0.0f;
    for (int k = tid >> 2; k < nb; k += 8) {
        const block_q8_1 * by = &vy[k];
        const int * uy = (const int *) by->qs;
        const int u0 = __ldg(uy + iqs + 0);
        const int u1 = __ldg(uy + iqs + 1);
        const float dy = __low2float(by->ds);
        {
            const block_q8_0 * bx = &xg[k];
            int s = ggml_cuda_dp4a(inkling_get_int_b2(bx->qs, iqs + 0), u0, 0);
            s     = ggml_cuda_dp4a(inkling_get_int_b2(bx->qs, iqs + 1), u1, s);
            sum_g += (float) s * __half2float(__ldg(&bx->d)) * dy;
        }
        {
            const block_q8_0 * bx = &xu[k];
            int s = ggml_cuda_dp4a(inkling_get_int_b2(bx->qs, iqs + 0), u0, 0);
            s     = ggml_cuda_dp4a(inkling_get_int_b2(bx->qs, iqs + 1), u1, s);
            sum_u += (float) s * __half2float(__ldg(&bx->d)) * dy;
        }
    }
    sum_g = warp_reduce_sum<32>(sum_g);
    sum_u = warp_reduce_sum<32>(sum_u);
    if (tid == 0) {
        dst[(size_t) e * n_ff + row] = inkling_situ_elem(sum_g, sum_u, inv_beta, beta, inv_lb, lb, has_lb);
    }
}

static inline void ggml_cuda_op_inkling_q8_0_gate_up_situ(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        ggml_tensor * dst,
        const float inv_beta, const float beta,
        const float inv_lb,   const float lb,
        const bool has_lb) {
    const int K     = (int) src0->ne[0];
    const int n_ff  = (int) (src0->ne[1] / 2);
    const int n_exp = (int) dst->ne[1];
    const int nb    = K / QK8_0;
    const size_t stride_ch_x = src0->nb[2] / sizeof(block_q8_0);
    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<block_q8_1> y_q8(ctx.pool(), (size_t) nb);
    inkling_quantize_q8_1_rows_f32<<<dim3((K + 255) / 256, 1, 1), 256, 0, stream>>>(
        (const float *) src1->data, y_q8.get(), K, 1);

    constexpr int NWARPS = 4;
    inkling_fused_q8_0_gate_up_situ_f32<NWARPS><<<dim3((n_ff + NWARPS - 1) / NWARPS, n_exp, 1), dim3(32, NWARPS, 1), 0, stream>>>(
        (const block_q8_0 *) src0->data, y_q8.get(), (const int32_t *) ids->data,
        (float *) dst->data, n_ff, nb, stride_ch_x, inv_beta, beta, inv_lb, lb, has_lb);
}

template <int NWARPS>
static __global__ void inkling_fused_q8_0_down_reduce_f32(
        const block_q8_0 * __restrict__ vx,
        const block_q8_1 * __restrict__ vy,
        const int32_t * __restrict__ ids,
        const float * __restrict__ rw,
        float * __restrict__ dst,
        const int nrows,
        const int nb,
        const int n_exp,
        const size_t stride_ch_x) {
    __shared__ int32_t s_exp[16];
    __shared__ float   s_w[16];
    if (threadIdx.y == 0 && threadIdx.x < n_exp) {
        s_exp[threadIdx.x] = ids[threadIdx.x];
        s_w[threadIdx.x]   = rw[threadIdx.x];
    }
    __syncthreads();

    const int row = blockIdx.x * NWARPS + threadIdx.y;
    if (row >= nrows) {
        return;
    }
    const int tid = threadIdx.x;
    const int iqs = (tid & 3) * 2;
    const int total_blks = n_exp * nb;
    float total = 0.0f;
    if (nb >= 8) {
        int e = 0;
        int k = tid >> 2;
        for (int idx = tid >> 2; idx < total_blks; idx += 8) {
            const int ch = s_exp[e];
            const block_q8_0 * bx = vx + (size_t) ch * stride_ch_x + (size_t) row * nb + k;
            const block_q8_1 * by = vy + (size_t) e * nb + k;
            const int * uy = (const int *) by->qs;
            int s = ggml_cuda_dp4a(inkling_get_int_b2(bx->qs, iqs + 0), __ldg(uy + iqs + 0), 0);
            s     = ggml_cuda_dp4a(inkling_get_int_b2(bx->qs, iqs + 1), __ldg(uy + iqs + 1), s);
            total += (float) s * __half2float(__ldg(&bx->d)) * __low2float(by->ds) * s_w[e];
            k += 8;
            if (k >= nb) {
                k -= nb;
                e++;
            }
        }
    } else {
        for (int idx = tid >> 2; idx < total_blks; idx += 8) {
            const int e  = idx / nb;
            const int k  = idx - e * nb;
            const int ch = s_exp[e];
            const block_q8_0 * bx = vx + (size_t) ch * stride_ch_x + (size_t) row * nb + k;
            const block_q8_1 * by = vy + (size_t) e * nb + k;
            const int * uy = (const int *) by->qs;
            int s = ggml_cuda_dp4a(inkling_get_int_b2(bx->qs, iqs + 0), __ldg(uy + iqs + 0), 0);
            s     = ggml_cuda_dp4a(inkling_get_int_b2(bx->qs, iqs + 1), __ldg(uy + iqs + 1), s);
            total += (float) s * __half2float(__ldg(&bx->d)) * __low2float(by->ds) * s_w[e];
        }
    }
    total = warp_reduce_sum<32>(total);
    if (tid == 0) {
        dst[row] = total;
    }
}

static inline void ggml_cuda_op_inkling_q8_0_down_reduce(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * router_w,
        ggml_tensor * dst) {
    const int K     = (int) src0->ne[0];
    const int nrows = (int) src0->ne[1];
    const int n_exp = (int) src1->ne[1];
    const int nb    = K / QK8_0;
    const size_t stride_ch_x = src0->nb[2] / sizeof(block_q8_0);
    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<block_q8_1> y_q8(ctx.pool(), (size_t) n_exp * nb);
    inkling_quantize_q8_1_rows_f32<<<dim3((K + 255) / 256, n_exp, 1), 256, 0, stream>>>(
        (const float *) src1->data, y_q8.get(), K, n_exp);

    constexpr int NWARPS = 4;
    inkling_fused_q8_0_down_reduce_f32<NWARPS><<<dim3((nrows + NWARPS - 1) / NWARPS, 1, 1), dim3(32, NWARPS, 1), 0, stream>>>(
        (const block_q8_0 *) src0->data, y_q8.get(), (const int32_t *) ids->data,
        (const float *) router_w->data, (float *) dst->data, nrows, nb, n_exp, stride_ch_x);
}

// ============================================================================
// Wave 6C Batch 2 & Wave 6D Mega-Fusions (Batch-3 Control Split-K=2,
// Batch-3 RMS_NORM, Combined K+V shortconv4+clip/rms+set_rows+state,
// Q+R Clip Pair, and Fused 1-Kernel Router+TopK128)
// ============================================================================

// 17. Batch-3 Attention control() Split-K=2 (36 ops / 6 launches -> 2 launches, 64 -> 384 blocks, -46% latency!):
struct inkling_ctrl3_ptrs {
    const float4 * cur[3];
    const float4 * msg[3];
    const float4 * wd[3];
    const float4 * wf[3];
    const float4 * wg[3];
    float *        dst[3];
};

static __global__ void inkling_fused_control_down_3_split2_f32x4(
        const inkling_ctrl3_ptrs p,
        float * __restrict__ scratch,
        const int D4,
        const int R) {
    constexpr int SPLIT_K = 2;
    const int r    = blockIdx.x;
    const int sp   = blockIdx.y;
    const int b    = blockIdx.z;
    const int tid  = threadIdx.x; // 0..127 (4 warps)
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const float4 * cur = (b == 0) ? p.cur[0] : ((b == 1) ? p.cur[1] : p.cur[2]);
    const float4 * msg = (b == 0) ? p.msg[0] : ((b == 1) ? p.msg[1] : p.msg[2]);
    const float4 * wd  = (b == 0) ? p.wd[0]  : ((b == 1) ? p.wd[1]  : p.wd[2]);

    const int chunk = (D4 + SPLIT_K - 1) / SPLIT_K;
    const int i0    = sp * chunk;
    const int i1    = min(D4, i0 + chunk);

    const float4 * wc = wd + (size_t) r * (2 * D4);
    const float4 * wm = wc + D4;

    float dc = 0.0f, dm = 0.0f;
    float ss_c = 0.0f, ss_m = 0.0f;
    for (int i = i0 + tid; i < i1; i += 128) {
        const float4 cv  = __ldg(&cur[i]);
        const float4 mv  = __ldg(&msg[i]);
        const float4 wcv = __ldg(&wc[i]);
        const float4 wmv = __ldg(&wm[i]);
        dc += cv.x * wcv.x + cv.y * wcv.y + cv.z * wcv.z + cv.w * wcv.w;
        dm += mv.x * wmv.x + mv.y * wmv.y + mv.z * wmv.z + mv.w * wmv.w;
        if (r == 0) {
            ss_c += cv.x * cv.x + cv.y * cv.y + cv.z * cv.z + cv.w * cv.w;
            ss_m += mv.x * mv.x + mv.y * mv.y + mv.z * mv.z + mv.w * mv.w;
        }
    }
    dc = warp_reduce_sum<32>(dc);
    dm = warp_reduce_sum<32>(dm);
    __shared__ float sm[4][4];
    if (lane == 0) {
        sm[0][warp] = dc;
        sm[1][warp] = dm;
    }
    if (r == 0) {
        ss_c = warp_reduce_sum<32>(ss_c);
        ss_m = warp_reduce_sum<32>(ss_m);
        if (lane == 0) {
            sm[2][warp] = ss_c;
            sm[3][warp] = ss_m;
        }
    }
    __syncthreads();
    if (warp == 0 && lane == 0) {
        float * sb = scratch + (size_t) b * (4 * R + 4);
        sb[sp * R + r]       = sm[0][0] + sm[0][1] + sm[0][2] + sm[0][3];
        sb[(2 + sp) * R + r] = sm[1][0] + sm[1][1] + sm[1][2] + sm[1][3];
        if (r == 0) {
            sb[4 * R + sp]     = sm[2][0] + sm[2][1] + sm[2][2] + sm[2][3];
            sb[4 * R + 2 + sp] = sm[3][0] + sm[3][1] + sm[3][2] + sm[3][3];
        }
    }
}

static __global__ void inkling_fused_control_up_3_split2_f32x4(
        const inkling_ctrl3_ptrs p,
        const float * __restrict__ scratch,
        const int D,
        const int R,
        const float inv_D,
        const float eps,
        const float forget_s,
        const float forget_b) {
    const int b   = blockIdx.y;
    const int tid = threadIdx.x; // 128 threads = 8 half-warps
    const float * sb = scratch + (size_t) b * (4 * R + 4);

    __shared__ __align__(16) float s_h[128];
    if (tid < R) {
        const float ssc = sb[4 * R + 0] + sb[4 * R + 1];
        const float ssm = sb[4 * R + 2] + sb[4 * R + 3];
        const float irc = rsqrtf(ssc * inv_D + eps);
        const float irm = rsqrtf(ssm * inv_D + eps);
        const float dc  = sb[0 * R + tid] + sb[1 * R + tid];
        const float dm  = sb[2 * R + tid] + sb[3 * R + tid];
        s_h[tid] = dc * irc + dm * irm;
    }
    __syncthreads();

    const int lane16 = tid & 15;
    const int row    = blockIdx.x * 8 + (tid >> 4);
    if (row >= D) {
        return;
    }

    const float4 * wf = (b == 0) ? p.wf[0] : ((b == 1) ? p.wf[1] : p.wf[2]);
    const float4 * wg = (b == 0) ? p.wg[0] : ((b == 1) ? p.wg[1] : p.wg[2]);
    const float * cur = (const float *) ((b == 0) ? p.cur[0] : ((b == 1) ? p.cur[1] : p.cur[2]));
    const float * msg = (const float *) ((b == 0) ? p.msg[0] : ((b == 1) ? p.msg[1] : p.msg[2]));
    float * dst       = (b == 0) ? p.dst[0] : ((b == 1) ? p.dst[1] : p.dst[2]);

    const int R4 = R >> 2;
    const float4 * sh4 = (const float4 *) s_h;
    const float4 * wf4 = wf + (size_t) row * R4;
    const float4 * wg4 = wg + (size_t) row * R4;

    float df = 0.0f, dg = 0.0f;
    for (int k = lane16; k < R4; k += 16) {
        const float4 hv  = sh4[k];
        const float4 wfv = __ldg(&wf4[k]);
        const float4 wgv = __ldg(&wg4[k]);
        df += wfv.x * hv.x + wfv.y * hv.y + wfv.z * hv.z + wfv.w * hv.w;
        dg += wgv.x * hv.x + wgv.y * hv.y + wgv.z * hv.z + wgv.w * hv.w;
    }
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) {
        df += __shfl_xor_sync(0xffffffff, df, o, 16);
        dg += __shfl_xor_sync(0xffffffff, dg, o, 16);
    }
    if (lane16 == 0) {
        dst[row] = (tanhf(df) * forget_s + forget_b) * cur[row] + tanhf(dg) * msg[row];
    }
}

static inline void ggml_cuda_op_inkling_control_3(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * const cur[3],
        const ggml_tensor * const msg[3],
        const ggml_tensor * const wd[3],
        const ggml_tensor * const wf[3],
        const ggml_tensor * const wg[3],
        ggml_tensor * const dst[3],
        const float eps,
        const float forget_s,
        const float forget_b) {
    const int D  = (int) cur[0]->ne[0];
    const int T  = (int) cur[0]->ne[1];
    const int R  = (int) wd[0]->ne[1];
    const int D4 = D >> 2;
    cudaStream_t stream = ctx.stream();
    if (T >= 2 && T <= 8) {
        inkling_ctrl_batched_ptrs p{};
        for (int b = 0; b < 3; ++b) {
            p.cur[b] = (const float4 *) cur[b]->data;
            p.msg[b] = (const float4 *) msg[b]->data;
            p.wd[b]  = (const float4 *) wd[b]->data;
            p.wf[b]  = (const float4 *) wf[b]->data;
            p.wg[b]  = (const float4 *) wg[b]->data;
            p.dst[b] = (float *) dst[b]->data;
        }
        ggml_cuda_pool_alloc<float> scratch(ctx.pool(), (size_t) 3 * T * (4 * R + 4));
        inkling_launch_control_batched(p, scratch.get(), 3, D, R, T, eps, forget_s, forget_b, stream);
        return;
    }
    inkling_ctrl3_ptrs p{};
    for (int b = 0; b < 3; ++b) {
        p.cur[b] = (const float4 *) cur[b]->data;
        p.msg[b] = (const float4 *) msg[b]->data;
        p.wd[b]  = (const float4 *) wd[b]->data;
        p.wf[b]  = (const float4 *) wf[b]->data;
        p.wg[b]  = (const float4 *) wg[b]->data;
        p.dst[b] = (float *) dst[b]->data;
    }
    ggml_cuda_pool_alloc<float> scratch(ctx.pool(), (size_t) 3 * (4 * R + 4));
    inkling_fused_control_down_3_split2_f32x4<<<dim3(R, 2, 3), 128, 0, stream>>>(p, scratch.get(), D4, R);
    inkling_fused_control_up_3_split2_f32x4<<<dim3((D + 7) / 8, 3, 1), 128, 0, stream>>>(
        p, scratch.get(), D, R, 1.0f / (float) D, eps, forget_s, forget_b);
}

// 18. Batch-3 RMS_NORM + MUL (6 ops / 3 launches -> 1 launch on 28 layers, D == 2304):
struct inkling_rms3_ptrs {
    const float4 * x[3];
    const float4 * w[3];
    float4 *       dst[3];
};

static __global__ void inkling_fused_rms_norm_mul_3_f32x4_576(
        const inkling_rms3_ptrs p,
        block_q8_1 * __restrict__ yq_out,
        const float eps) {
    const int b    = blockIdx.x;
    const int t    = blockIdx.y;
    const int tid  = threadIdx.x; // 0..575 (18 warps)
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const float4 * xp = (b == 0) ? p.x[0] : ((b == 1) ? p.x[1] : p.x[2]);
    const float4 * wp = (b == 0) ? p.w[0] : ((b == 1) ? p.w[1] : p.w[2]);
    float4 *       dp = (b == 0) ? p.dst[0] : ((b == 1) ? p.dst[1] : p.dst[2]);

    const size_t off = (size_t) t * 576 + tid;
    const float4 xv  = xp[off];
    const float4 wv  = wp[tid];

    const float ss = warp_reduce_sum<32>(xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w);
    __shared__ float s_ss[18];
    __shared__ float s_inv;
    if (lane == 0) {
        s_ss[warp] = ss;
    }
    __syncthreads();
    if (warp == 0) {
        const float tot = warp_reduce_sum<32>(lane < 18 ? s_ss[lane] : 0.0f);
        if (lane == 0) {
            s_inv = rsqrtf(tot * (1.0f / 2304.0f) + eps);
        }
    }
    __syncthreads();
    const float inv = s_inv;
    const float4 out = make_float4(xv.x * inv * wv.x, xv.y * inv * wv.y, xv.z * inv * wv.z, xv.w * inv * wv.w);
    dp[off] = out;
    if (yq_out != nullptr && t == 0) {
        inkling_store_q8_1_from_f32x4(out, yq_out, b * 72 + (tid >> 3), tid & 7);
    }
}

static inline void ggml_cuda_op_inkling_rms_norm_mul_3(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * const x[3],
        const ggml_tensor * const w[3],
        ggml_tensor * const dst[3],
        const int T,
        const float eps,
        block_q8_1 * yq_out = nullptr) {
    inkling_rms3_ptrs p{};
    for (int b = 0; b < 3; ++b) {
        p.x[b]   = (const float4 *) x[b]->data;
        p.w[b]   = (const float4 *) w[b]->data;
        p.dst[b] = (float4 *) dst[b]->data;
    }
    inkling_fused_rms_norm_mul_3_f32x4_576<<<dim3(3, T, 1), 576, 0, ctx.stream()>>>(p, yq_out, eps);
}

// 18b. Single RMS_NORM + MUL + inline Q8_1 (for Layer 0 single-input attention -> horiz_mmv, D == 2304, T == 1)
static __global__ void inkling_fused_rms_norm_mul_1_q8_f32x4_576(
        const float4 * __restrict__ x,
        const float4 * __restrict__ w,
        float4 * __restrict__ dst,
        block_q8_1 * __restrict__ yq_out,
        const float eps) {
    const int tid  = threadIdx.x; // 0..575
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const float4 xv = x[tid];
    const float4 wv = w[tid];
    const float ss = warp_reduce_sum<32>(xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w);
    __shared__ float s_ss[18];
    __shared__ float s_inv;
    if (lane == 0) {
        s_ss[warp] = ss;
    }
    __syncthreads();
    if (warp == 0) {
        const float tot = warp_reduce_sum<32>(lane < 18 ? s_ss[lane] : 0.0f);
        if (lane == 0) {
            s_inv = rsqrtf(tot * (1.0f / 2304.0f) + eps);
        }
    }
    __syncthreads();
    const float inv = s_inv;
    const float4 out = make_float4(xv.x * inv * wv.x, xv.y * inv * wv.y, xv.z * inv * wv.z, xv.w * inv * wv.w);
    dst[tid] = out;
    inkling_store_q8_1_from_f32x4(out, yq_out, tid >> 3, tid & 7);
}

static inline void ggml_cuda_op_inkling_rms_norm_mul_1_q8(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * x,
        const ggml_tensor * w,
        ggml_tensor * dst,
        block_q8_1 * yq_out,
        const float eps) {
    inkling_fused_rms_norm_mul_1_q8_f32x4_576<<<1, 576, 0, ctx.stream()>>>(
        (const float4 *) x->data, (const float4 *) w->data, (float4 *) dst->data, yq_out, eps);
}

// 19. Fused K+V shortconv4 + K_clip/V_rms + SET_ROWS + STATE_STORE (192 tiny kernels -> 48 kernels!):
struct inkling_kv_conv_side {
    const float *   past;
    const int32_t * past_row_idx;
    size_t          past_row_stride_f;
    const float *   cur;
    const float *   weight;
    const float *   dyn;
    float *         dst_state;
    void *          dst_cache;
    const void *    row_idxs;
    size_t          past_s1;
    size_t          past_s2;
    size_t          w_s1;
    size_t          state_s1;
    size_t          state_plane_s;
    size_t          cache_nb1;
    int             kw;
    float           conv_s;
    float           conv_b;
    // K clip params (unused for V)
    float           clip_s;
    float           clip_b;
    float           clip_thr;
    float           clip_min;
    float           clip_max;
    // V rms_norm eps (unused for K)
    float           rms_eps;
};

// DST_FMT: 0 = F32, 1 = F16, 2 = Q8_0
template <typename idx_t, int DST_FMT, bool has_dynamic, int MODE>
// MODE: 0 = K-only, 1 = V-only, 2 = K+V combined
static __global__ void inkling_fused_kv_shortconv4_set_rows_warp(
        const inkling_kv_conv_side ks,
        const inkling_kv_conv_side vs,
        const int head_dim,
        const int n_head_kv,
        const int T,
        const int n_seqs) {
    const int widx = blockIdx.x * blockDim.y + threadIdx.y;
    const int per_side = n_seqs * n_head_kv;
    const int total_w  = (MODE == 2) ? (2 * per_side) : per_side;
    if (widx >= total_w) {
        return;
    }
    const bool is_v = (MODE == 1) || (MODE == 2 && widx >= per_side);
    const int local = (MODE == 2 && is_v) ? (widx - per_side) : widx;
    const int s     = local / n_head_kv;
    const int h     = local - s * n_head_kv;
    const int lane  = threadIdx.x;
    const int C     = head_dim * n_head_kv;

    const float *   past       = is_v ? vs.past              : ks.past;
    const int32_t * past_ridx  = is_v ? vs.past_row_idx      : ks.past_row_idx;
    const size_t    past_rstr  = is_v ? vs.past_row_stride_f : ks.past_row_stride_f;
    const float *   cur        = is_v ? vs.cur               : ks.cur;
    const float *   weight     = is_v ? vs.weight            : ks.weight;
    const float *   dyn        = is_v ? vs.dyn               : ks.dyn;
    float *         dst_st     = is_v ? vs.dst_state         : ks.dst_state;
    void *          dst_c      = is_v ? vs.dst_cache         : ks.dst_cache;
    const idx_t *   row_idxs   = (const idx_t *) (is_v ? vs.row_idxs : ks.row_idxs);
    const size_t    past_s1    = is_v ? vs.past_s1           : ks.past_s1;
    const size_t    past_s2    = is_v ? vs.past_s2           : ks.past_s2;
    const size_t    w_s1       = is_v ? vs.w_s1              : ks.w_s1;
    const size_t    state_s1   = is_v ? vs.state_s1          : ks.state_s1;
    const size_t    state_pl_s = is_v ? vs.state_plane_s     : ks.state_plane_s;
    const int       kw         = is_v ? vs.kw                : ks.kw;
    const size_t    cache_nb1  = is_v ? vs.cache_nb1         : ks.cache_nb1;
    const float     conv_s     = is_v ? vs.conv_s            : ks.conv_s;
    const float     conv_b     = is_v ? vs.conv_b            : ks.conv_b;

    const float * past_s = past_ridx ? (past + (size_t) past_ridx[s] * past_rstr) : (past + (size_t) s * past_s2);

    float s0[4], s1[4], s2[4];
    float w0[4], w1[4], w2[4], w3[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int cl = lane + k * 32;
        if (cl < head_dim) {
            const int c = h * head_dim + cl;
            const float * p  = past_s + (size_t) c * past_s1;
            const float * wp = weight + (size_t) c * w_s1;
            s0[k] = p[0]; s1[k] = p[1]; s2[k] = p[2];
            w0[k] = wp[0]; w1[k] = wp[1]; w2[k] = wp[2]; w3[k] = wp[3];
        }
    }

    for (int t = 0; t < T; ++t) {
        const int tok = s * T + t;
        float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
        if constexpr (has_dynamic) {
            const float * dp = dyn + ((size_t) tok * n_head_kv + h) * 4;
            d0 = dp[0]; d1 = dp[1]; d2 = dp[2]; d3 = dp[3];
        }
        float v[4];
        float ss = 0.0f;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int cl = lane + k * 32;
            if (cl < head_dim) {
                const int c = h * head_dim + cl;
                const float x = cur[(size_t) tok * C + c];
                float conv = s0[k] * w0[k] + s1[k] * w1[k] + s2[k] * w2[k] + x * w3[k];
                if constexpr (has_dynamic) {
                    conv += x * d0 + s2[k] * d1 + s1[k] * d2 + s0[k] * d3;
                }
                const float val = (x + conv) * conv_s + conv_b;
                v[k] = val;
                ss += val * val;
                s0[k] = s1[k];
                s1[k] = s2[k];
                s2[k] = x;
            }
        }
        ss = warp_reduce_sum<32>(ss);
        float fac;
        if (is_v) {
            fac = rsqrtf(ss / (float) head_dim + vs.rms_eps);
        } else {
            const float norm = sqrtf(ss) * ks.clip_s + ks.clip_b;
            fac = fminf(ks.clip_max, fmaxf(ks.clip_min, ks.clip_thr / norm));
        }
        const int64_t dst_row = (int64_t) row_idxs[tok];
        char * dst_base = (char *) dst_c + dst_row * cache_nb1;
        if constexpr (DST_FMT == 2) {
            const int blk_base = h * (head_dim >> 5);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                if (k * 32 < head_dim) {
                    const float out_v = v[k] * fac;
                    const float amax  = warp_reduce_max<32>(fabsf(out_v));
                    const float d     = amax / 127.0f;
                    const float id    = (d != 0.0f) ? (1.0f / d) : 0.0f;
                    block_q8_0 * blk  = (block_q8_0 *) dst_base + blk_base + k;
                    blk->qs[lane]     = (int8_t) fmaxf(-127.0f, fminf(127.0f, roundf(out_v * id)));
                    if (lane == 0) {
                        blk->d = __float2half(d);
                    }
                }
            }
        } else {
            const int col_base = h * head_dim;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int cl = lane + k * 32;
                if (cl < head_dim) {
                    const float out_v = v[k] * fac;
                    if constexpr (DST_FMT == 1) {
                        ((half *) dst_base)[col_base + cl] = __float2half(out_v);
                    } else {
                        ((float *) dst_base)[col_base + cl] = out_v;
                    }
                }
            }
        }
        if (kw > 1 && t >= T - kw) {
            const int rslot = T - 1 - t;
            float * ns_base = dst_st + (size_t) rslot * state_pl_s + (size_t) s * state_s1;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int cl = lane + k * 32;
                if (cl < head_dim) {
                    const int c = h * head_dim + cl;
                    float * ns = ns_base + (size_t) c * 3;
                    ns[0] = s0[k];
                    ns[1] = s1[k];
                    ns[2] = s2[k];
                }
            }
        }
    }

    if (kw <= 1) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int cl = lane + k * 32;
            if (cl < head_dim) {
                const int c = h * head_dim + cl;
                float * ns = dst_st + (size_t) s * state_s1 + (size_t) c * 3;
                ns[0] = s0[k];
                ns[1] = s1[k];
                ns[2] = s2[k];
            }
        }
    }
}

template <int MODE>
static inline void ggml_cuda_op_inkling_kv_shortconv4_set_rows(
        ggml_backend_cuda_context & ctx,
        const inkling_kv_conv_side & ks,
        const inkling_kv_conv_side & vs,
        const int head_dim,
        const int n_head_kv,
        const int T,
        const int n_seqs,
        const bool is_i64,
        const int dst_fmt,
        const bool is_dyn) {
    const int total_w = ((MODE == 2) ? 2 : 1) * n_seqs * n_head_kv;
    constexpr int warps_per_block = 4;
    const dim3 block_dims(32, warps_per_block, 1);
    const dim3 grid_dims((total_w + warps_per_block - 1) / warps_per_block, 1, 1);
    cudaStream_t stream = ctx.stream();
    const int key = (is_i64 ? 6 : 0) + dst_fmt * 2 + (is_dyn ? 1 : 0);
    switch (key) {
        case  0: inkling_fused_kv_shortconv4_set_rows_warp<int32_t, 0, false, MODE><<<grid_dims, block_dims, 0, stream>>>(ks, vs, head_dim, n_head_kv, T, n_seqs); break;
        case  1: inkling_fused_kv_shortconv4_set_rows_warp<int32_t, 0, true,  MODE><<<grid_dims, block_dims, 0, stream>>>(ks, vs, head_dim, n_head_kv, T, n_seqs); break;
        case  2: inkling_fused_kv_shortconv4_set_rows_warp<int32_t, 1, false, MODE><<<grid_dims, block_dims, 0, stream>>>(ks, vs, head_dim, n_head_kv, T, n_seqs); break;
        case  3: inkling_fused_kv_shortconv4_set_rows_warp<int32_t, 1, true,  MODE><<<grid_dims, block_dims, 0, stream>>>(ks, vs, head_dim, n_head_kv, T, n_seqs); break;
        case  4: inkling_fused_kv_shortconv4_set_rows_warp<int32_t, 2, false, MODE><<<grid_dims, block_dims, 0, stream>>>(ks, vs, head_dim, n_head_kv, T, n_seqs); break;
        case  5: inkling_fused_kv_shortconv4_set_rows_warp<int32_t, 2, true,  MODE><<<grid_dims, block_dims, 0, stream>>>(ks, vs, head_dim, n_head_kv, T, n_seqs); break;
        case  6: inkling_fused_kv_shortconv4_set_rows_warp<int64_t, 0, false, MODE><<<grid_dims, block_dims, 0, stream>>>(ks, vs, head_dim, n_head_kv, T, n_seqs); break;
        case  7: inkling_fused_kv_shortconv4_set_rows_warp<int64_t, 0, true,  MODE><<<grid_dims, block_dims, 0, stream>>>(ks, vs, head_dim, n_head_kv, T, n_seqs); break;
        case  8: inkling_fused_kv_shortconv4_set_rows_warp<int64_t, 1, false, MODE><<<grid_dims, block_dims, 0, stream>>>(ks, vs, head_dim, n_head_kv, T, n_seqs); break;
        case  9: inkling_fused_kv_shortconv4_set_rows_warp<int64_t, 1, true,  MODE><<<grid_dims, block_dims, 0, stream>>>(ks, vs, head_dim, n_head_kv, T, n_seqs); break;
        case 10: inkling_fused_kv_shortconv4_set_rows_warp<int64_t, 2, false, MODE><<<grid_dims, block_dims, 0, stream>>>(ks, vs, head_dim, n_head_kv, T, n_seqs); break;
        case 11: inkling_fused_kv_shortconv4_set_rows_warp<int64_t, 2, true,  MODE><<<grid_dims, block_dims, 0, stream>>>(ks, vs, head_dim, n_head_kv, T, n_seqs); break;
    }
}

// 20. Fused Q_clip + R_clip Pair (18-20 ops / 2 launches -> 1 launch on all 48 layers!):
struct inkling_clip_desc {
    const float * x;
    const float * scale;
    const float * tau;
    float *       dst;
    int           ne0;
    int           n_rows;
    int           scale_rows;
    int           tau_div;
    float         scale_s;
    float         scale_b;
    float         thr;
    float         cmin;
    float         cmax;
};

static __global__ void inkling_fused_l2_clip_pair_f32_warp(
        const inkling_clip_desc a,
        const inkling_clip_desc b) {
    const int g = blockIdx.x * blockDim.y + threadIdx.y;
    const int total = a.n_rows + b.n_rows;
    if (g >= total) {
        return;
    }
    const bool is_b = (g >= a.n_rows);
    const int row   = is_b ? (g - a.n_rows) : g;
    const int lane  = threadIdx.x;

    const float * x      = is_b ? b.x          : a.x;
    const float * scale  = is_b ? b.scale      : a.scale;
    const float * tau    = is_b ? b.tau        : a.tau;
    float *       dst    = is_b ? b.dst        : a.dst;
    const int     ne0    = is_b ? b.ne0        : a.ne0;
    const int     srows  = is_b ? b.scale_rows : a.scale_rows;
    const int     tdiv   = is_b ? b.tau_div    : a.tau_div;
    const float   sc_s   = is_b ? b.scale_s    : a.scale_s;
    const float   sc_b   = is_b ? b.scale_b    : a.scale_b;
    const float   thr    = is_b ? b.thr        : a.thr;
    const float   cmin   = is_b ? b.cmin       : a.cmin;
    const float   cmax   = is_b ? b.cmax       : a.cmax;

    const float * x_row   = x   + (size_t) row * ne0;
    float *       dst_row = dst + (size_t) row * ne0;
    const float * sc_row  = scale ? (scale + (size_t) (row % srows) * ne0) : nullptr;
    const float   tau_val = tau ? tau[row / tdiv] : 1.0f;

    float v[4];
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int c = lane + k * 32;
        if (c < ne0) {
            float val = x_row[c];
            if (sc_row) {
                val *= sc_row[c];
            }
            v[k] = val;
            ss += val * val;
        }
    }
    ss = warp_reduce_sum<32>(ss);
    const float norm = sqrtf(ss) * sc_s + sc_b;
    const float fac  = fminf(cmax, fmaxf(cmin, thr / norm)) * tau_val;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int c = lane + k * 32;
        if (c < ne0) {
            dst_row[c] = v[k] * fac;
        }
    }
}

static inline void ggml_cuda_op_inkling_l2_clip_pair(
        ggml_backend_cuda_context & ctx,
        const inkling_clip_desc & a,
        const inkling_clip_desc & b) {
    const int total = a.n_rows + b.n_rows;
    constexpr int warps_per_block = 4;
    const dim3 block_dims(32, warps_per_block, 1);
    const dim3 grid_dims((total + warps_per_block - 1) / warps_per_block, 1, 1);
    inkling_fused_l2_clip_pair_f32_warp<<<grid_dims, block_dims, 0, ctx.stream()>>>(a, b);
}

// 21. Fused 1-Kernel F32 Router GEMV [D x 128] + TopK-MoE (13 ops -> 1 single 64-block kernel on all 48 layers!):
static __global__ void inkling_fused_router_topk128_64blk(
        const float4 * __restrict__ w,
        const float4 * __restrict__ x,
        const float *  __restrict__ bias,
        float *        __restrict__ s_wt,
        float *        __restrict__ s_sel,
        int32_t *      __restrict__ d_cnt,
        int32_t *      __restrict__ ids,
        float *        __restrict__ weights,
        const int D4,
        const int n_exp_used,
        const bool with_norm,
        const bool norm_add_eps,
        const float clamp_val,
        const float scale_val) {
    const int local_e = threadIdx.y >> 1; // 0 or 1
    const int half_w  = threadIdx.y & 1;  // 0 or 1
    const int e       = blockIdx.x * 2 + local_e;
    const int lane    = threadIdx.x;
    const float4 * wr = w + (size_t) e * D4;

    float acc = 0.0f;
    for (int i = half_w * 32 + lane; i < D4; i += 64) {
        const float4 xv = __ldg(&x[i]);
        const float4 wv = __ldg(&wr[i]);
        acc += xv.x * wv.x + xv.y * wv.y + xv.z * wv.z + xv.w * wv.w;
    }
    acc = warp_reduce_sum<32>(acc);

    __shared__ float sm[2][2];
    if (lane == 0) {
        sm[local_e][half_w] = acc;
    }
    __syncthreads();

    if (threadIdx.y < 2 && lane == 0) {
        const float logit = sm[threadIdx.y][0] + sm[threadIdx.y][1];
        float wt = 1.0f / (1.0f + expf(-logit));
        if (__isnanf(wt)) {
            wt = -FLT_MAX;
        }
        const int ex = blockIdx.x * 2 + threadIdx.y;
        float sel = bias ? (wt + bias[ex]) : wt;
        if (__isnanf(sel)) {
            sel = -FLT_MAX;
        }
        s_wt[ex]  = wt;
        s_sel[ex] = sel;
    }
    __syncthreads();

    __shared__ bool is_last;
    if (threadIdx.y == 0 && lane == 0) {
        __threadfence();
        const int done = atomicAdd(d_cnt, 1) + 1;
        is_last = (done == 64);
        if (is_last) {
            *d_cnt = 0;
        }
    }
    __syncthreads();

    if (!is_last || threadIdx.y != 0) {
        return;
    }

    float wt[4], sel[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        wt[i]  = s_wt[i * 32 + lane];
        sel[i] = s_sel[i * 32 + lane];
    }
    float out_w = 0.0f;
    float wsum  = 0.0f;
    for (int k = 0; k < n_exp_used; ++k) {
        float mv = wt[0];
        float ms = sel[0];
        int   me = lane;
#pragma unroll
        for (int i = 1; i < 4; ++i) {
            const int ex = i * 32 + lane;
            if (sel[i] > ms) {
                mv = wt[i];
                ms = sel[i];
                me = ex;
            }
        }
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffff, mv, m);
            const float os = __shfl_xor_sync(0xffffffff, ms, m);
            const int   oe = __shfl_xor_sync(0xffffffff, me, m);
            if (os > ms || (os == ms && oe < me)) {
                mv = ov;
                ms = os;
                me = oe;
            }
        }
        if ((me & 31) == lane) {
            sel[me >> 5] = -INFINITY;
            ids[k] = me;
            if (with_norm) {
                wsum += mv;
            }
        }
        if (k == lane) {
            out_w = mv;
        }
    }
    if (with_norm) {
        wsum = warp_reduce_sum<32>(wsum);
        wsum = norm_add_eps ? (wsum + clamp_val) : fmaxf(wsum, clamp_val);
        out_w *= (1.0f / wsum);
    }
    if (lane < n_exp_used) {
        weights[lane] = out_w * scale_val;
    }
}

static inline int32_t * inkling_get_router_counter(const int device, const int stream_no) {
    static int32_t * d_cnts[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS] = {};
    const int s = (stream_no >= 0 && stream_no < GGML_CUDA_MAX_STREAMS) ? stream_no : 0;
    if (d_cnts[device][s] == nullptr) {
        int32_t * ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&ptr, sizeof(int32_t)));
        CUDA_CHECK(cudaMemset(ptr, 0, sizeof(int32_t)));
        d_cnts[device][s] = ptr;
    }
    return d_cnts[device][s];
}

// Persistent 4-byte GPU slot holding a safe copy of rs_s_copy[0] across the whole graph
// (prevents ggml_gallocr reuse of rs_s_copy after the last skipped GET_ROWS from clobbering Layer 47!).
static inline int32_t * inkling_get_rs_idx_slot(const int device, const int stream_no) {
    static int32_t * d_slots[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS] = {};
    const int s = (stream_no >= 0 && stream_no < GGML_CUDA_MAX_STREAMS) ? stream_no : 0;
    if (d_slots[device][s] == nullptr) {
        int32_t * ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&ptr, sizeof(int32_t)));
        CUDA_CHECK(cudaMemset(ptr, 0, sizeof(int32_t)));
        d_slots[device][s] = ptr;
    }
    return d_slots[device][s];
}

static inline void ggml_cuda_op_inkling_router_topk128(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * w,
        const ggml_tensor * x,
        const ggml_tensor * bias,
        ggml_tensor * ids,
        ggml_tensor * weights,
        const int n_exp_used,
        const bool with_norm,
        const bool norm_add_eps,
        const float clamp_val,
        const float scale_val) {
    const int D4 = (int) (w->ne[0] >> 2);
    ggml_cuda_pool_alloc<float> scratch(ctx.pool(), 256);
    int32_t * d_cnt = inkling_get_router_counter(ctx.device, ctx.curr_stream_no);
    inkling_fused_router_topk128_64blk<<<64, dim3(32, 4, 1), 0, ctx.stream()>>>(
        (const float4 *) w->data,
        (const float4 *) x->data,
        bias ? (const float *) bias->data : nullptr,
        scratch.get(),
        scratch.get() + 128,
        d_cnt,
        (int32_t *) ids->data,
        (float *) weights->data,
        D4, n_exp_used, with_norm, norm_add_eps, clamp_val, scale_val);
}

// ============================================================================
// Wave 7A Mega-Fusions (Full 3-Kernel MoE Pipeline: Idle-SM Router+Q8_1,
// Double-Buffered cp.async.cg 4w/blk gate_up+SiTU -> Direct Pre-Interleaved Q8_1,
// & Zero-Intermediate-F32 down_reduce — 100% Bit-Exact, -96 Launches/tok!)
// ============================================================================

// 22a. Router + TopK128 + Idle-SM Input Pre-Quantization (64 router blocks + ceil(D/256) quant blocks = 73 blocks on 82 SMs!)
template <bool PREINT>
static __global__ void inkling_fused_router_topk128_q8_73blk(
        const float4 * __restrict__ w,
        const float4 * __restrict__ x,
        const float *  __restrict__ bias,
        float *        __restrict__ s_wt,
        float *        __restrict__ s_sel,
        int32_t *      __restrict__ d_cnt,
        int32_t *      __restrict__ ids,
        float *        __restrict__ weights,
        block_q8_1 *   __restrict__ y_q8,
        const int D4,
        const int n_exp_used,
        const bool with_norm,
        const bool norm_add_eps,
        const float clamp_val,
        const float scale_val) {
    if (blockIdx.x >= 64) {
        const int D = D4 * 4;
        const int qblk = blockIdx.x - 64;
        const int tid  = threadIdx.y * 32 + threadIdx.x; // 0..127
        const float * xf = (const float *) x;
#pragma unroll
        for (int iter = 0; iter < 2; ++iter) {
            const int i0 = qblk * 256 + iter * 128 + tid;
            if (i0 < D) {
                const float xi   = xf[i0];
                const float amax = warp_reduce_max<32>(fabsf(xi));
                const float d    = amax / 127.0f;
                const int8_t q   = (amax == 0.0f) ? 0 : (int8_t) roundf(xi / d);
                const int ib     = i0 >> 5;
                const int iqs    = i0 & 31;
                const int dst_iqs = PREINT ? ((iqs & 16) | ((iqs & 7) << 1) | ((iqs >> 3) & 1)) : iqs;
                y_q8[ib].qs[dst_iqs] = q;
                if (iqs == 0) {
                    y_q8[ib].ds = make_half2(__float2half(d), __float2half(0.0f));
                }
            }
        }
        return;
    }

    const int local_e = threadIdx.y >> 1;
    const int half_w  = threadIdx.y & 1;
    const int e       = blockIdx.x * 2 + local_e;
    const int lane    = threadIdx.x;
    const float4 * wr = w + (size_t) e * D4;

    float acc = 0.0f;
    for (int i = half_w * 32 + lane; i < D4; i += 64) {
        const float4 xv = __ldg(&x[i]);
        const float4 wv = __ldg(&wr[i]);
        acc += xv.x * wv.x + xv.y * wv.y + xv.z * wv.z + xv.w * wv.w;
    }
    acc = warp_reduce_sum<32>(acc);

    __shared__ float sm[2][2];
    if (lane == 0) {
        sm[local_e][half_w] = acc;
    }
    __syncthreads();

    if (threadIdx.y < 2 && lane == 0) {
        const float logit = sm[threadIdx.y][0] + sm[threadIdx.y][1];
        float wt = 1.0f / (1.0f + expf(-logit));
        if (__isnanf(wt)) {
            wt = -FLT_MAX;
        }
        const int ex = blockIdx.x * 2 + threadIdx.y;
        float sel = bias ? (wt + bias[ex]) : wt;
        if (__isnanf(sel)) {
            sel = -FLT_MAX;
        }
        s_wt[ex]  = wt;
        s_sel[ex] = sel;
    }
    __syncthreads();

    __shared__ bool is_last;
    if (threadIdx.y == 0 && lane == 0) {
        __threadfence();
        const int done = atomicAdd(d_cnt, 1) + 1;
        is_last = (done == 64);
        if (is_last) {
            *d_cnt = 0;
        }
    }
    __syncthreads();

    if (!is_last || threadIdx.y != 0) {
        return;
    }

    float wt[4], sel[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        wt[i]  = s_wt[i * 32 + lane];
        sel[i] = s_sel[i * 32 + lane];
    }
    float out_w = 0.0f;
    float wsum  = 0.0f;
    for (int k = 0; k < n_exp_used; ++k) {
        float mv = wt[0];
        float ms = sel[0];
        int   me = lane;
#pragma unroll
        for (int i = 1; i < 4; ++i) {
            const int ex = i * 32 + lane;
            if (sel[i] > ms) {
                mv = wt[i];
                ms = sel[i];
                me = ex;
            }
        }
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffff, mv, m);
            const float os = __shfl_xor_sync(0xffffffff, ms, m);
            const int   oe = __shfl_xor_sync(0xffffffff, me, m);
            if (os > ms || (os == ms && oe < me)) {
                mv = ov;
                ms = os;
                me = oe;
            }
        }
        if ((me & 31) == lane) {
            sel[me >> 5] = -INFINITY;
            ids[k] = me;
            if (with_norm) {
                wsum += mv;
            }
        }
        if (k == lane) {
            out_w = mv;
        }
    }
    if (with_norm) {
        wsum = warp_reduce_sum<32>(wsum);
        wsum = norm_add_eps ? (wsum + clamp_val) : fmaxf(wsum, clamp_val);
        out_w *= (1.0f / wsum);
    }
    if (lane < n_exp_used) {
        weights[lane] = out_w * scale_val;
    }
}

// 22b. Double-Buffered cp.async.cg 4-warp NVFP4 gate_up+SiTU -> Direct Pre-Interleaved Q8_1 (32 rows/blk, 0 F32 VRAM write!)
static __global__ void inkling_fused_nvfp4_gate_up_situ_db4w_q8_1_nb36(
        const block_nvfp4 * __restrict__ vx,
        const block_q8_1 * __restrict__ vy,
        const int32_t * __restrict__ ids,
        const float * __restrict__ sc,
        block_q8_1 * __restrict__ dst_q8,
        const int n_ff,
        const size_t stride_ch_x,
        const float inv_beta, const float beta,
        const float inv_lb,   const float lb,
        const bool has_lb) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __shared__ __align__(16) uint32_t s_row[4][2][324]; // 10,368 bytes
    __shared__ float s_val[32];                         //    128 bytes -> 10,496 bytes total!
    const int warp = threadIdx.y; // 0..3
    const int tid  = threadIdx.x; // 0..31
    const int row0 = blockIdx.x * 32 + warp * 8;
    const int e    = blockIdx.y;
    const int ch   = ids[e];
    const float s  = sc ? sc[ch] : 1.0f;
    const block_nvfp4 * base = vx + (size_t) ch * stride_ch_x;

    auto issue_row = [&](int buf, int r_abs) {
        const uint4 * src4 = (const uint4 *) (base + (size_t) r_abs * 36);
        uint4 * sm4 = (uint4 *) s_row[warp][buf];
        inkling_cp_async_cg_16(&sm4[tid],      &src4[tid]);
        inkling_cp_async_cg_16(&sm4[tid + 32], &src4[tid + 32]);
        if (tid < 17) {
            inkling_cp_async_cg_16(&sm4[tid + 64], &src4[tid + 64]);
        }
        inkling_cp_async_commit();
    };

    auto dot_buf = [&](int buf) -> float {
        const block_nvfp4 * sb = (const block_nvfp4 *) s_row[warp][buf];
        const int iqs = (tid & 1) * 4;
        const int is0 = iqs >> 1;
        float sum = 0.0f;
#pragma unroll
        for (int k = tid >> 1; k < 36; k += 16) {
            const block_nvfp4 * bq4 = &sb[k];
            const block_q8_1  * bq8 = &vy[k * 2 + (is0 >> 1)];
            const int * q4p = (const int *) bq4->qs + iqs;
            const uint16_t dp = *reinterpret_cast<const uint16_t *>(bq4->d + is0);
            const int * u = (const int *) bq8->qs;
            const int2 v0 = inkling_unpack_nvfp4_raw(q4p[0]);
            const int2 v1 = inkling_unpack_nvfp4_raw(q4p[1]);
            int s0 = ggml_cuda_dp4a(v0.x, __ldg(u + 0), 0);
            s0     = ggml_cuda_dp4a(v0.y, __ldg(u + 1), s0);
            s0     = ggml_cuda_dp4a(v1.x, __ldg(u + 2), s0);
            s0     = ggml_cuda_dp4a(v1.y, __ldg(u + 3), s0);
            const int2 v2 = inkling_unpack_nvfp4_raw(q4p[2]);
            const int2 v3 = inkling_unpack_nvfp4_raw(q4p[3]);
            int s1 = ggml_cuda_dp4a(v2.x, __ldg(u + 4), 0);
            s1     = ggml_cuda_dp4a(v2.y, __ldg(u + 5), s1);
            s1     = ggml_cuda_dp4a(v3.x, __ldg(u + 6), s1);
            s1     = ggml_cuda_dp4a(v3.y, __ldg(u + 7), s1);
            sum += __low2float(bq8->ds) * (ggml_cuda_ue4m3_to_fp32(dp & 0xFFu) * (float) s0 + ggml_cuda_ue4m3_to_fp32(dp >> 8) * (float) s1);
        }
        return warp_reduce_sum<32>(sum);
    };

    issue_row(0, row0);
#pragma unroll 1
    for (int rl = 0; rl < 8; ++rl) {
        const int r = row0 + rl;
        issue_row(1, r + n_ff);
        inkling_cp_async_wait<1>();
        __syncwarp();
        const float sum_g = dot_buf(0);
        __syncwarp();
        if (rl + 1 < 8) {
            issue_row(0, r + 1);
            inkling_cp_async_wait<1>();
        } else {
            inkling_cp_async_wait<0>();
        }
        __syncwarp();
        const float sum_u = dot_buf(1);
        __syncwarp();
        if (tid == 0) {
            s_val[warp * 8 + rl] = inkling_situ_elem(sum_g * s, sum_u * s, inv_beta, beta, inv_lb, lb, has_lb);
        }
    }
    __syncthreads();
    if (warp == 0) {
        const float xi   = s_val[tid];
        const float amax = warp_reduce_max<32>(fabsf(xi));
        const float d    = amax / 127.0f;
        const int8_t q   = (amax == 0.0f) ? 0 : (int8_t) roundf(xi / d);
        const int ib     = e * (n_ff >> 5) + blockIdx.x;
        const int dst_iqs = (tid & 16) | ((tid & 7) << 1) | ((tid >> 3) & 1);
        dst_q8[ib].qs[dst_iqs] = q;
        if (tid == 0) {
            dst_q8[ib].ds = make_half2(__float2half(d), __float2half(0.0f));
        }
    }
#endif
}

static __global__ void inkling_fused_nvfp4_gate_up_situ_gen_q8_1(
        const block_nvfp4 * __restrict__ vx,
        const block_q8_1 * __restrict__ vy,
        const int32_t * __restrict__ ids,
        const float * __restrict__ sc,
        block_q8_1 * __restrict__ dst_q8,
        const int n_ff,
        const int nb,
        const size_t stride_ch_x,
        const float inv_beta, const float beta,
        const float inv_lb,   const float lb,
        const bool has_lb) {
    __shared__ float s_val[32];
    const int warp = threadIdx.y; // 0..3
    const int tid  = threadIdx.x; // 0..31
    const int row0 = blockIdx.x * 32 + warp * 8;
    const int e    = blockIdx.y;
    const int ch   = ids[e];
    const float s  = sc ? sc[ch] : 1.0f;
    const int iqs  = (tid & 1) * 4;
    for (int rl = 0; rl < 8; ++rl) {
        const int r = row0 + rl;
        const block_nvfp4 * xg = vx + (size_t) ch * stride_ch_x + (size_t) r * nb;
        const block_nvfp4 * xu = xg + (size_t) n_ff * nb;
        float sum_g = 0.0f, sum_u = 0.0f;
        for (int k = tid >> 1; k < nb; k += 16) {
            sum_g += inkling_vec_dot_nvfp4_preint(&xg[k], &vy[k * 2], iqs);
            sum_u += inkling_vec_dot_nvfp4_preint(&xu[k], &vy[k * 2], iqs);
        }
        sum_g = warp_reduce_sum<32>(sum_g);
        sum_u = warp_reduce_sum<32>(sum_u);
        if (tid == 0) {
            s_val[warp * 8 + rl] = inkling_situ_elem(sum_g * s, sum_u * s, inv_beta, beta, inv_lb, lb, has_lb);
        }
    }
    __syncthreads();
    if (warp == 0) {
        const float xi   = s_val[tid];
        const float amax = warp_reduce_max<32>(fabsf(xi));
        const float d    = amax / 127.0f;
        const int8_t q   = (amax == 0.0f) ? 0 : (int8_t) roundf(xi / d);
        const int ib     = e * (n_ff >> 5) + blockIdx.x;
        const int dst_iqs = (tid & 16) | ((tid & 7) << 1) | ((tid >> 3) & 1);
        dst_q8[ib].qs[dst_iqs] = q;
        if (tid == 0) {
            dst_q8[ib].ds = make_half2(__float2half(d), __float2half(0.0f));
        }
    }
}

// 22c. Q8_0 gate_up+SiTU -> Direct Standard Q8_1 (8 warps x 4 rows = 32 rows/blk, 0 F32 VRAM write, 4 Q8_0 MoE layers!)
static __global__ void inkling_fused_q8_0_gate_up_situ_q8_1(
        const block_q8_0 * __restrict__ vx,
        const block_q8_1 * __restrict__ vy,
        const int32_t * __restrict__ ids,
        block_q8_1 * __restrict__ dst_q8,
        const int n_ff,
        const int nb,
        const size_t stride_ch_x,
        const float inv_beta, const float beta,
        const float inv_lb,   const float lb,
        const bool has_lb) {
    __shared__ float s_val[32];
    const int warp = threadIdx.y; // 0..7
    const int tid  = threadIdx.x; // 0..31
    const int row0 = blockIdx.x * 32 + warp * 4;
    const int e    = blockIdx.y;
    const int ch   = ids[e];
    const int iqs  = (tid & 3) * 2;
    for (int rl = 0; rl < 4; ++rl) {
        const int r = row0 + rl;
        const block_q8_0 * xg = vx + (size_t) ch * stride_ch_x + (size_t) r * nb;
        const block_q8_0 * xu = xg + (size_t) n_ff * nb;
        float sum_g = 0.0f, sum_u = 0.0f;
        for (int k = tid >> 2; k < nb; k += 8) {
            const block_q8_1 * by = &vy[k];
            const int * uy = (const int *) by->qs;
            const int u0 = __ldg(uy + iqs + 0);
            const int u1 = __ldg(uy + iqs + 1);
            const float dy = __low2float(by->ds);
            {
                const block_q8_0 * bx = &xg[k];
                int s = ggml_cuda_dp4a(inkling_get_int_b2(bx->qs, iqs + 0), u0, 0);
                s     = ggml_cuda_dp4a(inkling_get_int_b2(bx->qs, iqs + 1), u1, s);
                sum_g += (float) s * __half2float(__ldg(&bx->d)) * dy;
            }
            {
                const block_q8_0 * bx = &xu[k];
                int s = ggml_cuda_dp4a(inkling_get_int_b2(bx->qs, iqs + 0), u0, 0);
                s     = ggml_cuda_dp4a(inkling_get_int_b2(bx->qs, iqs + 1), u1, s);
                sum_u += (float) s * __half2float(__ldg(&bx->d)) * dy;
            }
        }
        sum_g = warp_reduce_sum<32>(sum_g);
        sum_u = warp_reduce_sum<32>(sum_u);
        if (tid == 0) {
            s_val[warp * 4 + rl] = inkling_situ_elem(sum_g, sum_u, inv_beta, beta, inv_lb, lb, has_lb);
        }
    }
    __syncthreads();
    if (warp == 0) {
        const float xi   = s_val[tid];
        const float amax = warp_reduce_max<32>(fabsf(xi));
        const float d    = amax / 127.0f;
        const int8_t q   = (amax == 0.0f) ? 0 : (int8_t) roundf(xi / d);
        const int ib     = e * (n_ff >> 5) + blockIdx.x;
        dst_q8[ib].qs[tid] = q;
        if (tid == 0) {
            dst_q8[ib].ds = make_half2(__float2half(d), __float2half(0.0f));
        }
    }
}

static inline void ggml_cuda_op_inkling_moe_nvfp4_full(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * w_router,
        const ggml_tensor * x,
        const ggml_tensor * bias,
        ggml_tensor * ids,
        ggml_tensor * weights,
        const int n_exp_used,
        const bool with_norm,
        const bool norm_add_eps,
        const float clamp_val,
        const float scale_val,
        const ggml_tensor * w_gu,
        const ggml_tensor * sc_gu,
        const float inv_beta, const float beta,
        const float inv_lb,   const float lb,
        const bool has_lb,
        const ggml_tensor * w_down,
        const ggml_tensor * sc_down,
        ggml_tensor * dst) {
    const int K      = (int) w_gu->ne[0];
    const int n_ff   = (int) (w_gu->ne[1] / 2);
    const int nrows  = (int) w_down->ne[1];
    const int nb_gu  = K / QK_NVFP4;
    const int nb_dn  = n_ff / QK_NVFP4;
    const size_t stride_gu = w_gu->nb[2] / sizeof(block_nvfp4);
    const size_t stride_dn = w_down->nb[2] / sizeof(block_nvfp4);
    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<float>      r_scratch(ctx.pool(), 256);
    ggml_cuda_pool_alloc<block_q8_1> y_q8_in(ctx.pool(), (size_t) (K / QK8_1));
    ggml_cuda_pool_alloc<block_q8_1> y_q8_mid(ctx.pool(), (size_t) n_exp_used * (n_ff / QK8_1));
    int32_t * d_cnt = inkling_get_router_counter(ctx.device, ctx.curr_stream_no);

    // Launch 1: 64 router blocks + ceil(K/256) idle-SM input pre-quantize blocks (73 blocks on 82 SMs!)
    const int n_qblks = (K + 255) / 256;
    inkling_fused_router_topk128_q8_73blk<true><<<64 + n_qblks, dim3(32, 4, 1), 0, stream>>>(
        (const float4 *) w_router->data, (const float4 *) x->data,
        bias ? (const float *) bias->data : nullptr,
        r_scratch.get(), r_scratch.get() + 128, d_cnt,
        (int32_t *) ids->data, (float *) weights->data, y_q8_in.get(),
        K >> 2, n_exp_used, with_norm, norm_add_eps, clamp_val, scale_val);

    // Launch 2: NVFP4 gate_up+SiTU (default: 1152-block cpasync<8> avoids 288-block tail-wave imbalance on 82 SMs;
    // W7_USE_DB4W=1 selects direct-Q8_1 4-warp double-buffered kernel)
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const bool use_cpasync = (cc >= GGML_CUDA_CC_AMPERE) && (nb_gu == 36) &&
        (((uintptr_t) w_gu->data & 15) == 0) &&
        (((stride_gu * sizeof(block_nvfp4)) & 15) == 0);
    static const bool w7_use_db4w = (getenv("W7_USE_DB4W") != nullptr);
    if (use_cpasync && !w7_use_db4w) {
        ggml_cuda_pool_alloc<float> mid_f32(ctx.pool(), (size_t) n_exp_used * n_ff);
        constexpr int NWARPS = 8;
        inkling_fused_nvfp4_gate_up_situ_cpasync_nb36<NWARPS><<<dim3((n_ff + NWARPS - 1) / NWARPS, n_exp_used, 1), dim3(32, NWARPS, 1), 0, stream>>>(
            (const block_nvfp4 *) w_gu->data, y_q8_in.get(), (const int32_t *) ids->data,
            sc_gu ? (const float *) sc_gu->data : nullptr, mid_f32.get(),
            n_ff, stride_gu, inv_beta, beta, inv_lb, lb, has_lb);
        inkling_quantize_q8_1_nvfp4_preint_f32<<<dim3((n_ff + 255) / 256, n_exp_used, 1), 256, 0, stream>>>(
            mid_f32.get(), y_q8_mid.get(), n_ff, n_exp_used);
    } else if (use_cpasync) {
        inkling_fused_nvfp4_gate_up_situ_db4w_q8_1_nb36<<<dim3(n_ff >> 5, n_exp_used, 1), dim3(32, 4, 1), 0, stream>>>(
            (const block_nvfp4 *) w_gu->data, y_q8_in.get(), (const int32_t *) ids->data,
            sc_gu ? (const float *) sc_gu->data : nullptr, y_q8_mid.get(),
            n_ff, stride_gu, inv_beta, beta, inv_lb, lb, has_lb);
    } else {
        inkling_fused_nvfp4_gate_up_situ_gen_q8_1<<<dim3(n_ff >> 5, n_exp_used, 1), dim3(32, 4, 1), 0, stream>>>(
            (const block_nvfp4 *) w_gu->data, y_q8_in.get(), (const int32_t *) ids->data,
            sc_gu ? (const float *) sc_gu->data : nullptr, y_q8_mid.get(),
            n_ff, nb_gu, stride_gu, inv_beta, beta, inv_lb, lb, has_lb);
    }

    // Launch 3: NVFP4 down_reduce<4> (576 blocks = 7.02 blocks/SM, zero tail-wave imbalance!)
    constexpr int NWARPS = 4;
    if (n_exp_used == 8 && nb_dn == 18) {
        inkling_fused_nvfp4_down_reduce_8x18<NWARPS><<<dim3((nrows + NWARPS - 1) / NWARPS, 1, 1), dim3(32, NWARPS, 1), 0, stream>>>(
            (const block_nvfp4 *) w_down->data, y_q8_mid.get(), (const int32_t *) ids->data,
            sc_down ? (const float *) sc_down->data : nullptr, (const float *) weights->data,
            (float *) dst->data, nrows, stride_dn);
    } else {
        inkling_fused_nvfp4_down_reduce_gen<NWARPS><<<dim3((nrows + NWARPS - 1) / NWARPS, 1, 1), dim3(32, NWARPS, 1), 0, stream>>>(
            (const block_nvfp4 *) w_down->data, y_q8_mid.get(), (const int32_t *) ids->data,
            sc_down ? (const float *) sc_down->data : nullptr, (const float *) weights->data,
            (float *) dst->data, nrows, nb_dn, n_exp_used, stride_dn);
    }
}

static inline void ggml_cuda_op_inkling_moe_q8_0_full(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * w_router,
        const ggml_tensor * x,
        const ggml_tensor * bias,
        ggml_tensor * ids,
        ggml_tensor * weights,
        const int n_exp_used,
        const bool with_norm,
        const bool norm_add_eps,
        const float clamp_val,
        const float scale_val,
        const ggml_tensor * w_gu,
        const float inv_beta, const float beta,
        const float inv_lb,   const float lb,
        const bool has_lb,
        const ggml_tensor * w_down,
        ggml_tensor * dst) {
    const int K      = (int) w_gu->ne[0];
    const int n_ff   = (int) (w_gu->ne[1] / 2);
    const int nrows  = (int) w_down->ne[1];
    const int nb_gu  = K / QK8_0;
    const int nb_dn  = n_ff / QK8_0;
    const size_t stride_gu = w_gu->nb[2] / sizeof(block_q8_0);
    const size_t stride_dn = w_down->nb[2] / sizeof(block_q8_0);
    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<float>      r_scratch(ctx.pool(), 256);
    ggml_cuda_pool_alloc<block_q8_1> y_q8_in(ctx.pool(), (size_t) nb_gu);
    ggml_cuda_pool_alloc<block_q8_1> y_q8_mid(ctx.pool(), (size_t) n_exp_used * nb_dn);
    int32_t * d_cnt = inkling_get_router_counter(ctx.device, ctx.curr_stream_no);

    const int n_qblks = (K + 255) / 256;
    inkling_fused_router_topk128_q8_73blk<false><<<64 + n_qblks, dim3(32, 4, 1), 0, stream>>>(
        (const float4 *) w_router->data, (const float4 *) x->data,
        bias ? (const float *) bias->data : nullptr,
        r_scratch.get(), r_scratch.get() + 128, d_cnt,
        (int32_t *) ids->data, (float *) weights->data, y_q8_in.get(),
        K >> 2, n_exp_used, with_norm, norm_add_eps, clamp_val, scale_val);

    inkling_fused_q8_0_gate_up_situ_q8_1<<<dim3(n_ff >> 5, n_exp_used, 1), dim3(32, 8, 1), 0, stream>>>(
        (const block_q8_0 *) w_gu->data, y_q8_in.get(), (const int32_t *) ids->data,
        y_q8_mid.get(), n_ff, nb_gu, stride_gu, inv_beta, beta, inv_lb, lb, has_lb);

    constexpr int NWARPS = 4;
    inkling_fused_q8_0_down_reduce_f32<NWARPS><<<dim3((nrows + NWARPS - 1) / NWARPS, 1, 1), dim3(32, NWARPS, 1), 0, stream>>>(
        (const block_q8_0 *) w_down->data, y_q8_mid.get(), (const int32_t *) ids->data,
        (const float *) weights->data, (float *) dst->data, nrows, nb_dn, n_exp_used, stride_dn);
}

// ============================================================================
// Wave 7B Breakthrough: 3-Warp Double-Buffered Ampere cp.async.cg GQA=3
//   Fused Rel-Pos Flash Attention + Zero-Sink + Sigmoid + Q8_1 -> Aligned-32 wo
//   Replaces rel_pos_mask (13 ops) + FLASH_ATTN_EXT (3 launches) + sigmoid*mul + quantize
//   with 1 single launch (-192 launches/tok, 0 F16 mask VRAM roundtrip, 3x GQA reuse!)
// ============================================================================

static __device__ __forceinline__ int2 inkling_load_q8_8b_align32(
        const block_q8_0 * bx,
        const int iqs,
        const bool k_odd) {
    if (k_odd) {
        const int * p = (const int *) (bx->qs + iqs * 4);
        return make_int2(__ldg(p + 0), __ldg(p + 1));
    } else {
        const uint32_t * p = (const uint32_t *) ((const char *) (bx->qs + iqs * 4) - 2);
        const uint32_t a = __ldg(p + 0);
        const uint32_t b = __ldg(p + 1);
        const uint32_t c = __ldg(p + 2);
        return make_int2((int) __funnelshift_r(a, b, 16), (int) __funnelshift_r(b, c, 16));
    }
}

template <int NWARPS, int NB_FIXED>
static __global__ void inkling_q8_0_mmv_align32_f32(
        const block_q8_0 * __restrict__ w,
        const block_q8_1 * __restrict__ yq,
        float * __restrict__ dst,
        const int nrows,
        const int nb_arg) {
    const int r = blockIdx.x * NWARPS + threadIdx.y;
    if (r >= nrows) {
        return;
    }
    const int tid = threadIdx.x;
    const int nb  = (NB_FIXED > 0) ? NB_FIXED : nb_arg;
    const block_q8_0 * xrow = w + (size_t) r * nb;
    const int iqs   = (tid & 3) * 2;
    const bool k_odd = ((tid >> 2) & 1) != 0;
    float sum = 0.0f;
#pragma unroll
    for (int k = tid >> 2; k < nb; k += 8) {
        const block_q8_0 * bx = &xrow[k];
        const block_q8_1 * by = &yq[k];
        const int * uy = (const int *) by->qs;
        const int2 xv  = inkling_load_q8_8b_align32(bx, iqs, k_odd);
        int s = ggml_cuda_dp4a(xv.x, __ldg(uy + iqs + 0), 0);
        s     = ggml_cuda_dp4a(xv.y, __ldg(uy + iqs + 1), s);
        sum += (float) s * __half2float(__ldg(&bx->d)) * __low2float(__ldg(&by->ds));
    }
    sum = warp_reduce_sum<32>(sum);
    if (tid == 0) {
        dst[r] = sum;
    }
}

static __device__ __forceinline__ float inkling_warp_sum8(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 4, 8);
    v += __shfl_xor_sync(0xffffffff, v, 2, 8);
    v += __shfl_xor_sync(0xffffffff, v, 1, 8);
    return v;
}

template <bool MASK_F16, int MAX_SPLITS>
__launch_bounds__(96, 8)
static __global__ void inkling_rel_pos_fattn_gqa3_cpasync_q8(
        const float *   __restrict__ Q,
        const half  *   __restrict__ K,
        const half  *   __restrict__ V,
        const float *   __restrict__ R,
        const float *   __restrict__ rel_bias,
        const float *   __restrict__ sinks,
        const int32_t * __restrict__ rel_pos,
        const void  *   __restrict__ raw_mask,
        const float *   __restrict__ Gcur,
        block_q8_1  *   __restrict__ dst_q8,
        float       *   __restrict__ scratch, // [8][MAX_SPLITS][3][132]
        int32_t     *   __restrict__ cnt,     // [8]
        const int n_kv,
        const int n_extent,
        const size_t q_head_stride_f,
        const size_t k_head_stride_h,
        const size_t k_tok_stride_h,
        const size_t v_head_stride_h,
        const size_t v_tok_stride_h,
        const float kq_scale,
        const int n_splits) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int TILE = 16;
    const int sp    = blockIdx.x;
    const int kvh   = blockIdx.y; // 0..7
    const int lane  = threadIdx.x; // 0..31
    const int warp  = threadIdx.y; // 0..2 -> Q head ql = warp
    const int tid   = warp * 32 + lane;
    const int klane = lane & 7;
    const int kgrp  = lane >> 3;
    const int qh    = kvh * 3 + warp;

    const float4 * qp = (const float4 *) (Q + (size_t) qh * q_head_stride_f) + klane * 4;
    float4 q0 = __ldg(qp + 0); q0.x *= kq_scale; q0.y *= kq_scale; q0.z *= kq_scale; q0.w *= kq_scale;
    float4 q1 = __ldg(qp + 1); q1.x *= kq_scale; q1.y *= kq_scale; q1.z *= kq_scale; q1.w *= kq_scale;
    float4 q2 = __ldg(qp + 2); q2.x *= kq_scale; q2.y *= kq_scale; q2.z *= kq_scale; q2.w *= kq_scale;
    float4 q3 = __ldg(qp + 3); q3.x *= kq_scale; q3.y *= kq_scale; q3.z *= kq_scale; q3.w *= kq_scale;
    const float2 r_v = __ldg((const float2 *) (R + (size_t) qh * 16) + klane);
    const float sink = __ldg(&sinks[qh]);

    float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float m_max = -1e30f;
    float l_sum = 0.0f;

    __shared__ __align__(16) uint4  s_KV[2][TILE][16]; // 8192 B
    __shared__ __align__(16) float2 s_RB[TILE][8];     // 1024 B
    __shared__ float s_M[2][TILE];                     //  128 B
    __shared__ float s_W[3][TILE];                     //  192 B

    const int n_tiles     = (n_kv + TILE - 1) / TILE;
    const int chunk_tiles = (n_tiles + n_splits - 1) / n_splits;
    const int t_beg       = sp * chunk_tiles;
    const int t_end       = min(n_tiles, t_beg + chunk_tiles);

    const half * K_base = K + (size_t) kvh * k_head_stride_h;
    const half * V_base = V + (size_t) kvh * v_head_stride_h;

    auto issue_kv = [&](int stage, int tile_idx, const half * base, size_t tok_stride, uint32_t act) {
        const int j0  = tile_idx * TILE;
        const int col = tid & 15;
        const int r0  = tid >> 4;
        if ((act >> r0) & 1u) {
            inkling_cp_async_cg_16(&s_KV[stage][r0][col], (const uint4 *) (base + (size_t) (j0 + r0) * tok_stride) + col);
        }
        const int r1 = r0 + 6;
        if ((act >> r1) & 1u) {
            inkling_cp_async_cg_16(&s_KV[stage][r1][col], (const uint4 *) (base + (size_t) (j0 + r1) * tok_stride) + col);
        }
        const int r2 = r0 + 12;
        if (r2 < TILE && ((act >> r2) & 1u)) {
            inkling_cp_async_cg_16(&s_KV[stage][r2][col], (const uint4 *) (base + (size_t) (j0 + r2) * tok_stride) + col);
        }
        inkling_cp_async_commit();
    };

    auto probe_tile = [&](int tile_idx, int mslot) -> uint32_t {
        const int j0 = tile_idx * TILE;
        float mv = -INFINITY;
        if (lane < TILE && j0 + lane < n_kv) {
            if constexpr (MASK_F16) {
                mv = __half2float(__ldg(&((const half *) raw_mask)[j0 + lane]));
            } else {
                mv = __ldg(&((const float *) raw_mask)[j0 + lane]);
            }
        }
        const uint32_t act = __ballot_sync(0xffffffff, mv > -1e20f) & 0xFFFFu;
        if (warp == 0 && lane < TILE) {
            s_M[mslot][lane] = mv;
        }
        return act;
    };

    int cur_t = t_beg;
    uint32_t cur_act = 0u;
    int mslot = 0;
    while (cur_t < t_end) {
        cur_act = probe_tile(cur_t, mslot);
        if (cur_act != 0u) {
            break;
        }
        cur_t++;
    }

    if (cur_t < t_end) {
        issue_kv(0, cur_t, K_base, k_tok_stride_h, cur_act);
        const int j0 = cur_t * TILE;
        for (int idx = tid; idx < TILE * 8; idx += 96) {
            const int r = idx >> 3;
            const int c = idx & 7;
            float2 rb = make_float2(0.0f, 0.0f);
            if ((cur_act >> r) & 1u) {
                const int d = __ldg(&rel_pos[j0 + r]);
                if (d >= 0 && d < n_extent) {
                    rb = __ldg((const float2 *) (rel_bias + (size_t) d * 16) + c);
                }
            }
            s_RB[r][c] = rb;
        }
    }

    while (cur_t < t_end) {
        inkling_cp_async_wait<0>();
        __syncthreads();
        issue_kv(1, cur_t, V_base, v_tok_stride_h, cur_act);

        float sc_reg = -1e30f;
        float nmax   = m_max;
        const int gbase = kgrp * 4;
#pragma unroll
        for (int step = 0; step < 4; ++step) {
            const int bit = gbase + step;
            const uint4 k0 = s_KV[0][bit][klane * 2 + 0];
            const uint4 k1 = s_KV[0][bit][klane * 2 + 1];
            const float2 rb = s_RB[bit][klane];
            const float2 kf0 = __half22float2(*reinterpret_cast<const half2 *>(&k0.x));
            const float2 kf1 = __half22float2(*reinterpret_cast<const half2 *>(&k0.y));
            const float2 kf2 = __half22float2(*reinterpret_cast<const half2 *>(&k0.z));
            const float2 kf3 = __half22float2(*reinterpret_cast<const half2 *>(&k0.w));
            const float2 kf4 = __half22float2(*reinterpret_cast<const half2 *>(&k1.x));
            const float2 kf5 = __half22float2(*reinterpret_cast<const half2 *>(&k1.y));
            const float2 kf6 = __half22float2(*reinterpret_cast<const half2 *>(&k1.z));
            const float2 kf7 = __half22float2(*reinterpret_cast<const half2 *>(&k1.w));
            float a = q0.x*kf0.x + q0.y*kf0.y + q0.z*kf1.x + q0.w*kf1.y
                    + q1.x*kf2.x + q1.y*kf2.y + q1.z*kf3.x + q1.w*kf3.y
                    + q2.x*kf4.x + q2.y*kf4.y + q2.z*kf5.x + q2.w*kf5.y
                    + q3.x*kf6.x + q3.y*kf6.y + q3.z*kf7.x + q3.w*kf7.y
                    + r_v.x*rb.x + r_v.y*rb.y;
            const float s_val = inkling_warp_sum8(a);
            const float s     = ((cur_act >> bit) & 1u) ? (s_val - sink + s_M[mslot][bit]) : -1e30f;
            nmax = fmaxf(nmax, s);
            if (klane == step) {
                sc_reg = s;
            }
        }
        if (klane < 4) {
            s_W[warp][gbase + klane] = sc_reg;
        }
        __syncwarp();

        nmax = fmaxf(nmax, __shfl_xor_sync(0xffffffff, nmax, 8));
        nmax = fmaxf(nmax, __shfl_xor_sync(0xffffffff, nmax, 16));
        const float rsc = expf(m_max - nmax);
        m_max = nmax;

        float w_lane = 0.0f;
        if (lane < TILE) {
            w_lane = ((cur_act >> lane) & 1u) ? expf(s_W[warp][lane] - m_max) : 0.0f;
            s_W[warp][lane] = w_lane;
        }
        l_sum = l_sum * rsc + warp_reduce_sum<32>(w_lane);
        o.x *= rsc; o.y *= rsc; o.z *= rsc; o.w *= rsc;

        int nxt_t = cur_t + 1;
        uint32_t nxt_act = 0u;
        const int nslot = mslot ^ 1;
        while (nxt_t < t_end) {
            nxt_act = probe_tile(nxt_t, nslot);
            if (nxt_act != 0u) {
                break;
            }
            nxt_t++;
        }

        inkling_cp_async_wait<0>();
        __syncthreads();

        if (nxt_t < t_end) {
            issue_kv(0, nxt_t, K_base, k_tok_stride_h, nxt_act);
        }

        const uint2 * sv2 = (const uint2 *) &s_KV[1][0][0];
#pragma unroll
        for (int bit = 0; bit < TILE; ++bit) {
            const float kw = s_W[warp][bit];
            if (kw != 0.0f) {
                const uint2 vv = sv2[bit * 32 + lane];
                const float2 vf0 = __half22float2(*reinterpret_cast<const half2 *>(&vv.x));
                const float2 vf1 = __half22float2(*reinterpret_cast<const half2 *>(&vv.y));
                o.x += kw * vf0.x; o.y += kw * vf0.y; o.z += kw * vf1.x; o.w += kw * vf1.y;
            }
        }

        if (nxt_t < t_end) {
            const int j0 = nxt_t * TILE;
            for (int idx = tid; idx < TILE * 8; idx += 96) {
                const int r = idx >> 3;
                const int c = idx & 7;
                float2 rb = make_float2(0.0f, 0.0f);
                if ((nxt_act >> r) & 1u) {
                    const int d = __ldg(&rel_pos[j0 + r]);
                    if (d >= 0 && d < n_extent) {
                        rb = __ldg((const float2 *) (rel_bias + (size_t) d * 16) + c);
                    }
                }
                s_RB[r][c] = rb;
            }
        }
        cur_t   = nxt_t;
        cur_act = nxt_act;
        mslot   = nslot;
    }

    float * s_out = (float *) &s_KV[0][0][0];
    __syncthreads();

    auto finalize_head = [&](float fm, float fl, float * fo) {
        const float nm     = fmaxf(fm, 0.0f);
        const float e_old  = expf(fm - nm);
        const float e_sink = expf(0.0f - nm);
        const float inv_l  = e_old / (fl * e_old + e_sink);
#pragma unroll
        for (int blk = 0; blk < 4; ++blk) {
            const int c      = blk * 32 + lane;
            const float val  = fo[c] * inv_l;
            const float g    = Gcur[(size_t) qh * 128 + c];
            const float xi   = val * (1.0f / (1.0f + expf(-g)));
            const float amax = warp_reduce_max<32>(fabsf(xi));
            const float d    = amax / 127.0f;
            const int8_t q   = (amax == 0.0f) ? 0 : (int8_t) roundf(xi / d);
            const int ib     = qh * 4 + blk;
            dst_q8[ib].qs[lane] = q;
            if (lane == 0) {
                dst_q8[ib].ds = make_half2(__float2half(d), __float2half(0.0f));
            }
        }
    };

    if (n_splits == 1) {
        float * fo = s_out + warp * 132;
        ((float4 *) fo)[lane] = o;
        __syncwarp();
        finalize_head(m_max, l_sum, fo);
        return;
    }

    float * sp_ptr = scratch + ((size_t) (kvh * MAX_SPLITS + sp) * 3 + warp) * 132;
    if (l_sum > 0.0f) {
        ((float4 *) sp_ptr)[lane] = o;
    }
    if (lane == 0) {
        sp_ptr[128] = m_max;
        sp_ptr[129] = l_sum;
    }
    __syncthreads();
    __shared__ bool is_last;
    if (warp == 0 && lane == 0) {
        __threadfence();
        const int done = atomicAdd(&cnt[kvh], 1) + 1;
        is_last = (done == n_splits);
        if (is_last) {
            cnt[kvh] = 0;
        }
    }
    __syncthreads();
    if (!is_last) {
        return;
    }

    float fm = -1e30f;
    for (int s = 0; s < n_splits; ++s) {
        const float * sp_p = scratch + ((size_t) (kvh * MAX_SPLITS + s) * 3 + warp) * 132;
        if (sp_p[129] > 0.0f) {
            fm = fmaxf(fm, sp_p[128]);
        }
    }
    float fl = 0.0f;
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int s = 0; s < n_splits; ++s) {
        const float * sp_p = scratch + ((size_t) (kvh * MAX_SPLITS + s) * 3 + warp) * 132;
        const float sl = sp_p[129];
        if (sl == 0.0f) {
            continue;
        }
        const float es = expf(sp_p[128] - fm);
        fl += sl * es;
        const float4 sv = ((const float4 *) sp_p)[lane];
        acc.x += sv.x * es;
        acc.y += sv.y * es;
        acc.z += sv.z * es;
        acc.w += sv.w * es;
    }
    float * fo = s_out + warp * 132;
    ((float4 *) fo)[lane] = acc;
    __syncwarp();
    finalize_head(fm, fl, fo);
#endif
}

static inline int32_t * inkling_get_fattn_counters(const int device, const int stream_no) {
    static int32_t * d_cnts[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS] = {};
    const int s = (stream_no >= 0 && stream_no < GGML_CUDA_MAX_STREAMS) ? stream_no : 0;
    if (d_cnts[device][s] == nullptr) {
        int32_t * ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&ptr, 16 * sizeof(int32_t)));
        CUDA_CHECK(cudaMemset(ptr, 0, 16 * sizeof(int32_t)));
        d_cnts[device][s] = ptr;
    }
    return d_cnts[device][s];
}

static inline void ggml_cuda_op_inkling_rel_pos_fattn_wo(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * Q,
        const ggml_tensor * K,
        const ggml_tensor * V,
        const ggml_tensor * R,
        const ggml_tensor * rel_bias,
        const ggml_tensor * sinks,
        const ggml_tensor * rel_pos,
        const ggml_tensor * raw_mask,
        const ggml_tensor * Gcur,
        const ggml_tensor * wo,
        ggml_tensor * dst,
        const float kq_scale) {
    constexpr int MAX_SPLITS = 64;
    const int n_kv     = (int) K->ne[1];
    const int n_extent = (int) rel_bias->ne[1];
    const int nrows    = (int) wo->ne[1];
    const int nb       = (int) (wo->ne[0] / QK8_0); // 96
    const int n_tiles  = (n_kv + 15) / 16;

    int n_splits;
    if (n_kv <= 256) {
        n_splits = 16;
    } else if (n_kv <= 2048) {
        n_splits = 32;
    } else if (n_kv <= 12288) {
        n_splits = 48;
    } else {
        n_splits = 64;
    }
    if (n_splits > n_tiles) {
        n_splits = n_tiles;
    }
    if (n_splits < 1) {
        n_splits = 1;
    }

    const size_t q_head_stride_f = Q->nb[2] / sizeof(float);
    const size_t k_head_stride_h = K->nb[2] / sizeof(half);
    const size_t k_tok_stride_h  = K->nb[1] / sizeof(half);
    const size_t v_head_stride_h = V->nb[2] / sizeof(half);
    const size_t v_tok_stride_h  = V->nb[1] / sizeof(half);

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<block_q8_1> y_q8(ctx.pool(), (size_t) nb);
    ggml_cuda_pool_alloc<float>      scratch(ctx.pool());
    float * scratch_ptr = nullptr;
    if (n_splits > 1) {
        scratch.alloc((size_t) 8 * MAX_SPLITS * 3 * 132);
        scratch_ptr = scratch.get();
    }
    int32_t * d_cnt = inkling_get_fattn_counters(ctx.device, ctx.curr_stream_no);

    const dim3 fa_grid(n_splits, 8, 1);
    const dim3 fa_block(32, 3, 1);
    if (raw_mask->type == GGML_TYPE_F16) {
        inkling_rel_pos_fattn_gqa3_cpasync_q8<true, MAX_SPLITS><<<fa_grid, fa_block, 0, stream>>>(
            (const float *) Q->data, (const half *) K->data, (const half *) V->data,
            (const float *) R->data, (const float *) rel_bias->data, (const float *) sinks->data,
            (const int32_t *) rel_pos->data, raw_mask->data, (const float *) Gcur->data,
            y_q8.get(), scratch_ptr, d_cnt, n_kv, n_extent,
            q_head_stride_f, k_head_stride_h, k_tok_stride_h, v_head_stride_h, v_tok_stride_h,
            kq_scale, n_splits);
    } else {
        inkling_rel_pos_fattn_gqa3_cpasync_q8<false, MAX_SPLITS><<<fa_grid, fa_block, 0, stream>>>(
            (const float *) Q->data, (const half *) K->data, (const half *) V->data,
            (const float *) R->data, (const float *) rel_bias->data, (const float *) sinks->data,
            (const int32_t *) rel_pos->data, raw_mask->data, (const float *) Gcur->data,
            y_q8.get(), scratch_ptr, d_cnt, n_kv, n_extent,
            q_head_stride_f, k_head_stride_h, k_tok_stride_h, v_head_stride_h, v_tok_stride_h,
            kq_scale, n_splits);
    }

    constexpr int NWARPS = 4;
    const dim3 wo_block(32, NWARPS, 1);
    const dim3 wo_grid((nrows + NWARPS - 1) / NWARPS, 1, 1);
    if (nb == 96 && (((uintptr_t) wo->data) & 3) == 0) {
        inkling_q8_0_mmv_align32_f32<NWARPS, 96><<<wo_grid, wo_block, 0, stream>>>(
            (const block_q8_0 *) wo->data, y_q8.get(), (float *) dst->data, nrows, 96);
    } else if (nb == 96) {
        inkling_q8_0_mmv_f32<NWARPS, 96><<<wo_grid, wo_block, 0, stream>>>(
            (const block_q8_0 *) wo->data, y_q8.get(), (float *) dst->data, nrows, 96);
    } else {
        inkling_q8_0_mmv_f32<NWARPS, 0><<<wo_grid, wo_block, 0, stream>>>(
            (const block_q8_0 *) wo->data, y_q8.get(), (float *) dst->data, nrows, nb);
    }
}

// ============================================================================
// Wave 8 Mega-Fusions (Batched T = 2..16 Speculative Verification Breakthrough:
// Batched Router + 1-Block Expert Grouping + Expert-Grouped cp.async.cg NVFP4 & Q8_0 MoE)
// ============================================================================

struct inkling_moe_groups {
    int32_t num_active;
    int32_t active_exp[128];
    int32_t exp_count[128];
    uint8_t exp_toks[128][16];
    uint8_t exp_slots[128][16];
};

// 24a. Batched Router + TopK128 + Optional Idle-SM Input Q8_1 Quantize (exact compile-time T = 2..8, ZERO spills!)
template <bool PREINT, bool WITH_Q8, int T>
static __global__ void inkling_fused_router_topk128_q8_batched(
        const float4 * __restrict__ w,
        const float4 * __restrict__ x,
        const float *  __restrict__ bias,
        float *        __restrict__ s_wt,
        float *        __restrict__ s_sel,
        int32_t *      __restrict__ d_cnt,
        int32_t *      __restrict__ ids,
        float *        __restrict__ weights,
        block_q8_1 *   __restrict__ y_q8,
        const int D4,
        const int n_exp_used,
        const int ids_stride,
        const int n_qblks,
        const bool with_norm,
        const bool norm_add_eps,
        const float clamp_val,
        const float scale_val) {
    if constexpr (WITH_Q8) {
        if (blockIdx.x >= 64) {
            const int D    = D4 * 4;
            const int nb   = D >> 5;
            const int qidx = blockIdx.x - 64;
            const int t    = qidx / n_qblks;
            const int qblk = qidx - t * n_qblks;
            const int tid  = threadIdx.y * 32 + threadIdx.x; // 0..127
            const float * xf = (const float *) x + (size_t) t * D;
#pragma unroll
            for (int iter = 0; iter < 2; ++iter) {
                const int i0 = qblk * 256 + iter * 128 + tid;
                if (i0 < D) {
                    const float xi    = xf[i0];
                    const float amax  = warp_reduce_max<32>(fabsf(xi));
                    const float d     = amax / 127.0f;
                    const int8_t q    = (amax == 0.0f) ? 0 : (int8_t) roundf(xi / d);
                    const int ib      = t * nb + (i0 >> 5);
                    const int iqs     = i0 & 31;
                    const int dst_iqs = PREINT ? ((iqs & 16) | ((iqs & 7) << 1) | ((iqs >> 3) & 1)) : iqs;
                    y_q8[ib].qs[dst_iqs] = q;
                    if (iqs == 0) {
                        y_q8[ib].ds = make_half2(__float2half(d), __float2half(0.0f));
                    }
                }
            }
            return;
        }
    }

    const int local_e = threadIdx.y >> 1;
    const int half_w  = threadIdx.y & 1;
    const int e       = blockIdx.x * 2 + local_e;
    const int lane    = threadIdx.x;
    const float4 * wr = w + (size_t) e * D4;

    float acc[T];
#pragma unroll
    for (int t = 0; t < T; ++t) {
        acc[t] = 0.0f;
    }
    for (int i = half_w * 32 + lane; i < D4; i += 64) {
        const float4 wv = __ldg(&wr[i]);
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float4 xv = __ldg(&x[(size_t) t * D4 + i]);
            acc[t] += xv.x * wv.x + xv.y * wv.y + xv.z * wv.z + xv.w * wv.w;
        }
    }

    __shared__ float sm[2][2][T];
#pragma unroll
    for (int t = 0; t < T; ++t) {
        const float v = warp_reduce_sum<32>(acc[t]);
        if (lane == 0) {
            sm[local_e][half_w][t] = v;
        }
    }
    __syncthreads();

    if (threadIdx.y < 2 && lane < T) {
        const int t  = lane;
        const int ex = blockIdx.x * 2 + threadIdx.y;
        const float logit = sm[threadIdx.y][0][t] + sm[threadIdx.y][1][t];
        float wt = 1.0f / (1.0f + expf(-logit));
        if (__isnanf(wt)) {
            wt = -FLT_MAX;
        }
        float sel = bias ? (wt + bias[ex]) : wt;
        if (__isnanf(sel)) {
            sel = -FLT_MAX;
        }
        s_wt[t * 128 + ex]  = wt;
        s_sel[t * 128 + ex] = sel;
    }
    __syncthreads();

    __shared__ bool is_last;
    if (threadIdx.y == 0 && lane == 0) {
        __threadfence();
        const int done = atomicAdd(d_cnt, 1) + 1;
        is_last = (done == 64);
        if (is_last) {
            *d_cnt = 0;
        }
    }
    __syncthreads();

    if (!is_last) {
        return;
    }

    for (int t = threadIdx.y; t < T; t += 4) {
        float wt[4], sel[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            wt[i]  = s_wt[t * 128 + i * 32 + lane];
            sel[i] = s_sel[t * 128 + i * 32 + lane];
        }
        float out_w = 0.0f;
        float wsum  = 0.0f;
        int32_t * ids_t = ids + (size_t) t * ids_stride;
        float *   w_t   = weights + (size_t) t * n_exp_used;
        for (int k = 0; k < n_exp_used; ++k) {
            float mv = wt[0];
            float ms = sel[0];
            int   me = lane;
#pragma unroll
            for (int i = 1; i < 4; ++i) {
                const int ex = i * 32 + lane;
                if (sel[i] > ms) {
                    mv = wt[i];
                    ms = sel[i];
                    me = ex;
                }
            }
#pragma unroll
            for (int m = 16; m > 0; m >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffff, mv, m);
                const float os = __shfl_xor_sync(0xffffffff, ms, m);
                const int   oe = __shfl_xor_sync(0xffffffff, me, m);
                if (os > ms || (os == ms && oe < me)) {
                    mv = ov;
                    ms = os;
                    me = oe;
                }
            }
            if ((me & 31) == lane) {
                sel[me >> 5] = -INFINITY;
                ids_t[k] = me;
                if (with_norm) {
                    wsum += mv;
                }
            }
            if (k == lane) {
                out_w = mv;
            }
        }
        if (with_norm) {
            wsum = warp_reduce_sum<32>(wsum);
            wsum = norm_add_eps ? (wsum + clamp_val) : fmaxf(wsum, clamp_val);
            out_w *= (1.0f / wsum);
        }
        if (lane < n_exp_used) {
            w_t[lane] = out_w * scale_val;
        }
    }
}

template <bool PREINT, bool WITH_Q8>
static inline void inkling_launch_router_batched(
        const float4 * w, const float4 * x, const float * bias,
        float * s_wt, float * s_sel, int32_t * d_cnt,
        int32_t * ids, float * weights, block_q8_1 * y_q8,
        const int D4, const int T, const int n_exp_used, const int ids_stride, const int n_qblks,
        const bool with_norm, const bool norm_add_eps, const float clamp_val, const float scale_val,
        cudaStream_t stream) {
    const int nblks = 64 + (WITH_Q8 ? n_qblks * T : 0);
    const dim3 bd(32, 4, 1);
    switch (T) {
        case 2: inkling_fused_router_topk128_q8_batched<PREINT, WITH_Q8, 2><<<nblks, bd, 0, stream>>>(w, x, bias, s_wt, s_sel, d_cnt, ids, weights, y_q8, D4, n_exp_used, ids_stride, n_qblks, with_norm, norm_add_eps, clamp_val, scale_val); break;
        case 3: inkling_fused_router_topk128_q8_batched<PREINT, WITH_Q8, 3><<<nblks, bd, 0, stream>>>(w, x, bias, s_wt, s_sel, d_cnt, ids, weights, y_q8, D4, n_exp_used, ids_stride, n_qblks, with_norm, norm_add_eps, clamp_val, scale_val); break;
        case 4: inkling_fused_router_topk128_q8_batched<PREINT, WITH_Q8, 4><<<nblks, bd, 0, stream>>>(w, x, bias, s_wt, s_sel, d_cnt, ids, weights, y_q8, D4, n_exp_used, ids_stride, n_qblks, with_norm, norm_add_eps, clamp_val, scale_val); break;
        case 5: inkling_fused_router_topk128_q8_batched<PREINT, WITH_Q8, 5><<<nblks, bd, 0, stream>>>(w, x, bias, s_wt, s_sel, d_cnt, ids, weights, y_q8, D4, n_exp_used, ids_stride, n_qblks, with_norm, norm_add_eps, clamp_val, scale_val); break;
        case 6: inkling_fused_router_topk128_q8_batched<PREINT, WITH_Q8, 6><<<nblks, bd, 0, stream>>>(w, x, bias, s_wt, s_sel, d_cnt, ids, weights, y_q8, D4, n_exp_used, ids_stride, n_qblks, with_norm, norm_add_eps, clamp_val, scale_val); break;
        case 7: inkling_fused_router_topk128_q8_batched<PREINT, WITH_Q8, 7><<<nblks, bd, 0, stream>>>(w, x, bias, s_wt, s_sel, d_cnt, ids, weights, y_q8, D4, n_exp_used, ids_stride, n_qblks, with_norm, norm_add_eps, clamp_val, scale_val); break;
        default: inkling_fused_router_topk128_q8_batched<PREINT, WITH_Q8, 8><<<nblks, bd, 0, stream>>>(w, x, bias, s_wt, s_sel, d_cnt, ids, weights, y_q8, D4, n_exp_used, ids_stride, n_qblks, with_norm, norm_add_eps, clamp_val, scale_val); break;
    }
}

// 24b. 1-Block Expert Grouping Kernel (deduplicates unique active experts for T = 2..8, ~1.5 us, CUDA-graph safe!)
static __global__ void inkling_moe_group_experts(
        const int32_t * __restrict__ ids,
        inkling_moe_groups * __restrict__ grp,
        const int n_experts,
        const int n_exp_used,
        const int ids_stride,
        const int T) {
    __shared__ int32_t s_ids[192];
    const int e     = threadIdx.x; // 0..127
    const int warp  = e >> 5;
    const int lane  = e & 31;
    const int n_prs = n_exp_used * T;

    for (int p = e; p < n_prs; p += 128) {
        const int t = p / n_exp_used;
        const int k = p - t * n_exp_used;
        s_ids[p] = ids[(size_t) t * ids_stride + k];
    }
    __syncthreads();

    uint8_t my_t[16];
    uint8_t my_s[16];
    int cnt = 0;
    if (e < n_experts) {
        for (int t = 0; t < T; ++t) {
            const int base_p = t * n_exp_used;
            for (int k = 0; k < n_exp_used; ++k) {
                if (s_ids[base_p + k] == e && cnt < 16) {
                    my_t[cnt] = (uint8_t) t;
                    my_s[cnt] = (uint8_t) (base_p + k);
                    cnt++;
                }
            }
        }
    }

    const uint32_t mask = __ballot_sync(0xffffffff, cnt > 0);
    const int in_warp   = __popc(mask & ((1u << lane) - 1u));
    const int warp_tot  = __popc(mask);

    __shared__ int s_wtot[4];
    if (lane == 0) {
        s_wtot[warp] = warp_tot;
    }
    __syncthreads();

    int base_u = 0;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        if (w < warp) {
            base_u += s_wtot[w];
        }
    }
    if (e == 0) {
        grp->num_active = s_wtot[0] + s_wtot[1] + s_wtot[2] + s_wtot[3];
    }
    if (cnt > 0) {
        const int u = base_u + in_warp;
        grp->active_exp[u] = e;
        grp->exp_count[u]  = cnt;
        for (int c = 0; c < cnt; ++c) {
            grp->exp_toks[u][c]  = my_t[c];
            grp->exp_slots[u][c] = my_s[c];
        }
    }
}

// 24c. Zero-Spill Expert-Grouped cp.async.cg NVFP4 gate_up + scale + SiTU (T = 2..8):
//      Dispatches exact compile-time CNT (1..8) outside the k-loop so ALL sums live 100% in hardware registers!
template <int CNT>
static __device__ __forceinline__ void inkling_nvfp4_nb36_dot_cnt(
        const block_nvfp4 * __restrict__ sb,
        const block_q8_1 * __restrict__ vy,
        const uint8_t * __restrict__ s_toks,
        const int tid,
        float out[CNT]) {
    const int iqs = (tid & 1) * 4;
    const int is0 = iqs >> 1;
#pragma unroll
    for (int c = 0; c < CNT; ++c) {
        out[c] = 0.0f;
    }
    int tok_off[CNT];
#pragma unroll
    for (int c = 0; c < CNT; ++c) {
        tok_off[c] = (int) s_toks[c] * 72 + (is0 >> 1);
    }
#pragma unroll
    for (int k = tid >> 1; k < 36; k += 16) {
        const block_nvfp4 * bq4 = &sb[k];
        const int * q4p = (const int *) bq4->qs + iqs;
        const uint16_t dp = *reinterpret_cast<const uint16_t *>(bq4->d + is0);
        const float d0 = ggml_cuda_ue4m3_to_fp32(dp & 0xFFu);
        const float d1 = ggml_cuda_ue4m3_to_fp32(dp >> 8);
        const int2 v0 = inkling_unpack_nvfp4_raw(q4p[0]);
        const int2 v1 = inkling_unpack_nvfp4_raw(q4p[1]);
        const int2 v2 = inkling_unpack_nvfp4_raw(q4p[2]);
        const int2 v3 = inkling_unpack_nvfp4_raw(q4p[3]);
        const int k2 = k * 2;
#pragma unroll
        for (int c = 0; c < CNT; ++c) {
            const block_q8_1 * bq8 = &vy[tok_off[c] + k2];
            const int * uq = (const int *) bq8->qs;
            int s0 = ggml_cuda_dp4a(v0.x, __ldg(uq + 0), 0);
            s0     = ggml_cuda_dp4a(v0.y, __ldg(uq + 1), s0);
            s0     = ggml_cuda_dp4a(v1.x, __ldg(uq + 2), s0);
            s0     = ggml_cuda_dp4a(v1.y, __ldg(uq + 3), s0);
            int s1 = ggml_cuda_dp4a(v2.x, __ldg(uq + 4), 0);
            s1     = ggml_cuda_dp4a(v2.y, __ldg(uq + 5), s1);
            s1     = ggml_cuda_dp4a(v3.x, __ldg(uq + 6), s1);
            s1     = ggml_cuda_dp4a(v3.y, __ldg(uq + 7), s1);
            out[c] += __low2float(bq8->ds) * (d0 * (float) s0 + d1 * (float) s1);
        }
    }
#pragma unroll
    for (int c = 0; c < CNT; ++c) {
        out[c] = warp_reduce_sum<32>(out[c]);
    }
}

template <int NWARPS>
static __global__ void inkling_fused_nvfp4_gate_up_situ_grouped_cpasync_nb36(
        const block_nvfp4 * __restrict__ vx,
        const block_q8_1 * __restrict__ vy,
        const inkling_moe_groups * __restrict__ grp,
        const float * __restrict__ sc,
        float * __restrict__ dst,
        const int n_ff,
        const size_t stride_ch_x,
        const float inv_beta, const float beta,
        const float inv_lb,   const float lb,
        const bool has_lb) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const int u = blockIdx.y;
    if (u >= grp->num_active) {
        return;
    }
    const int row = blockIdx.x * NWARPS + threadIdx.y;
    if (row >= n_ff) {
        return;
    }

    __shared__ __align__(16) uint32_t s_row[NWARPS][324];
    __shared__ int     s_ch;
    __shared__ int     s_cnt;
    __shared__ float   s_scale;
    __shared__ uint8_t s_toks[8];
    __shared__ uint8_t s_slots[8];

    const int tid  = threadIdx.x;
    const int warp = threadIdx.y;
    if (warp == 0) {
        if (tid == 0) {
            const int ch = grp->active_exp[u];
            s_ch    = ch;
            s_cnt   = grp->exp_count[u];
            s_scale = sc ? sc[ch] : 1.0f;
        }
        if (tid < 8) {
            s_toks[tid]  = grp->exp_toks[u][tid];
            s_slots[tid] = grp->exp_slots[u][tid];
        }
    }
    __syncthreads();

    const int ch  = s_ch;
    const int cnt = s_cnt;
    const uint4 * xg4 = (const uint4 *) (vx + (size_t) ch * stride_ch_x + (size_t) row * 36);
    const uint4 * xu4 = (const uint4 *) (vx + (size_t) ch * stride_ch_x + (size_t) (row + n_ff) * 36);
    uint4 * sm4 = (uint4 *) s_row[warp];
    const block_nvfp4 * sb = (const block_nvfp4 *) s_row[warp];

    auto load_row = [&](const uint4 * src4) {
        inkling_cp_async_cg_16(&sm4[tid],      &src4[tid]);
        inkling_cp_async_cg_16(&sm4[tid + 32], &src4[tid + 32]);
        if (tid < 17) {
            inkling_cp_async_cg_16(&sm4[tid + 64], &src4[tid + 64]);
        }
        inkling_cp_async_commit();
        inkling_cp_async_wait<0>();
        __syncwarp();
    };

    auto exec_cnt = [&](auto tag) {
        constexpr int CNT = decltype(tag)::value;
        float sg[CNT], su[CNT];
        load_row(xg4);
        inkling_nvfp4_nb36_dot_cnt<CNT>(sb, vy, s_toks, tid, sg);
        __syncwarp();
        load_row(xu4);
        inkling_nvfp4_nb36_dot_cnt<CNT>(sb, vy, s_toks, tid, su);
        if (tid == 0) {
            const float s = s_scale;
#pragma unroll
            for (int c = 0; c < CNT; ++c) {
                dst[(size_t) s_slots[c] * n_ff + row] = inkling_situ_elem(sg[c] * s, su[c] * s, inv_beta, beta, inv_lb, lb, has_lb);
            }
        }
    };

    switch (cnt) {
        case 1: exec_cnt(std::integral_constant<int, 1>{}); break;
        case 2: exec_cnt(std::integral_constant<int, 2>{}); break;
        case 3: exec_cnt(std::integral_constant<int, 3>{}); break;
        case 4: exec_cnt(std::integral_constant<int, 4>{}); break;
        case 5: exec_cnt(std::integral_constant<int, 5>{}); break;
        case 6: exec_cnt(std::integral_constant<int, 6>{}); break;
        case 7: exec_cnt(std::integral_constant<int, 7>{}); break;
        default: exec_cnt(std::integral_constant<int, 8>{}); break;
    }
#endif
}

template <int NWARPS>
static __global__ void inkling_fused_nvfp4_gate_up_situ_grouped_gen(
        const block_nvfp4 * __restrict__ vx,
        const block_q8_1 * __restrict__ vy,
        const inkling_moe_groups * __restrict__ grp,
        const float * __restrict__ sc,
        float * __restrict__ dst,
        const int n_ff,
        const int nb,
        const size_t stride_ch_x,
        const float inv_beta, const float beta,
        const float inv_lb,   const float lb,
        const bool has_lb) {
    const int u = blockIdx.y;
    if (u >= grp->num_active) {
        return;
    }
    const int row = blockIdx.x * NWARPS + threadIdx.y;
    if (row >= n_ff) {
        return;
    }
    const int ch  = grp->active_exp[u];
    const int cnt = grp->exp_count[u];
    const int tid = threadIdx.x;
    const int iqs = (tid & 1) * 4;
    const int nb_q8 = nb * 2;
    const block_nvfp4 * xg = vx + (size_t) ch * stride_ch_x + (size_t) row * nb;
    const block_nvfp4 * xu = xg + (size_t) n_ff * nb;
    const float s = sc ? sc[ch] : 1.0f;
    for (int c = 0; c < cnt; ++c) {
        const int tok = grp->exp_toks[u][c];
        float sg = 0.0f, su = 0.0f;
        for (int k = tid >> 1; k < nb; k += 16) {
            sg += inkling_vec_dot_nvfp4_preint(&xg[k], &vy[(size_t) tok * nb_q8 + k * 2], iqs);
            su += inkling_vec_dot_nvfp4_preint(&xu[k], &vy[(size_t) tok * nb_q8 + k * 2], iqs);
        }
        sg = warp_reduce_sum<32>(sg);
        su = warp_reduce_sum<32>(su);
        if (tid == 0) {
            dst[(size_t) grp->exp_slots[u][c] * n_ff + row] = inkling_situ_elem(sg * s, su * s, inv_beta, beta, inv_lb, lb, has_lb);
        }
    }
}

// 24d. Zero-Spill Expert-Grouped cp.async.cg NVFP4 down (2 rows = 36 blocks = 1296 B = 81 uint4s per cp.async.cg!)
template <int CNT>
static __device__ __forceinline__ void inkling_nvfp4_nb18_dot_cnt(
        const block_nvfp4 * __restrict__ rb,
        const block_q8_1 * __restrict__ vy,
        const uint8_t * __restrict__ s_slots,
        const float * __restrict__ s_weff,
        float * __restrict__ dst_tmp,
        const int nrows,
        const int row,
        const int tid) {
    const int iqs = (tid & 1) * 4;
    const int is0 = iqs >> 1;
    float sums[CNT];
    int sl_off[CNT];
#pragma unroll
    for (int c = 0; c < CNT; ++c) {
        sums[c] = 0.0f;
        sl_off[c] = (int) s_slots[c] * 36 + (is0 >> 1);
    }
#pragma unroll
    for (int k = tid >> 1; k < 18; k += 16) {
        const block_nvfp4 * bq4 = &rb[k];
        const int * q4p = (const int *) bq4->qs + iqs;
        const uint16_t dp = *reinterpret_cast<const uint16_t *>(bq4->d + is0);
        const float d0 = ggml_cuda_ue4m3_to_fp32(dp & 0xFFu);
        const float d1 = ggml_cuda_ue4m3_to_fp32(dp >> 8);
        const int2 v0 = inkling_unpack_nvfp4_raw(q4p[0]);
        const int2 v1 = inkling_unpack_nvfp4_raw(q4p[1]);
        const int2 v2 = inkling_unpack_nvfp4_raw(q4p[2]);
        const int2 v3 = inkling_unpack_nvfp4_raw(q4p[3]);
        const int k2 = k * 2;
#pragma unroll
        for (int c = 0; c < CNT; ++c) {
            const block_q8_1 * bq8 = &vy[sl_off[c] + k2];
            const int * uq = (const int *) bq8->qs;
            int s0 = ggml_cuda_dp4a(v0.x, __ldg(uq + 0), 0);
            s0     = ggml_cuda_dp4a(v0.y, __ldg(uq + 1), s0);
            s0     = ggml_cuda_dp4a(v1.x, __ldg(uq + 2), s0);
            s0     = ggml_cuda_dp4a(v1.y, __ldg(uq + 3), s0);
            int s1 = ggml_cuda_dp4a(v2.x, __ldg(uq + 4), 0);
            s1     = ggml_cuda_dp4a(v2.y, __ldg(uq + 5), s1);
            s1     = ggml_cuda_dp4a(v3.x, __ldg(uq + 6), s1);
            s1     = ggml_cuda_dp4a(v3.y, __ldg(uq + 7), s1);
            sums[c] += __low2float(bq8->ds) * (d0 * (float) s0 + d1 * (float) s1);
        }
    }
#pragma unroll
    for (int c = 0; c < CNT; ++c) {
        const float v = warp_reduce_sum<32>(sums[c]);
        if (tid == 0) {
            dst_tmp[(size_t) s_slots[c] * nrows + row] = v * s_weff[c];
        }
    }
}

template <int NWARPS>
static __global__ void inkling_fused_nvfp4_down_grouped_cpasync_nb18(
        const block_nvfp4 * __restrict__ vx,
        const block_q8_1 * __restrict__ vy,
        const inkling_moe_groups * __restrict__ grp,
        const float * __restrict__ sc,
        const float * __restrict__ rw,
        float * __restrict__ dst_tmp,
        const int nrows,
        const size_t stride_ch_x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const int u = blockIdx.y;
    if (u >= grp->num_active) {
        return;
    }
    const int row0 = (blockIdx.x * NWARPS + threadIdx.y) * 2;
    if (row0 >= nrows) {
        return;
    }

    __shared__ __align__(16) uint32_t s_row[NWARPS][324];
    __shared__ int     s_ch;
    __shared__ int     s_cnt;
    __shared__ uint8_t s_slots[8];
    __shared__ float   s_weff[8];

    const int tid  = threadIdx.x;
    const int warp = threadIdx.y;
    if (warp == 0 && tid == 0) {
        s_ch  = grp->active_exp[u];
        s_cnt = grp->exp_count[u];
    }
    __syncthreads();
    const int ch  = s_ch;
    const int cnt = s_cnt;
    if (warp == 0 && tid < cnt) {
        const uint8_t sl = grp->exp_slots[u][tid];
        s_slots[tid] = sl;
        s_weff[tid]  = (sc ? sc[ch] : 1.0f) * rw[sl];
    }
    __syncthreads();

    const uint4 * src4 = (const uint4 *) (vx + (size_t) ch * stride_ch_x + (size_t) row0 * 18);
    uint4 * sm4 = (uint4 *) s_row[warp];
    inkling_cp_async_cg_16(&sm4[tid],      &src4[tid]);
    inkling_cp_async_cg_16(&sm4[tid + 32], &src4[tid + 32]);
    if (tid < 17) {
        inkling_cp_async_cg_16(&sm4[tid + 64], &src4[tid + 64]);
    }
    inkling_cp_async_commit();
    inkling_cp_async_wait<0>();
    __syncwarp();

    const block_nvfp4 * sb = (const block_nvfp4 *) s_row[warp];

    auto exec_cnt = [&](auto tag) {
        constexpr int CNT = decltype(tag)::value;
        inkling_nvfp4_nb18_dot_cnt<CNT>(sb,      vy, s_slots, s_weff, dst_tmp, nrows, row0,     tid);
        inkling_nvfp4_nb18_dot_cnt<CNT>(sb + 18, vy, s_slots, s_weff, dst_tmp, nrows, row0 + 1, tid);
    };

    switch (cnt) {
        case 1: exec_cnt(std::integral_constant<int, 1>{}); break;
        case 2: exec_cnt(std::integral_constant<int, 2>{}); break;
        case 3: exec_cnt(std::integral_constant<int, 3>{}); break;
        case 4: exec_cnt(std::integral_constant<int, 4>{}); break;
        case 5: exec_cnt(std::integral_constant<int, 5>{}); break;
        case 6: exec_cnt(std::integral_constant<int, 6>{}); break;
        case 7: exec_cnt(std::integral_constant<int, 7>{}); break;
        default: exec_cnt(std::integral_constant<int, 8>{}); break;
    }
#endif
}

template <int NWARPS>
static __global__ void inkling_fused_nvfp4_down_grouped_gen(
        const block_nvfp4 * __restrict__ vx,
        const block_q8_1 * __restrict__ vy,
        const inkling_moe_groups * __restrict__ grp,
        const float * __restrict__ sc,
        const float * __restrict__ rw,
        float * __restrict__ dst_tmp,
        const int nrows,
        const int nb,
        const size_t stride_ch_x) {
    const int u = blockIdx.y;
    if (u >= grp->num_active) {
        return;
    }
    const int row = blockIdx.x * NWARPS + threadIdx.y;
    if (row >= nrows) {
        return;
    }
    const int ch  = grp->active_exp[u];
    const int cnt = grp->exp_count[u];
    const int tid = threadIdx.x;
    const int iqs = (tid & 1) * 4;
    const int nb_q8 = nb * 2;
    const block_nvfp4 * xr = vx + (size_t) ch * stride_ch_x + (size_t) row * nb;
    const float s = sc ? sc[ch] : 1.0f;
    for (int c = 0; c < cnt; ++c) {
        const int sl = grp->exp_slots[u][c];
        float sum = 0.0f;
        for (int k = tid >> 1; k < nb; k += 16) {
            sum += inkling_vec_dot_nvfp4_preint(&xr[k], &vy[(size_t) sl * nb_q8 + k * 2], iqs);
        }
        sum = warp_reduce_sum<32>(sum);
        if (tid == 0) {
            dst_tmp[(size_t) sl * nrows + row] = sum * s * rw[sl];
        }
    }
}

// 24e. Zero-Spill Expert-Grouped Q8_0 gate_up + SiTU & down (for the 4 Q8_0 MoE layers, T = 2..8)
template <int CNT>
static __device__ __forceinline__ void inkling_q8_0_row_dot_cnt(
        const block_q8_0 * __restrict__ xr,
        const block_q8_1 * __restrict__ vy,
        const uint8_t * __restrict__ idxs,
        const int nb,
        const int tid,
        float out[CNT]) {
    const int iqs = (tid & 3) * 2;
    int off[CNT];
#pragma unroll
    for (int c = 0; c < CNT; ++c) {
        out[c] = 0.0f;
        off[c] = (int) idxs[c] * nb;
    }
    for (int k = tid >> 2; k < nb; k += 8) {
        const block_q8_0 * bx = &xr[k];
        const int x0 = inkling_get_int_b2(bx->qs, iqs + 0);
        const int x1 = inkling_get_int_b2(bx->qs, iqs + 1);
        const float dx = __half2float(__ldg(&bx->d));
#pragma unroll
        for (int c = 0; c < CNT; ++c) {
            const block_q8_1 * by = &vy[off[c] + k];
            const int * uy = (const int *) by->qs;
            int s = ggml_cuda_dp4a(x0, __ldg(uy + iqs + 0), 0);
            s     = ggml_cuda_dp4a(x1, __ldg(uy + iqs + 1), s);
            out[c] += (float) s * dx * __low2float(by->ds);
        }
    }
#pragma unroll
    for (int c = 0; c < CNT; ++c) {
        out[c] = warp_reduce_sum<32>(out[c]);
    }
}

template <int NWARPS>
static __global__ void inkling_fused_q8_0_gate_up_situ_grouped_f32(
        const block_q8_0 * __restrict__ vx,
        const block_q8_1 * __restrict__ vy,
        const inkling_moe_groups * __restrict__ grp,
        float * __restrict__ dst,
        const int n_ff,
        const int nb,
        const size_t stride_ch_x,
        const float inv_beta, const float beta,
        const float inv_lb,   const float lb,
        const bool has_lb) {
    const int u = blockIdx.y;
    if (u >= grp->num_active) {
        return;
    }
    const int row = blockIdx.x * NWARPS + threadIdx.y;
    if (row >= n_ff) {
        return;
    }
    __shared__ int s_ch, s_cnt;
    __shared__ uint8_t s_toks[8], s_slots[8];
    const int tid = threadIdx.x;
    if (threadIdx.y == 0) {
        if (tid == 0) {
            s_ch  = grp->active_exp[u];
            s_cnt = grp->exp_count[u];
        }
        if (tid < 8) {
            s_toks[tid]  = grp->exp_toks[u][tid];
            s_slots[tid] = grp->exp_slots[u][tid];
        }
    }
    __syncthreads();

    const block_q8_0 * xg = vx + (size_t) s_ch * stride_ch_x + (size_t) row * nb;
    const block_q8_0 * xu = xg + (size_t) n_ff * nb;

    auto exec_cnt = [&](auto tag) {
        constexpr int CNT = decltype(tag)::value;
        float sg[CNT], su[CNT];
        inkling_q8_0_row_dot_cnt<CNT>(xg, vy, s_toks, nb, tid, sg);
        inkling_q8_0_row_dot_cnt<CNT>(xu, vy, s_toks, nb, tid, su);
        if (tid == 0) {
#pragma unroll
            for (int c = 0; c < CNT; ++c) {
                dst[(size_t) s_slots[c] * n_ff + row] = inkling_situ_elem(sg[c], su[c], inv_beta, beta, inv_lb, lb, has_lb);
            }
        }
    };
    switch (s_cnt) {
        case 1: exec_cnt(std::integral_constant<int, 1>{}); break;
        case 2: exec_cnt(std::integral_constant<int, 2>{}); break;
        case 3: exec_cnt(std::integral_constant<int, 3>{}); break;
        case 4: exec_cnt(std::integral_constant<int, 4>{}); break;
        case 5: exec_cnt(std::integral_constant<int, 5>{}); break;
        case 6: exec_cnt(std::integral_constant<int, 6>{}); break;
        case 7: exec_cnt(std::integral_constant<int, 7>{}); break;
        default: exec_cnt(std::integral_constant<int, 8>{}); break;
    }
}

template <int NWARPS>
static __global__ void inkling_fused_q8_0_down_grouped_f32(
        const block_q8_0 * __restrict__ vx,
        const block_q8_1 * __restrict__ vy,
        const inkling_moe_groups * __restrict__ grp,
        const float * __restrict__ rw,
        float * __restrict__ dst_tmp,
        const int nrows,
        const int nb,
        const size_t stride_ch_x) {
    const int u = blockIdx.y;
    if (u >= grp->num_active) {
        return;
    }
    const int row = blockIdx.x * NWARPS + threadIdx.y;
    if (row >= nrows) {
        return;
    }
    __shared__ int s_ch, s_cnt;
    __shared__ uint8_t s_slots[8];
    __shared__ float   s_weff[8];
    const int tid = threadIdx.x;
    if (threadIdx.y == 0 && tid == 0) {
        s_ch  = grp->active_exp[u];
        s_cnt = grp->exp_count[u];
    }
    __syncthreads();
    if (threadIdx.y == 0 && tid < s_cnt) {
        const uint8_t sl = grp->exp_slots[u][tid];
        s_slots[tid] = sl;
        s_weff[tid]  = rw[sl];
    }
    __syncthreads();

    const block_q8_0 * xr = vx + (size_t) s_ch * stride_ch_x + (size_t) row * nb;

    auto exec_cnt = [&](auto tag) {
        constexpr int CNT = decltype(tag)::value;
        float sd[CNT];
        inkling_q8_0_row_dot_cnt<CNT>(xr, vy, s_slots, nb, tid, sd);
        if (tid == 0) {
#pragma unroll
            for (int c = 0; c < CNT; ++c) {
                dst_tmp[(size_t) s_slots[c] * nrows + row] = sd[c] * s_weff[c];
            }
        }
    };
    switch (s_cnt) {
        case 1: exec_cnt(std::integral_constant<int, 1>{}); break;
        case 2: exec_cnt(std::integral_constant<int, 2>{}); break;
        case 3: exec_cnt(std::integral_constant<int, 3>{}); break;
        case 4: exec_cnt(std::integral_constant<int, 4>{}); break;
        case 5: exec_cnt(std::integral_constant<int, 5>{}); break;
        case 6: exec_cnt(std::integral_constant<int, 6>{}); break;
        case 7: exec_cnt(std::integral_constant<int, 7>{}); break;
        default: exec_cnt(std::integral_constant<int, 8>{}); break;
    }
}

// 24f. Deterministic L2-Resident Weighted Reduction ([nrows, n_exp, T] -> [nrows, T])
static __global__ void inkling_moe_reduce_slots_f32x4(
        const float4 * __restrict__ tmp,
        float4 * __restrict__ dst,
        const int nrows4,
        const int n_exp) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nrows4) {
        return;
    }
    const int t = blockIdx.y;
    const int base_s = t * n_exp;
    float4 acc = tmp[(size_t) base_s * nrows4 + i];
    for (int e = 1; e < n_exp; ++e) {
        const float4 v = tmp[(size_t) (base_s + e) * nrows4 + i];
        acc.x += v.x;
        acc.y += v.y;
        acc.z += v.z;
        acc.w += v.w;
    }
    dst[(size_t) t * nrows4 + i] = acc;
}

static inline void ggml_cuda_op_inkling_router_topk128_batched(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * w,
        const ggml_tensor * x,
        const ggml_tensor * bias,
        ggml_tensor * ids,
        ggml_tensor * weights,
        const int n_exp_used,
        const bool with_norm,
        const bool norm_add_eps,
        const float clamp_val,
        const float scale_val) {
    const int D4         = (int) (w->ne[0] >> 2);
    const int T          = (int) x->ne[1];
    const int ids_stride = (int) (ids->nb[1] / sizeof(int32_t));
    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<float> scratch(ctx.pool(), (size_t) 2 * T * 128);
    int32_t * d_cnt = inkling_get_router_counter(ctx.device, ctx.curr_stream_no);
    inkling_launch_router_batched<false, false>(
        (const float4 *) w->data, (const float4 *) x->data,
        bias ? (const float *) bias->data : nullptr,
        scratch.get(), scratch.get() + (size_t) T * 128, d_cnt,
        (int32_t *) ids->data, (float *) weights->data, nullptr,
        D4, T, n_exp_used, ids_stride, 0, with_norm, norm_add_eps, clamp_val, scale_val, stream);
}

static inline void ggml_cuda_op_inkling_moe_nvfp4_full_batched(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * w_router,
        const ggml_tensor * x,
        const ggml_tensor * bias,
        ggml_tensor * ids,
        ggml_tensor * weights,
        const int n_exp_used,
        const bool with_norm,
        const bool norm_add_eps,
        const float clamp_val,
        const float scale_val,
        const ggml_tensor * w_gu,
        const ggml_tensor * sc_gu,
        const float inv_beta, const float beta,
        const float inv_lb,   const float lb,
        const bool has_lb,
        const ggml_tensor * w_down,
        const ggml_tensor * sc_down,
        ggml_tensor * dst) {
    const int K       = (int) w_gu->ne[0];
    const int T       = (int) x->ne[1];
    const int n_ff    = (int) (w_gu->ne[1] / 2);
    const int nrows   = (int) w_down->ne[1];
    const int n_exp_t = (int) w_gu->ne[2];
    const int nb_gu   = K / QK_NVFP4;
    const int nb_dn   = n_ff / QK_NVFP4;
    const int n_pairs = n_exp_used * T;
    const int max_U   = (n_pairs < n_exp_t) ? n_pairs : n_exp_t;
    const int ids_stride = (int) (ids->nb[1] / sizeof(int32_t));
    const size_t stride_gu = w_gu->nb[2] / sizeof(block_nvfp4);
    const size_t stride_dn = w_down->nb[2] / sizeof(block_nvfp4);
    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<float>              r_scratch(ctx.pool(), (size_t) 2 * T * 128);
    ggml_cuda_pool_alloc<block_q8_1>         y_q8_in(ctx.pool(), (size_t) T * (K / QK8_1));
    ggml_cuda_pool_alloc<inkling_moe_groups> grp(ctx.pool(), 1);
    ggml_cuda_pool_alloc<float>              mid_f32(ctx.pool(), (size_t) n_pairs * n_ff);
    ggml_cuda_pool_alloc<block_q8_1>         y_q8_mid(ctx.pool(), (size_t) n_pairs * (n_ff / QK8_1));
    ggml_cuda_pool_alloc<float>              dn_tmp(ctx.pool(), (size_t) n_pairs * nrows);
    int32_t * d_cnt = inkling_get_router_counter(ctx.device, ctx.curr_stream_no);

    // Launch 1: Batched Router + TopK128 + Pre-Interleaved Input Q8_1 Quantize
    const int n_qblks = (K + 255) / 256;
    inkling_launch_router_batched<true, true>(
        (const float4 *) w_router->data, (const float4 *) x->data,
        bias ? (const float *) bias->data : nullptr,
        r_scratch.get(), r_scratch.get() + (size_t) T * 128, d_cnt,
        (int32_t *) ids->data, (float *) weights->data, y_q8_in.get(),
        K >> 2, T, n_exp_used, ids_stride, n_qblks, with_norm, norm_add_eps, clamp_val, scale_val, stream);

    // Launch 2: 1-Block Expert Grouping (~1.5 us, deduplicates all active experts)
    inkling_moe_group_experts<<<1, 128, 0, stream>>>((const int32_t *) ids->data, grp.get(), n_exp_t, n_exp_used, ids_stride, T);

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const bool use_cpasync_gu = (cc >= GGML_CUDA_CC_AMPERE) && (nb_gu == 36) &&
        (((uintptr_t) w_gu->data & 15) == 0) &&
        (((stride_gu * sizeof(block_nvfp4)) & 15) == 0);
    const bool use_cpasync_dn = (cc >= GGML_CUDA_CC_AMPERE) && (nb_dn == 18) && ((nrows & 1) == 0) &&
        (((uintptr_t) w_down->data & 15) == 0) &&
        (((stride_dn * sizeof(block_nvfp4)) & 15) == 0);

    // Launch 3: Zero-Spill Expert-Grouped NVFP4 gate_up + scale + SiTU
    if (use_cpasync_gu) {
        constexpr int NWARPS = 8;
        const dim3 grid_gu((n_ff + NWARPS - 1) / NWARPS, max_U, 1);
        inkling_fused_nvfp4_gate_up_situ_grouped_cpasync_nb36<NWARPS><<<grid_gu, dim3(32, NWARPS, 1), 0, stream>>>(
            (const block_nvfp4 *) w_gu->data, y_q8_in.get(), grp.get(),
            sc_gu ? (const float *) sc_gu->data : nullptr, mid_f32.get(),
            n_ff, stride_gu, inv_beta, beta, inv_lb, lb, has_lb);
    } else {
        constexpr int NWARPS = 4;
        const dim3 grid_gu((n_ff + NWARPS - 1) / NWARPS, max_U, 1);
        inkling_fused_nvfp4_gate_up_situ_grouped_gen<NWARPS><<<grid_gu, dim3(32, NWARPS, 1), 0, stream>>>(
            (const block_nvfp4 *) w_gu->data, y_q8_in.get(), grp.get(),
            sc_gu ? (const float *) sc_gu->data : nullptr, mid_f32.get(),
            n_ff, nb_gu, stride_gu, inv_beta, beta, inv_lb, lb, has_lb);
    }

    // Launch 4: Pre-interleaved Q8_1 Quantize of SiTU output (L2 resident)
    inkling_quantize_q8_1_nvfp4_preint_f32<<<dim3((n_ff + 255) / 256, n_pairs, 1), 256, 0, stream>>>(
        mid_f32.get(), y_q8_mid.get(), n_ff, n_pairs);

    // Launch 5: Zero-Spill Expert-Grouped NVFP4 down + scale + router weight
    if (use_cpasync_dn) {
        constexpr int NWARPS = 8;
        const dim3 grid_dn((nrows / 2 + NWARPS - 1) / NWARPS, max_U, 1);
        inkling_fused_nvfp4_down_grouped_cpasync_nb18<NWARPS><<<grid_dn, dim3(32, NWARPS, 1), 0, stream>>>(
            (const block_nvfp4 *) w_down->data, y_q8_mid.get(), grp.get(),
            sc_down ? (const float *) sc_down->data : nullptr, (const float *) weights->data,
            dn_tmp.get(), nrows, stride_dn);
    } else {
        constexpr int NWARPS = 4;
        const dim3 grid_dn((nrows + NWARPS - 1) / NWARPS, max_U, 1);
        inkling_fused_nvfp4_down_grouped_gen<NWARPS><<<grid_dn, dim3(32, NWARPS, 1), 0, stream>>>(
            (const block_nvfp4 *) w_down->data, y_q8_mid.get(), grp.get(),
            sc_down ? (const float *) sc_down->data : nullptr, (const float *) weights->data,
            dn_tmp.get(), nrows, nb_dn, stride_dn);
    }

    // Launch 6: Deterministic L2-Resident Reduction -> dst
    const int nrows4 = nrows >> 2;
    inkling_moe_reduce_slots_f32x4<<<dim3((nrows4 + 255) / 256, T, 1), 256, 0, stream>>>(
        (const float4 *) dn_tmp.get(), (float4 *) dst->data, nrows4, n_exp_used);
}

static inline void ggml_cuda_op_inkling_moe_q8_0_full_batched(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * w_router,
        const ggml_tensor * x,
        const ggml_tensor * bias,
        ggml_tensor * ids,
        ggml_tensor * weights,
        const int n_exp_used,
        const bool with_norm,
        const bool norm_add_eps,
        const float clamp_val,
        const float scale_val,
        const ggml_tensor * w_gu,
        const float inv_beta, const float beta,
        const float inv_lb,   const float lb,
        const bool has_lb,
        const ggml_tensor * w_down,
        ggml_tensor * dst) {
    const int K       = (int) w_gu->ne[0];
    const int T       = (int) x->ne[1];
    const int n_ff    = (int) (w_gu->ne[1] / 2);
    const int nrows   = (int) w_down->ne[1];
    const int n_exp_t = (int) w_gu->ne[2];
    const int nb_gu   = K / QK8_0;
    const int nb_dn   = n_ff / QK8_0;
    const int n_pairs = n_exp_used * T;
    const int max_U   = (n_pairs < n_exp_t) ? n_pairs : n_exp_t;
    const int ids_stride = (int) (ids->nb[1] / sizeof(int32_t));
    const size_t stride_gu = w_gu->nb[2] / sizeof(block_q8_0);
    const size_t stride_dn = w_down->nb[2] / sizeof(block_q8_0);
    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<float>              r_scratch(ctx.pool(), (size_t) 2 * T * 128);
    ggml_cuda_pool_alloc<block_q8_1>         y_q8_in(ctx.pool(), (size_t) T * nb_gu);
    ggml_cuda_pool_alloc<inkling_moe_groups> grp(ctx.pool(), 1);
    ggml_cuda_pool_alloc<float>              mid_f32(ctx.pool(), (size_t) n_pairs * n_ff);
    ggml_cuda_pool_alloc<block_q8_1>         y_q8_mid(ctx.pool(), (size_t) n_pairs * nb_dn);
    ggml_cuda_pool_alloc<float>              dn_tmp(ctx.pool(), (size_t) n_pairs * nrows);
    int32_t * d_cnt = inkling_get_router_counter(ctx.device, ctx.curr_stream_no);

    const int n_qblks = (K + 255) / 256;
    inkling_launch_router_batched<false, true>(
        (const float4 *) w_router->data, (const float4 *) x->data,
        bias ? (const float *) bias->data : nullptr,
        r_scratch.get(), r_scratch.get() + (size_t) T * 128, d_cnt,
        (int32_t *) ids->data, (float *) weights->data, y_q8_in.get(),
        K >> 2, T, n_exp_used, ids_stride, n_qblks, with_norm, norm_add_eps, clamp_val, scale_val, stream);

    inkling_moe_group_experts<<<1, 128, 0, stream>>>((const int32_t *) ids->data, grp.get(), n_exp_t, n_exp_used, ids_stride, T);

    constexpr int NWARPS = 8;
    const dim3 grid_gu((n_ff + NWARPS - 1) / NWARPS, max_U, 1);
    const dim3 grid_dn((nrows + NWARPS - 1) / NWARPS, max_U, 1);
    inkling_fused_q8_0_gate_up_situ_grouped_f32<NWARPS><<<grid_gu, dim3(32, NWARPS, 1), 0, stream>>>(
        (const block_q8_0 *) w_gu->data, y_q8_in.get(), grp.get(), mid_f32.get(),
        n_ff, nb_gu, stride_gu, inv_beta, beta, inv_lb, lb, has_lb);

    inkling_quantize_q8_1_rows_f32<<<dim3((n_ff + 255) / 256, n_pairs, 1), 256, 0, stream>>>(
        mid_f32.get(), y_q8_mid.get(), n_ff, n_pairs);

    inkling_fused_q8_0_down_grouped_f32<NWARPS><<<grid_dn, dim3(32, NWARPS, 1), 0, stream>>>(
        (const block_q8_0 *) w_down->data, y_q8_mid.get(), grp.get(), (const float *) weights->data,
        dn_tmp.get(), nrows, nb_dn, stride_dn);

    const int nrows4 = nrows >> 2;
    inkling_moe_reduce_slots_f32x4<<<dim3((nrows4 + 255) / 256, T, 1), 256, 0, stream>>>(
        (const float4 *) dn_tmp.get(), (float4 *) dst->data, nrows4, n_exp_used);
}

// ============================================================================
// Wave 8 Stage 8.4: Low-Rank Precomputed DSpark Head (W_out @ (H @ z + b_h))
// Replaces 492 MB Q8_0 target.output read (~595 us/step) with 103 MB FP16 M = W_out @ H (~125 us/step)
// ============================================================================

static __global__ void inkling_dequant_q8_0_rows_f32(
        const block_q8_0 * __restrict__ src,
        float * __restrict__ dst,
        const int K,
        const int n_rows) {
    const int nb = K >> 5;
    const size_t total = (size_t) n_rows * K;
    const size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const int row = (int) (idx / K);
    const int col = (int) (idx - (size_t) row * K);
    const int ib  = row * nb + (col >> 5);
    const int iqs = col & 31;
    const block_q8_0 & b = src[ib];
    dst[idx] = __half2float(b.d) * (float) b.qs[iqs];
}

static __global__ void inkling_bf16_to_f32_elem(
        const nv_bfloat16 * __restrict__ src,
        float * __restrict__ dst,
        const size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __bfloat162float(src[i]);
    }
}

static __global__ void inkling_f16_to_f32_elem(
        const half * __restrict__ src,
        float * __restrict__ dst,
        const size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __half2float(src[i]);
    }
}

static __global__ void inkling_f32_to_f16_elem(
        const float * __restrict__ src,
        half * __restrict__ dst,
        const size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __float2half(src[i]);
    }
}

struct inkling_dspark_lr_entry {
    int          device;
    const void * w_out;
    const void * w_ho;
    const void * b_ho;
    int          V_full;
    int          K;
    int          R;
    half *       d_M;
    float *      d_bout;
};

static inline const inkling_dspark_lr_entry * inkling_dspark_lr_get_or_build(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * w_out,
        const ggml_tensor * w_ho,
        const ggml_tensor * b_ho) {
    static std::mutex s_mtx;
    static std::vector<inkling_dspark_lr_entry> s_cache;

    const int dev    = ctx.device;
    const int V_full = (int) w_out->ne[1];
    const int K      = (int) w_out->ne[0];
    const int R      = (int) w_ho->ne[0];

    {
        std::lock_guard<std::mutex> lock(s_mtx);
        for (const auto & e : s_cache) {
            if (e.device == dev && e.w_out == w_out->data && e.w_ho == w_ho->data && e.b_ho == b_ho->data &&
                    e.V_full == V_full && e.K == K && e.R == R) {
                return &e;
            }
        }
    }

    cudaStream_t stream = ctx.stream();
    cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &cap_status) != cudaSuccess || cap_status != cudaStreamCaptureStatusNone) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(s_mtx);
    for (const auto & e : s_cache) {
        if (e.device == dev && e.w_out == w_out->data && e.w_ho == w_ho->data && e.b_ho == b_ho->data &&
                e.V_full == V_full && e.K == K && e.R == R) {
            return &e;
        }
    }

    if (s_cache.size() >= 2) {
        int prev_dev = dev;
        if (s_cache.front().device != dev) {
            ggml_cuda_set_device(s_cache.front().device);
        }
        cudaFree(s_cache.front().d_M);
        cudaFree(s_cache.front().d_bout);
        if (s_cache.front().device != prev_dev) {
            ggml_cuda_set_device(prev_dev);
        }
        s_cache.erase(s_cache.begin());
    }

    half  * d_M    = nullptr;
    float * d_bout = nullptr;
    if (cudaMalloc(&d_M, (size_t) V_full * R * sizeof(half)) != cudaSuccess) {
        return nullptr;
    }
    if (cudaMalloc(&d_bout, (size_t) V_full * sizeof(float)) != cudaSuccess) {
        cudaFree(d_M);
        return nullptr;
    }

    constexpr int CHUNK_V = 4096;
    float * d_H_f32   = nullptr;
    float * d_W_chunk = nullptr;
    float * d_M_chunk = nullptr;
    const size_t n_H = (size_t) K * R;
    const bool h_is_f32 = (w_ho->type == GGML_TYPE_F32);
    if ((!h_is_f32 && cudaMalloc(&d_H_f32, n_H * sizeof(float)) != cudaSuccess) ||
            cudaMalloc(&d_W_chunk, (size_t) CHUNK_V * K * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&d_M_chunk, (size_t) CHUNK_V * R * sizeof(float)) != cudaSuccess) {
        cudaFree(d_M_chunk);
        cudaFree(d_W_chunk);
        cudaFree(d_H_f32);
        cudaFree(d_bout);
        cudaFree(d_M);
        return nullptr;
    }

    const float * H_ptr = nullptr;
    if (h_is_f32) {
        H_ptr = (const float *) w_ho->data;
    } else if (w_ho->type == GGML_TYPE_BF16) {
        inkling_bf16_to_f32_elem<<<(unsigned) ((n_H + 255) / 256), 256, 0, stream>>>(
            (const nv_bfloat16 *) w_ho->data, d_H_f32, n_H);
        H_ptr = d_H_f32;
    } else {
        inkling_f16_to_f32_elem<<<(unsigned) ((n_H + 255) / 256), 256, 0, stream>>>(
            (const half *) w_ho->data, d_H_f32, n_H);
        H_ptr = d_H_f32;
    }

    cublasHandle_t cublas_h = ctx.cublas_handle();
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    const int nb_row = K >> 5;

    for (int v0 = 0; v0 < V_full; v0 += CHUNK_V) {
        const int vc = (v0 + CHUNK_V <= V_full) ? CHUNK_V : (V_full - v0);
        const size_t n_w = (size_t) vc * K;
        inkling_dequant_q8_0_rows_f32<<<(unsigned) ((n_w + 255) / 256), 256, 0, stream>>>(
            (const block_q8_0 *) w_out->data + (size_t) v0 * nb_row, d_W_chunk, K, vc);

        // M_chunk[R x vc] (col-major) = H[R x K] * W_chunk[K x vc] -> row-major [vc x R]
        CUBLAS_CHECK(cublasGemmEx(
            cublas_h, CUBLAS_OP_N, CUBLAS_OP_N,
            R, vc, K,
            &alpha,
            H_ptr,     CUDA_R_32F, R,
            d_W_chunk, CUDA_R_32F, K,
            &beta,
            d_M_chunk, CUDA_R_32F, R,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

        const size_t n_m = (size_t) vc * R;
        inkling_f32_to_f16_elem<<<(unsigned) ((n_m + 255) / 256), 256, 0, stream>>>(
            d_M_chunk, d_M + (size_t) v0 * R, n_m);

        // b_out[vc x 1] = W_chunk^T[vc x K] * b_ho[K x 1]
        CUBLAS_CHECK(cublasGemmEx(
            cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
            vc, 1, K,
            &alpha,
            d_W_chunk,  CUDA_R_32F, K,
            b_ho->data, CUDA_R_32F, K,
            &beta,
            d_bout + v0, CUDA_R_32F, vc,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaFree(d_M_chunk);
    cudaFree(d_W_chunk);
    cudaFree(d_H_f32);

    s_cache.push_back({ dev, w_out->data, w_ho->data, b_ho->data, V_full, K, R, d_M, d_bout });
    return &s_cache.back();
}

template <int NWARPS>
static __global__ void inkling_dspark_lowrank_head_f16_r256_b1(
        const half * __restrict__ M,
        const float * __restrict__ b_out,
        const float * __restrict__ latent,
        const float * __restrict__ base,
        float * __restrict__ dst,
        const int V_out,
        const float inv_cap,
        const float cap) {
    __shared__ float s_lat[256];
    const int tid_blk = threadIdx.y * 32 + threadIdx.x;
    for (int i = tid_blk; i < 256; i += NWARPS * 32) {
        s_lat[i] = latent[i];
    }
    __syncthreads();

    const int row = blockIdx.x * NWARPS + threadIdx.y;
    if (row >= V_out) {
        return;
    }
    const int lane = threadIdx.x;
    const uint4 v4 = ((const uint4 *) (M + (size_t) row * 256))[lane];
    const half2 h0 = *reinterpret_cast<const half2 *>(&v4.x);
    const half2 h1 = *reinterpret_cast<const half2 *>(&v4.y);
    const half2 h2 = *reinterpret_cast<const half2 *>(&v4.z);
    const half2 h3 = *reinterpret_cast<const half2 *>(&v4.w);
    const float2 f0 = __half22float2(h0);
    const float2 f1 = __half22float2(h1);
    const float2 f2 = __half22float2(h2);
    const float2 f3 = __half22float2(h3);
    const int c = lane * 8;
    float acc = f0.x * s_lat[c + 0] + f0.y * s_lat[c + 1]
              + f1.x * s_lat[c + 2] + f1.y * s_lat[c + 3]
              + f2.x * s_lat[c + 4] + f2.y * s_lat[c + 5]
              + f3.x * s_lat[c + 6] + f3.y * s_lat[c + 7];
    acc = warp_reduce_sum<32>(acc);
    if (lane == 0) {
        const float x = base[row] + b_out[row] + acc;
        dst[row] = cap * tanhf(x * inv_cap);
    }
}

template <int NWARPS, int NB>
static __global__ void inkling_dspark_lowrank_head_f16_r256_nb(
        const half * __restrict__ M,
        const float * __restrict__ b_out,
        const float * __restrict__ latent,
        const float * __restrict__ base,
        float * __restrict__ dst,
        const int V_out,
        const int base_stride,
        const float inv_cap,
        const float cap) {
    __shared__ float s_lat[NB][256];
    const int tid_blk = threadIdx.y * 32 + threadIdx.x;
    for (int i = tid_blk; i < NB * 256; i += NWARPS * 32) {
        s_lat[i >> 8][i & 255] = latent[i];
    }
    __syncthreads();

    const int row = blockIdx.x * NWARPS + threadIdx.y;
    if (row >= V_out) {
        return;
    }
    const int lane = threadIdx.x;
    const uint4 v4 = ((const uint4 *) (M + (size_t) row * 256))[lane];
    const half2 h0 = *reinterpret_cast<const half2 *>(&v4.x);
    const half2 h1 = *reinterpret_cast<const half2 *>(&v4.y);
    const half2 h2 = *reinterpret_cast<const half2 *>(&v4.z);
    const half2 h3 = *reinterpret_cast<const half2 *>(&v4.w);
    const float2 f0 = __half22float2(h0);
    const float2 f1 = __half22float2(h1);
    const float2 f2 = __half22float2(h2);
    const float2 f3 = __half22float2(h3);
    const int c = lane * 8;
    const float bo = (lane == 0) ? b_out[row] : 0.0f;
#pragma unroll
    for (int b = 0; b < NB; ++b) {
        float acc = f0.x * s_lat[b][c + 0] + f0.y * s_lat[b][c + 1]
                  + f1.x * s_lat[b][c + 2] + f1.y * s_lat[b][c + 3]
                  + f2.x * s_lat[b][c + 4] + f2.y * s_lat[b][c + 5]
                  + f3.x * s_lat[b][c + 6] + f3.y * s_lat[b][c + 7];
        acc = warp_reduce_sum<32>(acc);
        if (lane == 0) {
            const float x = base[(size_t) b * base_stride + row] + bo + acc;
            dst[(size_t) b * V_out + row] = cap * tanhf(x * inv_cap);
        }
    }
}

static inline bool ggml_cuda_op_inkling_dspark_lowrank_head(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * w_out,
        const ggml_tensor * w_ho,
        const ggml_tensor * b_ho,
        const ggml_tensor * latent,
        const ggml_tensor * base,
        ggml_tensor * dst,
        const float inv_cap,
        const float cap) {
    const inkling_dspark_lr_entry * entry = inkling_dspark_lr_get_or_build(ctx, w_out, w_ho, b_ho);
    if (!entry) {
        return false;
    }
    const int V_out = (int) dst->ne[0];
    const int nb    = (int) dst->ne[1];
    const int base_stride = (int) (base->nb[1] / sizeof(float));
    cudaStream_t stream = ctx.stream();
    constexpr int NWARPS = 4;
    const dim3 grid((V_out + NWARPS - 1) / NWARPS, 1, 1);
    const dim3 block(32, NWARPS, 1);

    switch (nb) {
        case 1:
            inkling_dspark_lowrank_head_f16_r256_b1<NWARPS><<<grid, block, 0, stream>>>(
                entry->d_M, entry->d_bout, (const float *) latent->data, (const float *) base->data,
                (float *) dst->data, V_out, inv_cap, cap);
            return true;
        case 2:
            inkling_dspark_lowrank_head_f16_r256_nb<NWARPS, 2><<<grid, block, 0, stream>>>(
                entry->d_M, entry->d_bout, (const float *) latent->data, (const float *) base->data,
                (float *) dst->data, V_out, base_stride, inv_cap, cap);
            return true;
        case 3:
            inkling_dspark_lowrank_head_f16_r256_nb<NWARPS, 3><<<grid, block, 0, stream>>>(
                entry->d_M, entry->d_bout, (const float *) latent->data, (const float *) base->data,
                (float *) dst->data, V_out, base_stride, inv_cap, cap);
            return true;
        case 4:
            inkling_dspark_lowrank_head_f16_r256_nb<NWARPS, 4><<<grid, block, 0, stream>>>(
                entry->d_M, entry->d_bout, (const float *) latent->data, (const float *) base->data,
                (float *) dst->data, V_out, base_stride, inv_cap, cap);
            return true;
        case 5:
            inkling_dspark_lowrank_head_f16_r256_nb<NWARPS, 5><<<grid, block, 0, stream>>>(
                entry->d_M, entry->d_bout, (const float *) latent->data, (const float *) base->data,
                (float *) dst->data, V_out, base_stride, inv_cap, cap);
            return true;
        case 6:
            inkling_dspark_lowrank_head_f16_r256_nb<NWARPS, 6><<<grid, block, 0, stream>>>(
                entry->d_M, entry->d_bout, (const float *) latent->data, (const float *) base->data,
                (float *) dst->data, V_out, base_stride, inv_cap, cap);
            return true;
        case 7:
            inkling_dspark_lowrank_head_f16_r256_nb<NWARPS, 7><<<grid, block, 0, stream>>>(
                entry->d_M, entry->d_bout, (const float *) latent->data, (const float *) base->data,
                (float *) dst->data, V_out, base_stride, inv_cap, cap);
            return true;
        case 8:
            inkling_dspark_lowrank_head_f16_r256_nb<NWARPS, 8><<<grid, block, 0, stream>>>(
                entry->d_M, entry->d_bout, (const float *) latent->data, (const float *) base->data,
                (float *) dst->data, V_out, base_stride, inv_cap, cap);
            return true;
        default:
            return false;
    }
}

// ============================================================================
// Wave 9A + 9B: Zero-Table On-The-Fly RelPos Flash Attention (F16 & Q8_0 KV)
//   Eliminates the 12 GiB F32 kq_b + 3 GiB F16 mask + 256 MiB self_rel_pos + 512 MiB Q8->F16 KV buffer.
//   Three SM 8.6 specialized paths:
//     1) T == 1 (decode):       8-warp 0-SMEM 0-barrier 2x-unrolled register-streaming + coalesced Q8_0/F16 + grid(8,splits)
//     2) T == 2..8 (spec):      3*T warps/blk coalesced Q8_0/F16 + dual s_K/s_V (2 barriers/tile) + grid(8,splits)
//     3) T > 8 or S > 1 (prefill): SM 8.6 Tensor Core m16n8k16 MMA + coalesced mask cache s_M + grid(24,q_blks,S) + coalesced Q8_0
// ============================================================================

struct inkling_q8_v4 {
    uint32_t q4;
    float    d;
};

// Zero-SHFL, branchless coalesced L1-cached Q8_0 4-element loader (136 B / 128-dim head = 4 x block_q8_0)
static __device__ __forceinline__ inkling_q8_v4 inkling_ldg_q8_4(
        const char * __restrict__ row_bytes,
        const int lane) {
    const int blk  = lane >> 3;
    const int sub  = lane & 7;
    const block_q8_0 * b = (const block_q8_0 *) (row_bytes + blk * 34);
    const uint32_t * row_u32 = (const uint32_t *) row_bytes;
    const int w_off = (blk == 0) ? sub : ((blk == 1) ? (9 + sub) : ((blk == 2) ? (17 + sub) : (26 + sub)));
    const uint32_t lo = __ldg(row_u32 + w_off);
    const uint32_t hi = (blk & 1) ? 0u : __ldg(row_u32 + w_off + 1);
    inkling_q8_v4 res;
    res.q4 = (blk & 1) ? lo : __funnelshift_r(lo, hi, 16);
    res.d  = __half2float(__ldg(&b->d));
    return res;
}

// Bit-exact zero-I2F int8x4 -> float4 via PRMT + FP32 magic bias (bypasses SM 8.6 XU/MIO pipe!)
static __device__ __forceinline__ float4 inkling_i8x4_to_f32x4(const uint32_t q4) {
    const uint32_t u4 = q4 ^ 0x80808080u;
    return make_float4(
        __uint_as_float(__byte_perm(u4, 0x4B000000u, 0x7650)) - 8388736.0f,
        __uint_as_float(__byte_perm(u4, 0x4B000000u, 0x7651)) - 8388736.0f,
        __uint_as_float(__byte_perm(u4, 0x4B000000u, 0x7652)) - 8388736.0f,
        __uint_as_float(__byte_perm(u4, 0x4B000000u, 0x7653)) - 8388736.0f);
}

static __device__ __forceinline__ float4 inkling_load_q8_128_warp(
        const char * __restrict__ row_bytes,
        const int lane) {
    const inkling_q8_v4 r = inkling_ldg_q8_4(row_bytes, lane);
    const float4 f = inkling_i8x4_to_f32x4(r.q4);
    return make_float4(f.x * r.d, f.y * r.d, f.z * r.d, f.w * r.d);
}

static __device__ __forceinline__ void inkling_quant_q32_lane(
        const float4 q,
        uint32_t & q_pack,
        float & q_d) {
    float amax = fmaxf(fmaxf(fabsf(q.x), fabsf(q.y)), fmaxf(fabsf(q.z), fabsf(q.w)));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1, 8));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2, 8));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 4, 8));
    const float d  = amax / 127.0f;
    const float id = (d != 0.0f) ? (1.0f / d) : 0.0f;
    const uint8_t b0 = (uint8_t) (int8_t) fmaxf(-127.0f, fminf(127.0f, roundf(q.x * id)));
    const uint8_t b1 = (uint8_t) (int8_t) fmaxf(-127.0f, fminf(127.0f, roundf(q.y * id)));
    const uint8_t b2 = (uint8_t) (int8_t) fmaxf(-127.0f, fminf(127.0f, roundf(q.z * id)));
    const uint8_t b3 = (uint8_t) (int8_t) fmaxf(-127.0f, fminf(127.0f, roundf(q.w * id)));
    q_pack = ((uint32_t) b0) | (((uint32_t) b1) << 8) | (((uint32_t) b2) << 16) | (((uint32_t) b3) << 24);
    q_d = d;
}

static __device__ __forceinline__ float4 inkling_unpack_f16_128_warp(const uint2 vv) {
    const float2 f0 = __half22float2(*reinterpret_cast<const half2 *>(&vv.x));
    const float2 f1 = __half22float2(*reinterpret_cast<const half2 *>(&vv.y));
    return make_float4(f0.x, f0.y, f1.x, f1.y);
}

static __device__ __forceinline__ uint2 inkling_pack_f32x4_to_f16x4(const float4 v) {
    const half2 h0 = __floats2half2_rn(v.x, v.y);
    const half2 h1 = __floats2half2_rn(v.z, v.w);
    uint2 out;
    out.x = *reinterpret_cast<const uint32_t *>(&h0);
    out.y = *reinterpret_cast<const uint32_t *>(&h1);
    return out;
}

static __device__ __forceinline__ float inkling_f16_smem_dot16(
        const uint4 * row_u4,
        const int klane,
        const float4 q0,
        const float4 q1,
        const float4 q2,
        const float4 q3) {
    const uint4 k0 = row_u4[klane * 2 + 0];
    const uint4 k1 = row_u4[klane * 2 + 1];
    const float2 kf0 = __half22float2(*reinterpret_cast<const half2 *>(&k0.x));
    const float2 kf1 = __half22float2(*reinterpret_cast<const half2 *>(&k0.y));
    const float2 kf2 = __half22float2(*reinterpret_cast<const half2 *>(&k0.z));
    const float2 kf3 = __half22float2(*reinterpret_cast<const half2 *>(&k0.w));
    const float2 kf4 = __half22float2(*reinterpret_cast<const half2 *>(&k1.x));
    const float2 kf5 = __half22float2(*reinterpret_cast<const half2 *>(&k1.y));
    const float2 kf6 = __half22float2(*reinterpret_cast<const half2 *>(&k1.z));
    const float2 kf7 = __half22float2(*reinterpret_cast<const half2 *>(&k1.w));
    return q0.x*kf0.x + q0.y*kf0.y + q0.z*kf1.x + q0.w*kf1.y
         + q1.x*kf2.x + q1.y*kf2.y + q1.z*kf3.x + q1.w*kf3.y
         + q2.x*kf4.x + q2.y*kf4.y + q2.z*kf5.x + q2.w*kf5.y
         + q3.x*kf6.x + q3.y*kf6.y + q3.z*kf7.x + q3.w*kf7.y;
}

static __device__ __forceinline__ void inkling_f16_smem_v4_acc(
        const uint4 * row_u4,
        const int lane,
        const float kw,
        float4 & o) {
    const uint2 vv = ((const uint2 *) row_u4)[lane];
    const float2 vf0 = __half22float2(*reinterpret_cast<const half2 *>(&vv.x));
    const float2 vf1 = __half22float2(*reinterpret_cast<const half2 *>(&vv.y));
    o.x += kw * vf0.x; o.y += kw * vf0.y; o.z += kw * vf1.x; o.w += kw * vf1.y;
}

// 9A/9B Kernel 1: 8-Warp Register-Streaming Single-Token Decode (T == 1, S == 1)
//   0 SMEM & 0 __syncthreads() in KV loop, dp4a for Q8_0 K, zero-I2F PRMT for Q8_0 V, zero-SHFL __ldg, 4 blocks/SM!
template <bool K_Q8, bool V_Q8, bool MASK_F16, bool FUSE_WO_Q8, int MAX_SPLITS>
__launch_bounds__(256, 4)
static __global__ void inkling_rel_pos_fattn_t1_kpos(
        const float   * __restrict__ Q,
        const char    * __restrict__ K_bytes,
        const char    * __restrict__ V_bytes,
        const float   * __restrict__ R,
        const float   * __restrict__ rel_bias,
        const float   * __restrict__ q_sinks,
        const int32_t * __restrict__ k_pos,
        const int32_t * __restrict__ q_pos,
        const void    * __restrict__ raw_mask,
        const float   * __restrict__ Gcur,      // non-null when FUSE_WO_Q8
        block_q8_1    * __restrict__ dst_q8,    // non-null when FUSE_WO_Q8
        float         * __restrict__ dst_f32,   // non-null when !FUSE_WO_Q8
        float         * __restrict__ scratch,   // [8][MAX_SPLITS][3][132]
        int32_t       * __restrict__ cnt,       // [8]
        const int n_kv,
        const int n_extent,
        const size_t q_head_stride_f,
        const size_t k_head_stride_b,
        const size_t k_tok_stride_b,
        const size_t v_head_stride_b,
        const size_t v_tok_stride_b,
        const float kq_scale,
        const int n_splits) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int NWARPS = 8;
    const int kvh  = blockIdx.x; // 0..7 (contiguous in grid.x -> all 8 KV heads hit L2 simultaneously!)
    const int sp   = blockIdx.y; // 0..n_splits-1
    const int lane = threadIdx.x; // 0..31
    const int warp = threadIdx.y; // 0..7
    const int qh0  = kvh * 3;

    float4 q0 = __ldg((const float4 *) (Q + (size_t) (qh0 + 0) * q_head_stride_f) + lane);
    float4 q1 = __ldg((const float4 *) (Q + (size_t) (qh0 + 1) * q_head_stride_f) + lane);
    float4 q2 = __ldg((const float4 *) (Q + (size_t) (qh0 + 2) * q_head_stride_f) + lane);
    q0.x *= kq_scale; q0.y *= kq_scale; q0.z *= kq_scale; q0.w *= kq_scale;
    q1.x *= kq_scale; q1.y *= kq_scale; q1.z *= kq_scale; q1.w *= kq_scale;
    q2.x *= kq_scale; q2.y *= kq_scale; q2.z *= kq_scale; q2.w *= kq_scale;

    // For Q8_0 K, pre-quantize Q per 32-dim block once into 3 int32 regs + 3 float scales for fast dp4a!
    uint32_t q0_i8 = 0u, q1_i8 = 0u, q2_i8 = 0u;
    float qd0 = 0.0f, qd1 = 0.0f, qd2 = 0.0f;
    if constexpr (K_Q8) {
        inkling_quant_q32_lane(q0, q0_i8, qd0);
        inkling_quant_q32_lane(q1, q1_i8, qd1);
        inkling_quant_q32_lane(q2, q2_i8, qd2);
    }

    // R has 16 dims -> lanes 0..15 hold 1 float per GQA head (3 regs total!)
    const float r0 = (lane < 16) ? __ldg(&R[(size_t) (qh0 + 0) * 16 + lane]) : 0.0f;
    const float r1 = (lane < 16) ? __ldg(&R[(size_t) (qh0 + 1) * 16 + lane]) : 0.0f;
    const float r2 = (lane < 16) ? __ldg(&R[(size_t) (qh0 + 2) * 16 + lane]) : 0.0f;
    const int qp0  = __ldg(&q_pos[0]);

    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float m0 = -1e30f, m1 = -1e30f, m2 = -1e30f;
    float l0 = 0.0f,   l1 = 0.0f,   l2 = 0.0f;

    const int chunk = (n_kv + n_splits - 1) / n_splits;
    const int j_beg = sp * chunk;
    const int j_end = min(n_kv, j_beg + chunk);

    const char * K_base = K_bytes + (size_t) kvh * k_head_stride_b;
    const char * V_base = V_bytes + (size_t) kvh * v_head_stride_b;

    auto read_mask_1 = [&](int j) -> float {
        if constexpr (MASK_F16) {
            return __half2float(__ldg(&((const half *) raw_mask)[j]));
        } else {
            return __ldg(&((const float *) raw_mask)[j]);
        }
    };

    auto step_one = [&](int j, float mv) {
        const int kp    = __ldg(&k_pos[j]);
        const int d_rel = (kp >= 0) ? (qp0 - kp) : -1;
        const char * kptr = K_base + (size_t) j * k_tok_stride_b;
        const char * vptr = V_base + (size_t) j * v_tok_stride_b;

        inkling_q8_v4 kq{}, vq{};
        uint2 kf{}, vf{};
        if constexpr (K_Q8) { kq = inkling_ldg_q8_4(kptr, lane); }
        else                { kf = __ldg((const uint2 *) kptr + lane); }
        if constexpr (V_Q8) { vq = inkling_ldg_q8_4(vptr, lane); }
        else                { vf = __ldg((const uint2 *) vptr + lane); }

        float a0, a1, a2;
        if constexpr (K_Q8) {
            a0 = (float) ggml_cuda_dp4a((int) q0_i8, (int) kq.q4, 0) * (kq.d * qd0);
            a1 = (float) ggml_cuda_dp4a((int) q1_i8, (int) kq.q4, 0) * (kq.d * qd1);
            a2 = (float) ggml_cuda_dp4a((int) q2_i8, (int) kq.q4, 0) * (kq.d * qd2);
        } else {
            const float4 kv = inkling_unpack_f16_128_warp(kf);
            a0 = q0.x*kv.x + q0.y*kv.y + q0.z*kv.z + q0.w*kv.w;
            a1 = q1.x*kv.x + q1.y*kv.y + q1.z*kv.z + q1.w*kv.w;
            a2 = q2.x*kv.x + q2.y*kv.y + q2.z*kv.z + q2.w*kv.w;
        }
        if (d_rel >= 0 && d_rel < n_extent) {
            if (lane < 16) {
                const float rb = __ldg(&rel_bias[(size_t) d_rel * 16 + lane]);
                a0 += r0 * rb;
                a1 += r1 * rb;
                a2 += r2 * rb;
            }
        }
        const float s0 = warp_reduce_sum<32>(a0) + mv;
        const float s1 = warp_reduce_sum<32>(a1) + mv;
        const float s2 = warp_reduce_sum<32>(a2) + mv;

        const float nm0  = fmaxf(m0, s0);
        const float rsc0 = (m0 > -1e20f) ? expf(m0 - nm0) : 0.0f;
        const float w0   = expf(s0 - nm0);
        m0 = nm0; l0 = l0 * rsc0 + w0;

        const float nm1  = fmaxf(m1, s1);
        const float rsc1 = (m1 > -1e20f) ? expf(m1 - nm1) : 0.0f;
        const float w1   = expf(s1 - nm1);
        m1 = nm1; l1 = l1 * rsc1 + w1;

        const float nm2  = fmaxf(m2, s2);
        const float rsc2 = (m2 > -1e20f) ? expf(m2 - nm2) : 0.0f;
        const float w2   = expf(s2 - nm2);
        m2 = nm2; l2 = l2 * rsc2 + w2;

        const float4 vv = V_Q8 ? inkling_i8x4_to_f32x4(vq.q4) : inkling_unpack_f16_128_warp(vf);
        const float w0v = V_Q8 ? (w0 * vq.d) : w0;
        const float w1v = V_Q8 ? (w1 * vq.d) : w1;
        const float w2v = V_Q8 ? (w2 * vq.d) : w2;
        o0.x = o0.x * rsc0 + w0v * vv.x; o0.y = o0.y * rsc0 + w0v * vv.y;
        o0.z = o0.z * rsc0 + w0v * vv.z; o0.w = o0.w * rsc0 + w0v * vv.w;
        o1.x = o1.x * rsc1 + w1v * vv.x; o1.y = o1.y * rsc1 + w1v * vv.y;
        o1.z = o1.z * rsc1 + w1v * vv.z; o1.w = o1.w * rsc1 + w1v * vv.w;
        o2.x = o2.x * rsc2 + w2v * vv.x; o2.y = o2.y * rsc2 + w2v * vv.y;
        o2.z = o2.z * rsc2 + w2v * vv.z; o2.w = o2.w * rsc2 + w2v * vv.w;
    };

    // Main 2x-unrolled register-streaming loop: ZERO SMEM, ZERO __syncthreads(), 0 SHFL loads, 0 V-I2F!
    for (int j = j_beg + warp; j < j_end; j += 2 * NWARPS) {
        const int ja = j;
        const int jb = j + NWARPS;
        const float mva = read_mask_1(ja);
        const bool  aca = (mva > -1e20f);
        const float mvb = (jb < j_end) ? read_mask_1(jb) : -INFINITY;
        const bool  acb = (mvb > -1e20f);

        if (aca && acb) {
            const int kpa = __ldg(&k_pos[ja]);
            const int kpb = __ldg(&k_pos[jb]);
            const int da  = (kpa >= 0) ? (qp0 - kpa) : -1;
            const int db  = (kpb >= 0) ? (qp0 - kpb) : -1;
            const char * kap = K_base + (size_t) ja * k_tok_stride_b;
            const char * kbp = K_base + (size_t) jb * k_tok_stride_b;
            const char * vap = V_base + (size_t) ja * v_tok_stride_b;
            const char * vbp = V_base + (size_t) jb * v_tok_stride_b;

            inkling_q8_v4 kqa{}, kqb{}, vqa{}, vqb{};
            uint2 kfa{}, kfb{}, vfa{}, vfb{};
            if constexpr (K_Q8) {
                kqa = inkling_ldg_q8_4(kap, lane);
                kqb = inkling_ldg_q8_4(kbp, lane);
            } else {
                kfa = __ldg((const uint2 *) kap + lane);
                kfb = __ldg((const uint2 *) kbp + lane);
            }
            if constexpr (V_Q8) {
                vqa = inkling_ldg_q8_4(vap, lane);
                vqb = inkling_ldg_q8_4(vbp, lane);
            } else {
                vfa = __ldg((const uint2 *) vap + lane);
                vfb = __ldg((const uint2 *) vbp + lane);
            }

            float a0, a1, a2, b0, b1, b2;
            if constexpr (K_Q8) {
                a0 = (float) ggml_cuda_dp4a((int) q0_i8, (int) kqa.q4, 0) * (kqa.d * qd0);
                a1 = (float) ggml_cuda_dp4a((int) q1_i8, (int) kqa.q4, 0) * (kqa.d * qd1);
                a2 = (float) ggml_cuda_dp4a((int) q2_i8, (int) kqa.q4, 0) * (kqa.d * qd2);
                b0 = (float) ggml_cuda_dp4a((int) q0_i8, (int) kqb.q4, 0) * (kqb.d * qd0);
                b1 = (float) ggml_cuda_dp4a((int) q1_i8, (int) kqb.q4, 0) * (kqb.d * qd1);
                b2 = (float) ggml_cuda_dp4a((int) q2_i8, (int) kqb.q4, 0) * (kqb.d * qd2);
            } else {
                const float4 kva = inkling_unpack_f16_128_warp(kfa);
                const float4 kvb = inkling_unpack_f16_128_warp(kfb);
                a0 = q0.x*kva.x + q0.y*kva.y + q0.z*kva.z + q0.w*kva.w;
                a1 = q1.x*kva.x + q1.y*kva.y + q1.z*kva.z + q1.w*kva.w;
                a2 = q2.x*kva.x + q2.y*kva.y + q2.z*kva.z + q2.w*kva.w;
                b0 = q0.x*kvb.x + q0.y*kvb.y + q0.z*kvb.z + q0.w*kvb.w;
                b1 = q1.x*kvb.x + q1.y*kvb.y + q1.z*kvb.z + q1.w*kvb.w;
                b2 = q2.x*kvb.x + q2.y*kvb.y + q2.z*kvb.z + q2.w*kvb.w;
            }

            if ((da >= 0 && da < n_extent) || (db >= 0 && db < n_extent)) {
                if (lane < 16) {
                    if (da >= 0 && da < n_extent) {
                        const float rb = __ldg(&rel_bias[(size_t) da * 16 + lane]);
                        a0 += r0 * rb; a1 += r1 * rb; a2 += r2 * rb;
                    }
                    if (db >= 0 && db < n_extent) {
                        const float rb = __ldg(&rel_bias[(size_t) db * 16 + lane]);
                        b0 += r0 * rb; b1 += r1 * rb; b2 += r2 * rb;
                    }
                }
            }

            const float s0a = warp_reduce_sum<32>(a0) + mva;
            const float s1a = warp_reduce_sum<32>(a1) + mva;
            const float s2a = warp_reduce_sum<32>(a2) + mva;
            const float s0b = warp_reduce_sum<32>(b0) + mvb;
            const float s1b = warp_reduce_sum<32>(b1) + mvb;
            const float s2b = warp_reduce_sum<32>(b2) + mvb;

            const float nm0  = fmaxf(m0, fmaxf(s0a, s0b));
            const float rsc0 = (m0 > -1e20f) ? expf(m0 - nm0) : 0.0f;
            const float w0a  = expf(s0a - nm0);
            const float w0b  = expf(s0b - nm0);
            m0 = nm0; l0 = l0 * rsc0 + w0a + w0b;

            const float nm1  = fmaxf(m1, fmaxf(s1a, s1b));
            const float rsc1 = (m1 > -1e20f) ? expf(m1 - nm1) : 0.0f;
            const float w1a  = expf(s1a - nm1);
            const float w1b  = expf(s1b - nm1);
            m1 = nm1; l1 = l1 * rsc1 + w1a + w1b;

            const float nm2  = fmaxf(m2, fmaxf(s2a, s2b));
            const float rsc2 = (m2 > -1e20f) ? expf(m2 - nm2) : 0.0f;
            const float w2a  = expf(s2a - nm2);
            const float w2b  = expf(s2b - nm2);
            m2 = nm2; l2 = l2 * rsc2 + w2a + w2b;

            const float4 vva = V_Q8 ? inkling_i8x4_to_f32x4(vqa.q4) : inkling_unpack_f16_128_warp(vfa);
            const float4 vvb = V_Q8 ? inkling_i8x4_to_f32x4(vqb.q4) : inkling_unpack_f16_128_warp(vfb);
            const float w0av = V_Q8 ? (w0a * vqa.d) : w0a, w0bv = V_Q8 ? (w0b * vqb.d) : w0b;
            const float w1av = V_Q8 ? (w1a * vqa.d) : w1a, w1bv = V_Q8 ? (w1b * vqb.d) : w1b;
            const float w2av = V_Q8 ? (w2a * vqa.d) : w2a, w2bv = V_Q8 ? (w2b * vqb.d) : w2b;
            o0.x = o0.x * rsc0 + w0av * vva.x + w0bv * vvb.x;
            o0.y = o0.y * rsc0 + w0av * vva.y + w0bv * vvb.y;
            o0.z = o0.z * rsc0 + w0av * vva.z + w0bv * vvb.z;
            o0.w = o0.w * rsc0 + w0av * vva.w + w0bv * vvb.w;
            o1.x = o1.x * rsc1 + w1av * vva.x + w1bv * vvb.x;
            o1.y = o1.y * rsc1 + w1av * vva.y + w1bv * vvb.y;
            o1.z = o1.z * rsc1 + w1av * vva.z + w1bv * vvb.z;
            o1.w = o1.w * rsc1 + w1av * vva.w + w1bv * vvb.w;
            o2.x = o2.x * rsc2 + w2av * vva.x + w2bv * vvb.x;
            o2.y = o2.y * rsc2 + w2av * vva.y + w2bv * vvb.y;
            o2.z = o2.z * rsc2 + w2av * vva.z + w2bv * vvb.z;
            o2.w = o2.w * rsc2 + w2av * vva.w + w2bv * vvb.w;
        } else if (aca) {
            step_one(ja, mva);
        } else if (acb) {
            step_one(jb, mvb);
        }
    }

    // In-block 8-warp reduction (1 single __syncthreads() for the entire 128K context!)
    __shared__ __align__(16) float4 s_O[NWARPS][3][32]; // 12,288 B
    __shared__ float s_m[NWARPS][3];
    __shared__ float s_l[NWARPS][3];

    s_O[warp][0][lane] = o0;
    s_O[warp][1][lane] = o1;
    s_O[warp][2][lane] = o2;
    if (lane == 0) {
        s_m[warp][0] = m0; s_m[warp][1] = m1; s_m[warp][2] = m2;
        s_l[warp][0] = l0; s_l[warp][1] = l1; s_l[warp][2] = l2;
    }
    __syncthreads();

    // Warps 0, 1, 2 each reduce one GQA head ql = warp across the 8 warps
    float  bm = -1e30f;
    float  bl = 0.0f;
    float4 bo = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (warp < 3) {
        const int ql = warp;
#pragma unroll
        for (int w = 0; w < NWARPS; ++w) {
            if (s_l[w][ql] > 0.0f) {
                bm = fmaxf(bm, s_m[w][ql]);
            }
        }
#pragma unroll
        for (int w = 0; w < NWARPS; ++w) {
            const float wl = s_l[w][ql];
            if (wl > 0.0f) {
                const float es = expf(s_m[w][ql] - bm);
                bl += wl * es;
                const float4 sv = s_O[w][ql][lane];
                bo.x += sv.x * es; bo.y += sv.y * es;
                bo.z += sv.z * es; bo.w += sv.w * es;
            }
        }
    }

    auto finalize_head = [&](int ql, float fm, float fl, float4 acc) {
        const int qh = qh0 + ql;
        const float sink = __ldg(&q_sinks[qh]);
        float inv_l = 0.0f;
        if (fl > 0.0f) {
            const float nm     = fmaxf(fm, sink);
            const float e_old  = expf(fm - nm);
            const float e_sink = expf(sink - nm);
            inv_l = e_old / (fl * e_old + e_sink);
        }
        if constexpr (FUSE_WO_Q8) {
            float * fo = (float *) &s_O[0][ql][0];
            ((float4 *) fo)[lane] = acc;
            __syncwarp();
#pragma unroll
            for (int blk = 0; blk < 4; ++blk) {
                const int c      = blk * 32 + lane;
                const float val  = fo[c] * inv_l;
                const float g    = Gcur[(size_t) qh * 128 + c];
                const float xi   = val * (1.0f / (1.0f + expf(-g)));
                const float amax = warp_reduce_max<32>(fabsf(xi));
                const float d    = amax / 127.0f;
                const int8_t q   = (amax == 0.0f) ? 0 : (int8_t) fmaxf(-127.0f, fminf(127.0f, roundf(xi / d)));
                const int ib     = qh * 4 + blk;
                dst_q8[ib].qs[lane] = q;
                if (lane == 0) {
                    dst_q8[ib].ds = make_half2(__float2half(d), __float2half(0.0f));
                }
            }
        } else {
            float4 res;
            res.x = acc.x * inv_l; res.y = acc.y * inv_l;
            res.z = acc.z * inv_l; res.w = acc.w * inv_l;
            ((float4 *) (dst_f32 + (size_t) qh * 128))[lane] = res;
        }
    };

    if (n_splits == 1) {
        if constexpr (FUSE_WO_Q8) {
            __syncthreads(); // ensure all 3 warps finished reading s_O before reusing s_O[0][ql]
        }
        if (warp < 3) {
            finalize_head(warp, bm, bl, bo);
        }
        return;
    }

    if (warp < 3) {
        float * sp_ptr = scratch + ((size_t) (kvh * MAX_SPLITS + sp) * 3 + warp) * 132;
        if (bl > 0.0f) {
            ((float4 *) sp_ptr)[lane] = bo;
        }
        if (lane == 0) {
            sp_ptr[128] = bm;
            sp_ptr[129] = bl;
        }
    }
    __syncthreads();
    __shared__ bool is_last;
    if (warp == 0 && lane == 0) {
        __threadfence();
        const int done = atomicAdd(&cnt[kvh], 1) + 1;
        is_last = (done == n_splits);
        if (is_last) {
            cnt[kvh] = 0;
        }
    }
    __syncthreads();
    if (!is_last || warp >= 3) {
        return;
    }

    float fm = -1e30f;
    for (int s = 0; s < n_splits; ++s) {
        const float * sp_p = scratch + ((size_t) (kvh * MAX_SPLITS + s) * 3 + warp) * 132;
        if (sp_p[129] > 0.0f) {
            fm = fmaxf(fm, sp_p[128]);
        }
    }
    float fl = 0.0f;
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int s = 0; s < n_splits; ++s) {
        const float * sp_p = scratch + ((size_t) (kvh * MAX_SPLITS + s) * 3 + warp) * 132;
        const float sl = sp_p[129];
        if (sl == 0.0f) {
            continue;
        }
        const float es = expf(sp_p[128] - fm);
        fl += sl * es;
        const float4 sv = ((const float4 *) sp_p)[lane];
        acc.x += sv.x * es; acc.y += sv.y * es;
        acc.z += sv.z * es; acc.w += sv.w * es;
    }
    finalize_head(warp, fm, fl, acc);
#endif
}

// 9A/9B Kernel 2: Batched Speculative Verification (T = 2..8, S = 1)
//   3*T warps per block read each K/V tile ONCE via coalesced 32-lane Q8_0/F16 loads into dual s_K/s_V (2 barriers/tile)
//   and grid(8, n_splits) shares 1088B KV rows in L2 cache!
template <bool K_Q8, bool V_Q8, bool MASK_F16, int T, int MAX_SPLITS>
static __global__ void inkling_rel_pos_fattn_batched_kpos(
        const float   * __restrict__ Q,
        const char    * __restrict__ K_bytes,
        const char    * __restrict__ V_bytes,
        const float   * __restrict__ R,
        const float   * __restrict__ rel_bias,
        const float   * __restrict__ q_sinks,
        const int32_t * __restrict__ k_pos,
        const int32_t * __restrict__ q_pos,
        const void    * __restrict__ raw_mask,
        float         * __restrict__ dst_f32,
        float         * __restrict__ scratch,   // [8][MAX_SPLITS][3*T][132]
        int32_t       * __restrict__ cnt,       // [8]
        const int n_kv,
        const int n_extent,
        const size_t q_tok_stride_f,
        const size_t q_head_stride_f,
        const size_t k_head_stride_b,
        const size_t k_tok_stride_b,
        const size_t v_head_stride_b,
        const size_t v_tok_stride_b,
        const float kq_scale,
        const int n_splits) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int TILE     = 16;
    constexpr int NWARPS   = 3 * T;
    constexpr int NTHREADS = NWARPS * 32;
    const int kvh   = blockIdx.x; // 0..7
    const int sp    = blockIdx.y; // 0..n_splits-1
    const int lane  = threadIdx.x; // 0..31
    const int warp  = threadIdx.y; // 0..3*T-1
    const int tid   = warp * 32 + lane;
    const int klane = lane & 7;
    const int kgrp  = lane >> 3;
    const int tok   = warp / 3;
    const int ql    = warp - tok * 3;
    const int qh    = kvh * 3 + ql;

    const float4 * qp = (const float4 *) (Q + (size_t) tok * q_tok_stride_f + (size_t) qh * q_head_stride_f) + klane * 4;
    float4 q0 = __ldg(qp + 0); q0.x *= kq_scale; q0.y *= kq_scale; q0.z *= kq_scale; q0.w *= kq_scale;
    float4 q1 = __ldg(qp + 1); q1.x *= kq_scale; q1.y *= kq_scale; q1.z *= kq_scale; q1.w *= kq_scale;
    float4 q2 = __ldg(qp + 2); q2.x *= kq_scale; q2.y *= kq_scale; q2.z *= kq_scale; q2.w *= kq_scale;
    float4 q3 = __ldg(qp + 3); q3.x *= kq_scale; q3.y *= kq_scale; q3.z *= kq_scale; q3.w *= kq_scale;
    const float2 r_v = __ldg((const float2 *) (R + ((size_t) tok * 24 + qh) * 16) + klane);
    const float sink = __ldg(&q_sinks[(size_t) tok * 24 + qh]);
    const int my_qp  = __ldg(&q_pos[tok]);

    float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float m_max = -1e30f;
    float l_sum = 0.0f;

    __shared__ __align__(16) uint4  s_K[TILE][16]; // 4096 B (half[16][128])
    __shared__ __align__(16) uint4  s_V[TILE][16]; // 4096 B (half[16][128])
    __shared__ float                s_M[T][TILE];
    __shared__ int32_t              s_kpos[TILE];
    __shared__ float                s_W[NWARPS][TILE];
    __shared__ int                  s_rel_flag;

    if (tid == 0) {
        s_rel_flag = 0;
    }
    __syncthreads();

    const int n_tiles     = (n_kv + TILE - 1) / TILE;
    const int chunk_tiles = (n_tiles + n_splits - 1) / n_splits;
    const int t_beg       = sp * chunk_tiles;
    const int t_end       = min(n_tiles, t_beg + chunk_tiles);

    const char * K_base = K_bytes + (size_t) kvh * k_head_stride_b;
    const char * V_base = V_bytes + (size_t) kvh * v_head_stride_b;

    for (int ti = t_beg; ti < t_end; ++ti) {
        const int j0 = ti * TILE;
        int my_any = 0;
        if (warp < T && lane < TILE) {
            float mv = -INFINITY;
            int kp = -1;
            if (j0 + lane < n_kv) {
                if constexpr (MASK_F16) {
                    mv = __half2float(__ldg(&((const half *) raw_mask)[(size_t) warp * n_kv + j0 + lane]));
                } else {
                    mv = __ldg(&((const float *) raw_mask)[(size_t) warp * n_kv + j0 + lane]);
                }
                kp = __ldg(&k_pos[j0 + lane]);
                if (warp == 0) {
                    s_kpos[lane] = kp;
                }
            } else if (warp == 0) {
                s_kpos[lane] = -1;
            }
            s_M[warp][lane] = mv;
            if (mv > -1e20f) {
                my_any = 1;
                const int d = (kp >= 0) ? (__ldg(&q_pos[warp]) - kp) : -1;
                if (d >= 0 && d < n_extent) {
                    s_rel_flag = 1;
                }
            }
        }
        if (!__syncthreads_or(my_any)) {
            continue;
        }
        const int rel_tile = s_rel_flag;

        // Load BOTH K and V tiles in parallel before ONE single __syncthreads()!
        if constexpr (!K_Q8) {
            for (int idx = tid; idx < TILE * 16; idx += NTHREADS) {
                const int r   = idx >> 4;
                const int col = idx & 15;
                if (j0 + r < n_kv) {
                    inkling_cp_async_cg_16(&s_K[r][col], (const uint4 *) (K_base + (size_t) (j0 + r) * k_tok_stride_b) + col);
                }
            }
        } else {
            for (int r = warp; r < TILE; r += NWARPS) {
                if (j0 + r < n_kv) {
                    const float4 kv = inkling_load_q8_128_warp(K_base + (size_t) (j0 + r) * k_tok_stride_b, lane);
                    ((uint2 *) &s_K[r][0])[lane] = inkling_pack_f32x4_to_f16x4(kv);
                }
            }
        }

        if constexpr (!V_Q8) {
            for (int idx = tid; idx < TILE * 16; idx += NTHREADS) {
                const int r   = idx >> 4;
                const int col = idx & 15;
                if (j0 + r < n_kv) {
                    inkling_cp_async_cg_16(&s_V[r][col], (const uint4 *) (V_base + (size_t) (j0 + r) * v_tok_stride_b) + col);
                }
            }
        } else {
            for (int r = warp; r < TILE; r += NWARPS) {
                if (j0 + r < n_kv) {
                    const float4 vv = inkling_load_q8_128_warp(V_base + (size_t) (j0 + r) * v_tok_stride_b, lane);
                    ((uint2 *) &s_V[r][0])[lane] = inkling_pack_f32x4_to_f16x4(vv);
                }
            }
        }

        if constexpr (!K_Q8 || !V_Q8) {
            inkling_cp_async_commit();
            inkling_cp_async_wait<0>();
        }
        __syncthreads();
        if (tid == 0) {
            s_rel_flag = 0;
        }

        // Check per-token active bitmask and compute BOTH QK^T softmax and PV with zero mid-tile barriers!
        const float my_mv_lane = (lane < TILE) ? s_M[tok][lane] : -INFINITY;
        const uint32_t act = __ballot_sync(0xffffffff, my_mv_lane > -1e20f) & 0xFFFFu;

        if (act != 0u) {
            float sc_reg = -1e30f;
            float nmax   = m_max;
            const int gbase = kgrp * 4;
#pragma unroll
            for (int step = 0; step < 4; ++step) {
                const int bit = gbase + step;
                float a = 0.0f;
                if ((act >> bit) & 1u) {
                    a = inkling_f16_smem_dot16(&s_K[bit][0], klane, q0, q1, q2, q3);
                    if (rel_tile) {
                        const int kp = s_kpos[bit];
                        const int d  = (kp >= 0) ? (my_qp - kp) : -1;
                        if (d >= 0 && d < n_extent) {
                            const float2 rb = __ldg((const float2 *) (rel_bias + (size_t) d * 16) + klane);
                            a += r_v.x * rb.x + r_v.y * rb.y;
                        }
                    }
                }
                const float s_val = inkling_warp_sum8(a);
                const float s     = ((act >> bit) & 1u) ? (s_val + s_M[tok][bit]) : -1e30f;
                nmax = fmaxf(nmax, s);
                if (klane == step) {
                    sc_reg = s;
                }
            }
            if (klane < 4) {
                s_W[warp][gbase + klane] = sc_reg;
            }
            __syncwarp();

            nmax = fmaxf(nmax, __shfl_xor_sync(0xffffffff, nmax, 8));
            nmax = fmaxf(nmax, __shfl_xor_sync(0xffffffff, nmax, 16));
            const float rsc = (m_max > -1e20f) ? expf(m_max - nmax) : 0.0f;
            m_max = nmax;

            float w_lane = 0.0f;
            if (lane < TILE) {
                w_lane = ((act >> lane) & 1u) ? expf(s_W[warp][lane] - m_max) : 0.0f;
                s_W[warp][lane] = w_lane;
            }
            l_sum = l_sum * rsc + warp_reduce_sum<32>(w_lane);
            o.x *= rsc; o.y *= rsc; o.z *= rsc; o.w *= rsc;
            __syncwarp();

#pragma unroll
            for (int bit = 0; bit < TILE; ++bit) {
                const float kw = s_W[warp][bit];
                if (kw != 0.0f) {
                    inkling_f16_smem_v4_acc(&s_V[bit][0], lane, kw, o);
                }
            }
        }
        __syncthreads();
    }

    auto write_final = [&](float fm, float fl, float4 acc) {
        float inv_l = 0.0f;
        if (fl > 0.0f) {
            const float nm     = fmaxf(fm, sink);
            const float e_old  = expf(fm - nm);
            const float e_sink = expf(sink - nm);
            inv_l = e_old / (fl * e_old + e_sink);
        }
        float4 res;
        res.x = acc.x * inv_l; res.y = acc.y * inv_l;
        res.z = acc.z * inv_l; res.w = acc.w * inv_l;
        ((float4 *) (dst_f32 + ((size_t) tok * 24 + qh) * 128))[lane] = res;
    };

    if (n_splits == 1) {
        write_final(m_max, l_sum, o);
        return;
    }

    float * sp_ptr = scratch + ((size_t) (kvh * MAX_SPLITS + sp) * NWARPS + warp) * 132;
    if (l_sum > 0.0f) {
        ((float4 *) sp_ptr)[lane] = o;
    }
    if (lane == 0) {
        sp_ptr[128] = m_max;
        sp_ptr[129] = l_sum;
    }
    __syncthreads();
    __shared__ bool is_last;
    if (warp == 0 && lane == 0) {
        __threadfence();
        const int done = atomicAdd(&cnt[kvh], 1) + 1;
        is_last = (done == n_splits);
        if (is_last) {
            cnt[kvh] = 0;
        }
    }
    __syncthreads();
    if (!is_last) {
        return;
    }

    float fm = -1e30f;
    for (int s = 0; s < n_splits; ++s) {
        const float * sp_p = scratch + ((size_t) (kvh * MAX_SPLITS + s) * NWARPS + warp) * 132;
        if (sp_p[129] > 0.0f) {
            fm = fmaxf(fm, sp_p[128]);
        }
    }
    float fl = 0.0f;
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int s = 0; s < n_splits; ++s) {
        const float * sp_p = scratch + ((size_t) (kvh * MAX_SPLITS + s) * NWARPS + warp) * 132;
        const float sl = sp_p[129];
        if (sl == 0.0f) {
            continue;
        }
        const float es = expf(sp_p[128] - fm);
        fl += sl * es;
        const float4 sv = ((const float4 *) sp_p)[lane];
        acc.x += sv.x * es; acc.y += sv.y * es;
        acc.z += sv.z * es; acc.w += sv.w * es;
    }
    write_final(fm, fl, acc);
#endif
}

static __device__ __forceinline__ void inkling_mma_m16n8k16_f32_f16(
        float & d0, float & d1, float & d2, float & d3,
        const uint32_t a0, const uint32_t a1, const uint32_t a2, const uint32_t a3,
        const uint32_t b0, const uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

static __device__ __forceinline__ float inkling_dot16_f32(const float * a, const float * b) {
    const float4 a0 = *(const float4 *) (a + 0);
    const float4 a1 = *(const float4 *) (a + 4);
    const float4 a2 = *(const float4 *) (a + 8);
    const float4 a3 = *(const float4 *) (a + 12);
    const float4 b0 = __ldg((const float4 *) (b + 0));
    const float4 b1 = __ldg((const float4 *) (b + 4));
    const float4 b2 = __ldg((const float4 *) (b + 8));
    const float4 b3 = __ldg((const float4 *) (b + 12));
    return a0.x*b0.x + a0.y*b0.y + a0.z*b0.z + a0.w*b0.w
         + a1.x*b1.x + a1.y*b1.y + a1.z*b1.z + a1.w*b1.w
         + a2.x*b2.x + a2.y*b2.y + a2.z*b2.z + a2.w*b2.w
         + a3.x*b3.x + a3.y*b3.y + a3.z*b3.z + a3.w*b3.w;
}

// 9A/9B Kernel 3: SM 8.6 Tensor Core Prefill & General Multi-Stream Flash Attention (D = 128, BQ = 64, BKV = 32)
//   Coalesced single-pass s_M[64][32] mask cache + grid(24, q_blks, S) for 3x GQA L2 reuse + coalesced Q8_0 KV.
template <bool K_Q8, bool V_Q8, bool MASK_F16>
__launch_bounds__(128, 3)
static __global__ void inkling_rel_pos_fattn_prefill_mma_kpos(
        const float   * __restrict__ Q,
        const char    * __restrict__ K_bytes,
        const char    * __restrict__ V_bytes,
        const float   * __restrict__ R,
        const float   * __restrict__ rel_bias,
        const float   * __restrict__ q_sinks,
        const int32_t * __restrict__ k_pos,
        const int32_t * __restrict__ q_pos,
        const void    * __restrict__ raw_mask,
        float         * __restrict__ dst_f32,
        const int T_s,
        const int n_kv,
        const int n_extent,
        const size_t q_tok_stride_f,
        const size_t q_head_stride_f,
        const size_t q_stream_stride_f,
        const size_t k_head_stride_b,
        const size_t k_tok_stride_b,
        const size_t k_stream_stride_b,
        const size_t v_head_stride_b,
        const size_t v_tok_stride_b,
        const size_t v_stream_stride_b,
        const float kq_scale) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int BQ  = 64;
    constexpr int BKV = 32;
    const int qh    = blockIdx.x; // 0..23 (3 GQA heads share same kvh & q_blk in L2!)
    const int q_blk = blockIdx.y;
    const int strm  = blockIdx.z; // 0..S-1
    const int kvh   = qh / 3;     // 0..7
    const int lane  = threadIdx.x; // 0..31
    const int warp  = threadIdx.y; // 0..3
    const int tid   = warp * 32 + lane; // 0..127
    const int gid   = lane >> 2;  // 0..7
    const int tid4  = lane & 3;   // 0..3

    const int q_block_base = q_blk * BQ;
    const int q_warp_base  = q_block_base + warp * 16;
    const int tok0         = q_warp_base + gid;
    const int tok1         = q_warp_base + gid + 8;

    const float * Q_s   = Q + (size_t) strm * q_stream_stride_f + (size_t) qh * q_head_stride_f;
    const char  * K_s   = K_bytes + (size_t) strm * k_stream_stride_b + (size_t) kvh * k_head_stride_b;
    const char  * V_s   = V_bytes + (size_t) strm * v_stream_stride_b + (size_t) kvh * v_head_stride_b;
    const int32_t * kp_s = k_pos + (size_t) strm * n_kv;
    const int32_t * qp_s = q_pos + (size_t) strm * T_s;
    const float * R_s   = R + ((size_t) strm * T_s * 24 + qh) * 16;
    const float * sk_s  = q_sinks + (size_t) strm * T_s * 24 + qh;
    float * dst_s       = dst_f32 + ((size_t) strm * T_s * 24 + qh) * 128;

    // Shared memory (~17.6 KB/block -> still fits 4 blocks/SM on SM 8.6!)
    __shared__ __align__(16) union {
        half s_K[BKV][136];
        half s_VT[128][34]; // 34 halfs = 17 u32 -> bank-conflict-free & 4B aligned!
    } smem;
    __shared__ __align__(16) half  s_M[BQ][BKV]; // 4096 B single-pass mask cache!
    __shared__ __align__(16) float s_R[BQ][16];
    __shared__ int32_t s_qpos[BQ];
    __shared__ int32_t s_kpos[BKV];

    // Load s_R[64][16] and s_qpos[64] cooperatively
    for (int idx = tid; idx < BQ * 4; idx += 128) {
        const int qr = idx >> 2;
        const int c4 = idx & 3;
        const int t  = q_block_base + qr;
        float4 rv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (t < T_s) {
            rv = __ldg((const float4 *) (R_s + (size_t) t * 24 * 16) + c4);
        }
        ((float4 *) &s_R[qr][0])[c4] = rv;
    }
    if (tid < BQ) {
        const int t = q_block_base + tid;
        s_qpos[tid] = (t < T_s) ? __ldg(&qp_s[t]) : -1;
    }
    __syncthreads();

    // Load Q into MMA A-fragments q_frag[8][4] (pre-scaled by kq_scale -> half2)
    uint32_t q_frag[8][4];
    const float * qptr0 = (tok0 < T_s) ? (Q_s + (size_t) tok0 * q_tok_stride_f) : nullptr;
    const float * qptr1 = (tok1 < T_s) ? (Q_s + (size_t) tok1 * q_tok_stride_f) : nullptr;
#pragma unroll
    for (int kd = 0; kd < 8; ++kd) {
        const int d0 = kd * 16 + tid4 * 2;
        const int d1 = d0 + 8;
        float2 f00 = qptr0 ? __ldg((const float2 *) (qptr0 + d0)) : make_float2(0.0f, 0.0f);
        float2 f10 = qptr1 ? __ldg((const float2 *) (qptr1 + d0)) : make_float2(0.0f, 0.0f);
        float2 f01 = qptr0 ? __ldg((const float2 *) (qptr0 + d1)) : make_float2(0.0f, 0.0f);
        float2 f11 = qptr1 ? __ldg((const float2 *) (qptr1 + d1)) : make_float2(0.0f, 0.0f);
        half2 h00 = __floats2half2_rn(f00.x * kq_scale, f00.y * kq_scale);
        half2 h10 = __floats2half2_rn(f10.x * kq_scale, f10.y * kq_scale);
        half2 h01 = __floats2half2_rn(f01.x * kq_scale, f01.y * kq_scale);
        half2 h11 = __floats2half2_rn(f11.x * kq_scale, f11.y * kq_scale);
        q_frag[kd][0] = *reinterpret_cast<const uint32_t *>(&h00);
        q_frag[kd][1] = *reinterpret_cast<const uint32_t *>(&h10);
        q_frag[kd][2] = *reinterpret_cast<const uint32_t *>(&h01);
        q_frag[kd][3] = *reinterpret_cast<const uint32_t *>(&h11);
    }

    const int my_qpos0  = (tok0 < T_s) ? s_qpos[warp * 16 + gid]     : -1;
    const int my_qpos1  = (tok1 < T_s) ? s_qpos[warp * 16 + gid + 8] : -1;
    const float my_sk0  = (tok0 < T_s) ? __ldg(&sk_s[(size_t) tok0 * 24]) : 0.0f;
    const float my_sk1  = (tok1 < T_s) ? __ldg(&sk_s[(size_t) tok1 * 24]) : 0.0f;

    int qpos_min = 0x7fffffff;
    int qpos_max = -1;
    for (int qr = 0; qr < BQ; ++qr) {
        const int qp = s_qpos[qr];
        if (qp >= 0) {
            if (qp < qpos_min) qpos_min = qp;
            if (qp > qpos_max) qpos_max = qp;
        }
    }

    float o[16][4];
#pragma unroll
    for (int dn = 0; dn < 16; ++dn) {
        o[dn][0] = 0.0f; o[dn][1] = 0.0f; o[dn][2] = 0.0f; o[dn][3] = 0.0f;
    }
    float m0 = -1e30f, l0 = 0.0f;
    float m1 = -1e30f, l1 = 0.0f;

    const bool mask_u4_ok = MASK_F16 && ((n_kv & 7) == 0) && ((((uintptr_t) raw_mask) & 15) == 0);
    const half h_ninf = __float2half(-INFINITY);
    const uint32_t u_ninf2 = 0xFC00FC00u;

    for (int j0 = 0; j0 < n_kv; j0 += BKV) {
        // Single-pass coalesced load of 64x32 mask tile into s_M[64][32] + early empty-tile skip!
        int any_fin = 0;
        if (mask_u4_ok && j0 + BKV <= n_kv) {
            // 256 uint4s total -> exactly 2 coalesced 16B loads per thread!
            const int r_base = tid >> 2;  // 0..31
            const int c_u4   = tid & 3;   // 0..3 (8 halfs each)
#pragma unroll
            for (int pass = 0; pass < 2; ++pass) {
                const int qr = r_base + pass * 32;
                const int t_local = q_block_base + qr;
                uint4 mv4 = make_uint4(u_ninf2, u_ninf2, u_ninf2, u_ninf2);
                if (t_local < T_s) {
                    const size_t midx = ((size_t) strm * T_s + t_local) * n_kv + j0 + c_u4 * 8;
                    mv4 = __ldg((const uint4 *) (((const half *) raw_mask) + midx));
                    if (mv4.x != u_ninf2 || mv4.y != u_ninf2 || mv4.z != u_ninf2 || mv4.w != u_ninf2) {
                        any_fin = 1;
                    }
                }
                ((uint4 *) &s_M[qr][0])[c_u4] = mv4;
            }
        } else {
            for (int idx = tid; idx < BQ * BKV; idx += 128) {
                const int qr = idx >> 5;
                const int jc = idx & 31;
                const int t_local = q_block_base + qr;
                const int j_abs   = j0 + jc;
                half hm = h_ninf;
                if (t_local < T_s && j_abs < n_kv) {
                    const size_t midx = ((size_t) strm * T_s + t_local) * n_kv + j_abs;
                    float fv = MASK_F16 ? __half2float(__ldg(&((const half *) raw_mask)[midx]))
                                        : __ldg(&((const float *) raw_mask)[midx]);
                    if (fv > -1e20f) {
                        any_fin = 1;
                        hm = __float2half(fv);
                    }
                }
                s_M[qr][jc] = hm;
            }
        }
        if (!__syncthreads_or(any_fin)) {
            continue;
        }

        // Load s_kpos[32] and check if any cell is within relative extent of this Q block
        int rel_flag = 0;
        if (tid < BKV) {
            const int kp = (j0 + tid < n_kv) ? __ldg(&kp_s[j0 + tid]) : -1;
            s_kpos[tid] = kp;
            if (kp >= 0 && qpos_max >= kp && qpos_min < kp + n_extent) {
                rel_flag = 1;
            }
        }
        const int has_rel = __syncthreads_or(rel_flag);

        // Load K tile (32 x 128) into smem.s_K[32][136] as half
        if constexpr (!K_Q8) {
            for (int idx = tid; idx < BKV * 16; idx += 128) {
                const int r   = idx >> 4;
                const int col = idx & 15;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (j0 + r < n_kv) {
                    v = __ldg((const uint4 *) (K_s + (size_t) (j0 + r) * k_tok_stride_b) + col);
                }
                ((uint4 *) &smem.s_K[r][0])[col] = v;
            }
        } else {
            // Coalesced 32-lane Q8_0 load: 4 warps x 8 iters = 32 rows
#pragma unroll
            for (int it = 0; it < 8; ++it) {
                const int r = it * 4 + warp;
                uint2 packed = make_uint2(0, 0);
                if (j0 + r < n_kv) {
                    const float4 kv = inkling_load_q8_128_warp(K_s + (size_t) (j0 + r) * k_tok_stride_b, lane);
                    packed = inkling_pack_f32x4_to_f16x4(kv);
                }
                ((uint2 *) &smem.s_K[r][0])[lane] = packed;
            }
        }
        __syncthreads();

        // Compute S = Q @ K^T [16 x 32] per warp via Tensor Core m16n8k16
        float s[4][4];
#pragma unroll
        for (int kn = 0; kn < 4; ++kn) {
            s[kn][0] = 0.0f; s[kn][1] = 0.0f; s[kn][2] = 0.0f; s[kn][3] = 0.0f;
        }
#pragma unroll
        for (int kd = 0; kd < 8; ++kd) {
            const int d0 = kd * 16 + tid4 * 2;
            const int d1 = d0 + 8;
#pragma unroll
            for (int kn = 0; kn < 4; ++kn) {
                const int kr = kn * 8 + gid;
                const uint32_t b0 = *(const uint32_t *) &smem.s_K[kr][d0];
                const uint32_t b1 = *(const uint32_t *) &smem.s_K[kr][d1];
                inkling_mma_m16n8k16_f32_f16(
                    s[kn][0], s[kn][1], s[kn][2], s[kn][3],
                    q_frag[kd][0], q_frag[kd][1], q_frag[kd][2], q_frag[kd][3],
                    b0, b1);
            }
        }

        // Add cached s_M (SMEM half2!) + on-the-fly RelPos bias (only when has_rel != 0!)
        const int qr0 = warp * 16 + gid;
        const int qr1 = qr0 + 8;
        const float * r_ptr0 = &s_R[qr0][0];
        const float * r_ptr1 = &s_R[qr1][0];
        float m_tile0 = -1e30f;
        float m_tile1 = -1e30f;
#pragma unroll
        for (int kn = 0; kn < 4; ++kn) {
            const int c0 = kn * 8 + tid4 * 2;
            const int c1 = c0 + 1;
            const float2 mv0 = __half22float2(*(const half2 *) &s_M[qr0][c0]);
            const float2 mv1 = __half22float2(*(const half2 *) &s_M[qr1][c0]);

            if (mv0.x > -1e20f) {
                float val = s[kn][0] + mv0.x;
                if (has_rel) {
                    const int kp = s_kpos[c0];
                    const int d  = (kp >= 0) ? (my_qpos0 - kp) : -1;
                    if (d >= 0 && d < n_extent) {
                        val += inkling_dot16_f32(r_ptr0, rel_bias + (size_t) d * 16);
                    }
                }
                s[kn][0] = val;
                m_tile0 = fmaxf(m_tile0, val);
            } else {
                s[kn][0] = -INFINITY;
            }

            if (mv0.y > -1e20f) {
                float val = s[kn][1] + mv0.y;
                if (has_rel) {
                    const int kp = s_kpos[c1];
                    const int d  = (kp >= 0) ? (my_qpos0 - kp) : -1;
                    if (d >= 0 && d < n_extent) {
                        val += inkling_dot16_f32(r_ptr0, rel_bias + (size_t) d * 16);
                    }
                }
                s[kn][1] = val;
                m_tile0 = fmaxf(m_tile0, val);
            } else {
                s[kn][1] = -INFINITY;
            }

            if (mv1.x > -1e20f) {
                float val = s[kn][2] + mv1.x;
                if (has_rel) {
                    const int kp = s_kpos[c0];
                    const int d  = (kp >= 0) ? (my_qpos1 - kp) : -1;
                    if (d >= 0 && d < n_extent) {
                        val += inkling_dot16_f32(r_ptr1, rel_bias + (size_t) d * 16);
                    }
                }
                s[kn][2] = val;
                m_tile1 = fmaxf(m_tile1, val);
            } else {
                s[kn][2] = -INFINITY;
            }

            if (mv1.y > -1e20f) {
                float val = s[kn][3] + mv1.y;
                if (has_rel) {
                    const int kp = s_kpos[c1];
                    const int d  = (kp >= 0) ? (my_qpos1 - kp) : -1;
                    if (d >= 0 && d < n_extent) {
                        val += inkling_dot16_f32(r_ptr1, rel_bias + (size_t) d * 16);
                    }
                }
                s[kn][3] = val;
                m_tile1 = fmaxf(m_tile1, val);
            } else {
                s[kn][3] = -INFINITY;
            }
        }

        // 4-lane row max reduction across tid4 = 0..3
        m_tile0 = fmaxf(m_tile0, __shfl_xor_sync(0xffffffff, m_tile0, 1));
        m_tile0 = fmaxf(m_tile0, __shfl_xor_sync(0xffffffff, m_tile0, 2));
        m_tile1 = fmaxf(m_tile1, __shfl_xor_sync(0xffffffff, m_tile1, 1));
        m_tile1 = fmaxf(m_tile1, __shfl_xor_sync(0xffffffff, m_tile1, 2));

        float rsc0 = 1.0f, rsc1 = 1.0f;
        float m_new0 = m0, m_new1 = m1;
        if (m_tile0 > -1e20f) {
            m_new0 = fmaxf(m0, m_tile0);
            rsc0   = (m0 > -1e20f) ? expf(m0 - m_new0) : 0.0f;
            m0     = m_new0;
        }
        if (m_tile1 > -1e20f) {
            m_new1 = fmaxf(m1, m_tile1);
            rsc1   = (m1 > -1e20f) ? expf(m1 - m_new1) : 0.0f;
            m1     = m_new1;
        }

        float l_tile0 = 0.0f, l_tile1 = 0.0f;
        uint32_t p_frag[2][4];
#pragma unroll
        for (int kn = 0; kn < 4; ++kn) {
            const float p00 = (s[kn][0] > -1e20f) ? expf(s[kn][0] - m_new0) : 0.0f;
            const float p01 = (s[kn][1] > -1e20f) ? expf(s[kn][1] - m_new0) : 0.0f;
            const float p10 = (s[kn][2] > -1e20f) ? expf(s[kn][2] - m_new1) : 0.0f;
            const float p11 = (s[kn][3] > -1e20f) ? expf(s[kn][3] - m_new1) : 0.0f;
            l_tile0 += p00 + p01;
            l_tile1 += p10 + p11;
            const half2 h0 = __floats2half2_rn(p00, p01);
            const half2 h1 = __floats2half2_rn(p10, p11);
            const int pk = kn >> 1;
            const int ps = (kn & 1) * 2;
            p_frag[pk][ps + 0] = *reinterpret_cast<const uint32_t *>(&h0);
            p_frag[pk][ps + 1] = *reinterpret_cast<const uint32_t *>(&h1);
        }

        l_tile0 += __shfl_xor_sync(0xffffffff, l_tile0, 1);
        l_tile0 += __shfl_xor_sync(0xffffffff, l_tile0, 2);
        l_tile1 += __shfl_xor_sync(0xffffffff, l_tile1, 1);
        l_tile1 += __shfl_xor_sync(0xffffffff, l_tile1, 2);
        l0 = l0 * rsc0 + l_tile0;
        l1 = l1 * rsc1 + l_tile1;

#pragma unroll
        for (int dn = 0; dn < 16; ++dn) {
            o[dn][0] *= rsc0; o[dn][1] *= rsc0;
            o[dn][2] *= rsc1; o[dn][3] *= rsc1;
        }

        __syncthreads(); // Done reading smem.s_K; now load V transposed into smem.s_VT[128][34]

        if constexpr (!V_Q8) {
            for (int idx = tid; idx < BKV * 64; idx += 128) {
                const int r  = idx >> 6;
                const int c2 = idx & 63;
                half2 h2 = make_half2(__float2half(0.0f), __float2half(0.0f));
                if (j0 + r < n_kv) {
                    h2 = __ldg((const half2 *) (V_s + (size_t) (j0 + r) * v_tok_stride_b) + c2);
                }
                smem.s_VT[c2 * 2 + 0][r] = __low2half(h2);
                smem.s_VT[c2 * 2 + 1][r] = __high2half(h2);
            }
        } else {
            const int d_base = lane * 4;
#pragma unroll
            for (int it = 0; it < 8; ++it) {
                const int r = it * 4 + warp;
                if (j0 + r < n_kv) {
                    const float4 vv = inkling_load_q8_128_warp(V_s + (size_t) (j0 + r) * v_tok_stride_b, lane);
                    smem.s_VT[d_base + 0][r] = __float2half(vv.x);
                    smem.s_VT[d_base + 1][r] = __float2half(vv.y);
                    smem.s_VT[d_base + 2][r] = __float2half(vv.z);
                    smem.s_VT[d_base + 3][r] = __float2half(vv.w);
                } else {
                    const half hz = __float2half(0.0f);
                    smem.s_VT[d_base + 0][r] = hz;
                    smem.s_VT[d_base + 1][r] = hz;
                    smem.s_VT[d_base + 2][r] = hz;
                    smem.s_VT[d_base + 3][r] = hz;
                }
            }
        }
        __syncthreads();

        // Compute O += P @ V [16 x 128] via Tensor Core m16n8k16
#pragma unroll
        for (int dn = 0; dn < 16; ++dn) {
            const int dr = dn * 8 + gid;
#pragma unroll
            for (int pk = 0; pk < 2; ++pk) {
                const int k0 = pk * 16 + tid4 * 2;
                const int k1 = k0 + 8;
                const uint32_t b0 = *(const uint32_t *) &smem.s_VT[dr][k0];
                const uint32_t b1 = *(const uint32_t *) &smem.s_VT[dr][k1];
                inkling_mma_m16n8k16_f32_f16(
                    o[dn][0], o[dn][1], o[dn][2], o[dn][3],
                    p_frag[pk][0], p_frag[pk][1], p_frag[pk][2], p_frag[pk][3],
                    b0, b1);
            }
        }
        __syncthreads();
    }

    // Finalize with per-query sink logit and write output [128, 24, T_s, S]
    float inv_l0 = 0.0f, inv_l1 = 0.0f;
    if (l0 > 0.0f) {
        const float nm = fmaxf(m0, my_sk0);
        const float eo = expf(m0 - nm);
        const float es = expf(my_sk0 - nm);
        inv_l0 = eo / (l0 * eo + es);
    }
    if (l1 > 0.0f) {
        const float nm = fmaxf(m1, my_sk1);
        const float eo = expf(m1 - nm);
        const float es = expf(my_sk1 - nm);
        inv_l1 = eo / (l1 * eo + es);
    }
    if (tok0 < T_s) {
        float * out0 = dst_s + (size_t) tok0 * 24 * 128;
#pragma unroll
        for (int dn = 0; dn < 16; ++dn) {
            const int c = dn * 8 + tid4 * 2;
            *(float2 *) (out0 + c) = make_float2(o[dn][0] * inv_l0, o[dn][1] * inv_l0);
        }
    }
    if (tok1 < T_s) {
        float * out1 = dst_s + (size_t) tok1 * 24 * 128;
#pragma unroll
        for (int dn = 0; dn < 16; ++dn) {
            const int c = dn * 8 + tid4 * 2;
            *(float2 *) (out1 + c) = make_float2(o[dn][2] * inv_l1, o[dn][3] * inv_l1);
        }
    }
#endif
}

// 9B-MMA Kernel: Zero-Block-Sync Tensor-Core (m16n8k16) Split-K Batched Verification (T = 2..8, S = 1)
//   Packs all 3*T GQA queries for each KV head into NQT = ceil(3*T/16) (1 for T<=5, 2 for T=6..8) MMA tiles.
//   4 warps/block stream 64 contiguous KV tokens/iter with ZERO __syncthreads() in the KV loop and full SM 8.6 MMA!
template <bool K_Q8, bool V_Q8, bool MASK_F16, int T, int MAX_SPLITS>
__launch_bounds__(128, 3)
static __global__ void inkling_rel_pos_fattn_batched_mma_kpos(
        const float   * __restrict__ Q,
        const char    * __restrict__ K_bytes,
        const char    * __restrict__ V_bytes,
        const float   * __restrict__ R,
        const float   * __restrict__ rel_bias,
        const float   * __restrict__ q_sinks,
        const int32_t * __restrict__ k_pos,
        const int32_t * __restrict__ q_pos,
        const void    * __restrict__ raw_mask,
        float         * __restrict__ dst_f32,
        float         * __restrict__ scratch,   // [8 * NQT][MAX_SPLITS][16][132]
        int32_t       * __restrict__ cnt,       // [8 * NQT]
        const int n_kv,
        const int n_extent,
        const size_t q_tok_stride_f,
        const size_t q_head_stride_f,
        const size_t k_head_stride_b,
        const size_t k_tok_stride_b,
        const size_t v_head_stride_b,
        const size_t v_tok_stride_b,
        const float kq_scale,
        const int n_splits) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int M_Q   = 3 * T;
    constexpr int NQT   = (M_Q + 15) / 16;
    const int kvh  = blockIdx.x; // 0..7
    const int sp   = blockIdx.y; // 0..n_splits-1
    const int qt   = blockIdx.z; // 0..NQT-1
    const int lane = threadIdx.x; // 0..31
    const int warp = threadIdx.y; // 0..3
    const int tid  = warp * 32 + lane;
    const int gid  = lane >> 2;   // 0..7
    const int tid4 = lane & 3;    // 0..3

    const int q_base = qt * 16;
    const int qr0    = q_base + gid;
    const int qr1    = qr0 + 8;

    __shared__ __align__(16) union {
        half  s_K[4][16][136];    // 17,408 B
        half  s_VT[4][128][18];   // 18,432 B (bank-conflict-free)
        float s_red[4][8][132];   // 16,896 B (16B-aligned stride for float4 intra-block 4-warp reduction!)
    } wsmem;
    __shared__ __align__(16) half  s_M[4][T][16]; // max 1,024 B
    __shared__ __align__(16) float s_R[16][16];   // 1,024 B
    __shared__ int32_t s_kpos[4][16];             //   256 B
    __shared__ int32_t s_qpos[16];                //    64 B
    __shared__ float   s_sink[16];                //    64 B
    __shared__ bool    is_last;

    // Load s_R[16][16], s_qpos[16], s_sink[16] once
    if (tid < 64) {
        const int r  = tid >> 2; // 0..15
        const int c4 = tid & 3;  // 0..3
        const int qr = q_base + r;
        float4 rv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (qr < M_Q) {
            const int tok = qr / 3;
            const int qh  = kvh * 3 + (qr - tok * 3);
            rv = __ldg((const float4 *) (R + ((size_t) tok * 24 + qh) * 16) + c4);
        }
        ((float4 *) &s_R[r][0])[c4] = rv;
    }
    if (tid < 16) {
        const int qr = q_base + tid;
        if (qr < M_Q) {
            const int tok = qr / 3;
            const int qh  = kvh * 3 + (qr - tok * 3);
            s_qpos[tid] = __ldg(&q_pos[tok]);
            s_sink[tid] = __ldg(&q_sinks[(size_t) tok * 24 + qh]);
        } else {
            s_qpos[tid] = -1;
            s_sink[tid] = 0.0f;
        }
    }
    __syncthreads();

    // Load Q into per-warp MMA A-fragments q_frag[8][4] (pre-scaled by kq_scale -> half2)
    uint32_t q_frag[8][4];
    const int tok0 = (qr0 < M_Q) ? (qr0 / 3) : -1;
    const int qh0  = (qr0 < M_Q) ? (kvh * 3 + (qr0 - tok0 * 3)) : 0;
    const int tok1 = (qr1 < M_Q) ? (qr1 / 3) : -1;
    const int qh1  = (qr1 < M_Q) ? (kvh * 3 + (qr1 - tok1 * 3)) : 0;
    const float * qptr0 = (tok0 >= 0) ? (Q + (size_t) tok0 * q_tok_stride_f + (size_t) qh0 * q_head_stride_f) : nullptr;
    const float * qptr1 = (tok1 >= 0) ? (Q + (size_t) tok1 * q_tok_stride_f + (size_t) qh1 * q_head_stride_f) : nullptr;

#pragma unroll
    for (int kd = 0; kd < 8; ++kd) {
        const int d0 = kd * 16 + tid4 * 2;
        const int d1 = d0 + 8;
        float2 f00 = qptr0 ? __ldg((const float2 *) (qptr0 + d0)) : make_float2(0.0f, 0.0f);
        float2 f10 = qptr1 ? __ldg((const float2 *) (qptr1 + d0)) : make_float2(0.0f, 0.0f);
        float2 f01 = qptr0 ? __ldg((const float2 *) (qptr0 + d1)) : make_float2(0.0f, 0.0f);
        float2 f11 = qptr1 ? __ldg((const float2 *) (qptr1 + d1)) : make_float2(0.0f, 0.0f);
        half2 h00 = __floats2half2_rn(f00.x * kq_scale, f00.y * kq_scale);
        half2 h10 = __floats2half2_rn(f10.x * kq_scale, f10.y * kq_scale);
        half2 h01 = __floats2half2_rn(f01.x * kq_scale, f01.y * kq_scale);
        half2 h11 = __floats2half2_rn(f11.x * kq_scale, f11.y * kq_scale);
        q_frag[kd][0] = *reinterpret_cast<const uint32_t *>(&h00);
        q_frag[kd][1] = *reinterpret_cast<const uint32_t *>(&h10);
        q_frag[kd][2] = *reinterpret_cast<const uint32_t *>(&h01);
        q_frag[kd][3] = *reinterpret_cast<const uint32_t *>(&h11);
    }

    const int my_qpos0 = s_qpos[gid];
    const int my_qpos1 = s_qpos[gid + 8];
    const int t_lo = q_base / 3;
    const int t_hi = min(T - 1, (q_base + 15) / 3);
    int qpos_min = 0x7fffffff, qpos_max = -1;
    for (int r = 0; r < 16; ++r) {
        const int qp = s_qpos[r];
        if (qp >= 0) {
            if (qp < qpos_min) qpos_min = qp;
            if (qp > qpos_max) qpos_max = qp;
        }
    }

    float o[16][4];
#pragma unroll
    for (int dn = 0; dn < 16; ++dn) {
        o[dn][0] = 0.0f; o[dn][1] = 0.0f; o[dn][2] = 0.0f; o[dn][3] = 0.0f;
    }
    float m0 = -1e30f, l0 = 0.0f;
    float m1 = -1e30f, l1 = 0.0f;

    const int n_tiles     = (n_kv + 15) / 16;
    const int chunk_tiles = (n_tiles + n_splits - 1) / n_splits;
    const int t_beg       = sp * chunk_tiles;
    const int t_end       = min(n_tiles, t_beg + chunk_tiles);

    const char * K_base = K_bytes + (size_t) kvh * k_head_stride_b;
    const char * V_base = V_bytes + (size_t) kvh * v_head_stride_b;
    const half h_ninf   = __float2half(-INFINITY);
    const int d_base    = lane * 4;

    // Warp-independent KV streaming: ZERO __syncthreads() in this entire loop!
    for (int ti = t_beg + warp; ti < t_end; ti += 4) {
        const int j0 = ti * 16;
        int my_any = 0;
        int kp = -1;
        if (lane < 16) {
            if (j0 + lane < n_kv) {
                kp = __ldg(&k_pos[j0 + lane]);
            }
            s_kpos[warp][lane] = kp;
            for (int t = t_lo; t <= t_hi; ++t) {
                half hm = h_ninf;
                if (j0 + lane < n_kv) {
                    const size_t midx = (size_t) t * n_kv + j0 + lane;
                    float fv = MASK_F16 ? __half2float(__ldg(&((const half *) raw_mask)[midx]))
                                        : __ldg(&((const float *) raw_mask)[midx]);
                    if (fv > -1e20f) {
                        my_any = 1;
                        hm = __float2half(fv);
                    }
                }
                s_M[warp][t][lane] = hm;
            }
        }
        if (!__any_sync(0xffffffff, my_any)) {
            continue;
        }
        const int rel_flag = (lane < 16 && kp >= 0 && qpos_max >= kp && qpos_min < kp + n_extent) ? 1 : 0;
        const int has_rel  = __any_sync(0xffffffff, rel_flag);

        // Load K tile (16 x 128) into private wsmem.s_K[warp][0..15][0..127]
        if constexpr (!K_Q8) {
#pragma unroll
            for (int r = 0; r < 16; ++r) {
                uint2 v = make_uint2(0, 0);
                if (j0 + r < n_kv) {
                    v = __ldg((const uint2 *) (K_base + (size_t) (j0 + r) * k_tok_stride_b) + lane);
                }
                ((uint2 *) &wsmem.s_K[warp][r][0])[lane] = v;
            }
        } else {
#pragma unroll
            for (int r = 0; r < 16; ++r) {
                uint2 packed = make_uint2(0, 0);
                if (j0 + r < n_kv) {
                    const float4 kv = inkling_load_q8_128_warp(K_base + (size_t) (j0 + r) * k_tok_stride_b, lane);
                    packed = inkling_pack_f32x4_to_f16x4(kv);
                }
                ((uint2 *) &wsmem.s_K[warp][r][0])[lane] = packed;
            }
        }
        __syncwarp();

        // Compute S = Q @ K^T [16 x 16] via 16 SM 8.6 Tensor Core m16n8k16 ops!
        float s[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
#pragma unroll
        for (int kd = 0; kd < 8; ++kd) {
            const int d0 = kd * 16 + tid4 * 2;
            const int d1 = d0 + 8;
#pragma unroll
            for (int kn = 0; kn < 2; ++kn) {
                const int kr = kn * 8 + gid;
                const uint32_t b0 = *(const uint32_t *) &wsmem.s_K[warp][kr][d0];
                const uint32_t b1 = *(const uint32_t *) &wsmem.s_K[warp][kr][d1];
                inkling_mma_m16n8k16_f32_f16(
                    s[kn][0], s[kn][1], s[kn][2], s[kn][3],
                    q_frag[kd][0], q_frag[kd][1], q_frag[kd][2], q_frag[kd][3],
                    b0, b1);
            }
        }

        // Mask + on-the-fly RelPos (only when has_rel != 0!) + online softmax
        const float * r_ptr0 = &s_R[gid][0];
        const float * r_ptr1 = &s_R[gid + 8][0];
        float m_tile0 = -1e30f;
        float m_tile1 = -1e30f;
#pragma unroll
        for (int kn = 0; kn < 2; ++kn) {
            const int c0 = kn * 8 + tid4 * 2;
            const int c1 = c0 + 1;
            const float2 mv0 = (tok0 >= 0) ? __half22float2(*(const half2 *) &s_M[warp][tok0][c0]) : make_float2(-INFINITY, -INFINITY);
            const float2 mv1 = (tok1 >= 0) ? __half22float2(*(const half2 *) &s_M[warp][tok1][c0]) : make_float2(-INFINITY, -INFINITY);

            if (mv0.x > -1e20f) {
                float val = s[kn][0] + mv0.x;
                if (has_rel) {
                    const int kpc = s_kpos[warp][c0];
                    const int d   = (kpc >= 0) ? (my_qpos0 - kpc) : -1;
                    if (d >= 0 && d < n_extent) {
                        val += inkling_dot16_f32(r_ptr0, rel_bias + (size_t) d * 16);
                    }
                }
                s[kn][0] = val;
                m_tile0 = fmaxf(m_tile0, val);
            } else {
                s[kn][0] = -INFINITY;
            }

            if (mv0.y > -1e20f) {
                float val = s[kn][1] + mv0.y;
                if (has_rel) {
                    const int kpc = s_kpos[warp][c1];
                    const int d   = (kpc >= 0) ? (my_qpos0 - kpc) : -1;
                    if (d >= 0 && d < n_extent) {
                        val += inkling_dot16_f32(r_ptr0, rel_bias + (size_t) d * 16);
                    }
                }
                s[kn][1] = val;
                m_tile0 = fmaxf(m_tile0, val);
            } else {
                s[kn][1] = -INFINITY;
            }

            if (mv1.x > -1e20f) {
                float val = s[kn][2] + mv1.x;
                if (has_rel) {
                    const int kpc = s_kpos[warp][c0];
                    const int d   = (kpc >= 0) ? (my_qpos1 - kpc) : -1;
                    if (d >= 0 && d < n_extent) {
                        val += inkling_dot16_f32(r_ptr1, rel_bias + (size_t) d * 16);
                    }
                }
                s[kn][2] = val;
                m_tile1 = fmaxf(m_tile1, val);
            } else {
                s[kn][2] = -INFINITY;
            }

            if (mv1.y > -1e20f) {
                float val = s[kn][3] + mv1.y;
                if (has_rel) {
                    const int kpc = s_kpos[warp][c1];
                    const int d   = (kpc >= 0) ? (my_qpos1 - kpc) : -1;
                    if (d >= 0 && d < n_extent) {
                        val += inkling_dot16_f32(r_ptr1, rel_bias + (size_t) d * 16);
                    }
                }
                s[kn][3] = val;
                m_tile1 = fmaxf(m_tile1, val);
            } else {
                s[kn][3] = -INFINITY;
            }
        }

        m_tile0 = fmaxf(m_tile0, __shfl_xor_sync(0xffffffff, m_tile0, 1));
        m_tile0 = fmaxf(m_tile0, __shfl_xor_sync(0xffffffff, m_tile0, 2));
        m_tile1 = fmaxf(m_tile1, __shfl_xor_sync(0xffffffff, m_tile1, 1));
        m_tile1 = fmaxf(m_tile1, __shfl_xor_sync(0xffffffff, m_tile1, 2));

        float rsc0 = 1.0f, rsc1 = 1.0f;
        float m_new0 = m0, m_new1 = m1;
        if (m_tile0 > -1e20f) {
            m_new0 = fmaxf(m0, m_tile0);
            rsc0   = (m0 > -1e20f) ? expf(m0 - m_new0) : 0.0f;
            m0     = m_new0;
        }
        if (m_tile1 > -1e20f) {
            m_new1 = fmaxf(m1, m_tile1);
            rsc1   = (m1 > -1e20f) ? expf(m1 - m_new1) : 0.0f;
            m1     = m_new1;
        }

        float l_tile0 = 0.0f, l_tile1 = 0.0f;
        uint32_t p_frag[4];
#pragma unroll
        for (int kn = 0; kn < 2; ++kn) {
            const float p00 = (s[kn][0] > -1e20f) ? expf(s[kn][0] - m_new0) : 0.0f;
            const float p01 = (s[kn][1] > -1e20f) ? expf(s[kn][1] - m_new0) : 0.0f;
            const float p10 = (s[kn][2] > -1e20f) ? expf(s[kn][2] - m_new1) : 0.0f;
            const float p11 = (s[kn][3] > -1e20f) ? expf(s[kn][3] - m_new1) : 0.0f;
            l_tile0 += p00 + p01;
            l_tile1 += p10 + p11;
            const half2 h0 = __floats2half2_rn(p00, p01);
            const half2 h1 = __floats2half2_rn(p10, p11);
            p_frag[kn * 2 + 0] = *reinterpret_cast<const uint32_t *>(&h0);
            p_frag[kn * 2 + 1] = *reinterpret_cast<const uint32_t *>(&h1);
        }

        l_tile0 += __shfl_xor_sync(0xffffffff, l_tile0, 1);
        l_tile0 += __shfl_xor_sync(0xffffffff, l_tile0, 2);
        l_tile1 += __shfl_xor_sync(0xffffffff, l_tile1, 1);
        l_tile1 += __shfl_xor_sync(0xffffffff, l_tile1, 2);
        l0 = l0 * rsc0 + l_tile0;
        l1 = l1 * rsc1 + l_tile1;

#pragma unroll
        for (int dn = 0; dn < 16; ++dn) {
            o[dn][0] *= rsc0; o[dn][1] *= rsc0;
            o[dn][2] *= rsc1; o[dn][3] *= rsc1;
        }
        __syncwarp(); // Done reading wsmem.s_K[warp]; reuse for wsmem.s_VT[warp]!

        // Load V tile (16 x 128) transposed into wsmem.s_VT[warp][0..127][0..15]
        const half hz = __float2half(0.0f);
        if constexpr (!V_Q8) {
#pragma unroll
            for (int r = 0; r < 16; ++r) {
                if (j0 + r < n_kv) {
                    const uint2 raw = __ldg((const uint2 *) (V_base + (size_t) (j0 + r) * v_tok_stride_b) + lane);
                    const half2 h0 = *reinterpret_cast<const half2 *>(&raw.x);
                    const half2 h1 = *reinterpret_cast<const half2 *>(&raw.y);
                    wsmem.s_VT[warp][d_base + 0][r] = __low2half(h0);
                    wsmem.s_VT[warp][d_base + 1][r] = __high2half(h0);
                    wsmem.s_VT[warp][d_base + 2][r] = __low2half(h1);
                    wsmem.s_VT[warp][d_base + 3][r] = __high2half(h1);
                } else {
                    wsmem.s_VT[warp][d_base + 0][r] = hz;
                    wsmem.s_VT[warp][d_base + 1][r] = hz;
                    wsmem.s_VT[warp][d_base + 2][r] = hz;
                    wsmem.s_VT[warp][d_base + 3][r] = hz;
                }
            }
        } else {
#pragma unroll
            for (int r = 0; r < 16; ++r) {
                if (j0 + r < n_kv) {
                    const float4 vv = inkling_load_q8_128_warp(V_base + (size_t) (j0 + r) * v_tok_stride_b, lane);
                    wsmem.s_VT[warp][d_base + 0][r] = __float2half(vv.x);
                    wsmem.s_VT[warp][d_base + 1][r] = __float2half(vv.y);
                    wsmem.s_VT[warp][d_base + 2][r] = __float2half(vv.z);
                    wsmem.s_VT[warp][d_base + 3][r] = __float2half(vv.w);
                } else {
                    wsmem.s_VT[warp][d_base + 0][r] = hz;
                    wsmem.s_VT[warp][d_base + 1][r] = hz;
                    wsmem.s_VT[warp][d_base + 2][r] = hz;
                    wsmem.s_VT[warp][d_base + 3][r] = hz;
                }
            }
        }
        __syncwarp();

        // Compute O += P @ V [16 x 128] via 16 SM 8.6 Tensor Core m16n8k16 ops!
#pragma unroll
        for (int dn = 0; dn < 16; ++dn) {
            const int dr = dn * 8 + gid;
            const uint32_t b0 = *(const uint32_t *) &wsmem.s_VT[warp][dr][tid4 * 2];
            const uint32_t b1 = *(const uint32_t *) &wsmem.s_VT[warp][dr][tid4 * 2 + 8];
            inkling_mma_m16n8k16_f32_f16(
                o[dn][0], o[dn][1], o[dn][2], o[dn][3],
                p_frag[0], p_frag[1], p_frag[2], p_frag[3],
                b0, b1);
        }
        __syncwarp();
    }

    // 2-pass intra-block reduction across the 4 warps (half_r=0: rows 0..7, half_r=1: rows 8..15)
    const int head_slot = kvh * NQT + qt;
    auto write_final_row = [&](int r_local, float fm, float fl, float4 acc) {
        const int qr = q_base + r_local;
        if (qr >= M_Q) return;
        const float sink = s_sink[r_local];
        float inv_l = 0.0f;
        if (fl > 0.0f) {
            const float nm     = fmaxf(fm, sink);
            const float e_old  = expf(fm - nm);
            const float e_sink = expf(sink - nm);
            inv_l = e_old / (fl * e_old + e_sink);
        }
        const int tok = qr / 3;
        const int qh  = kvh * 3 + (qr - tok * 3);
        float4 res = make_float4(acc.x * inv_l, acc.y * inv_l, acc.z * inv_l, acc.w * inv_l);
        ((float4 *) (dst_f32 + ((size_t) tok * 24 + qh) * 128))[lane] = res;
    };

    for (int half_r = 0; half_r < 2; ++half_r) {
        __syncthreads();
        const float lm = (half_r == 0) ? m0 : m1;
        const float ll = (half_r == 0) ? l0 : l1;
        float * wr_ptr = &wsmem.s_red[warp][gid][0];
#pragma unroll
        for (int dn = 0; dn < 16; ++dn) {
            const int c = dn * 8 + tid4 * 2;
            wr_ptr[c + 0] = (half_r == 0) ? o[dn][0] : o[dn][2];
            wr_ptr[c + 1] = (half_r == 0) ? o[dn][1] : o[dn][3];
        }
        if (tid4 == 0) {
            wr_ptr[128] = lm;
            wr_ptr[129] = ll;
        }
        __syncthreads();

        // Each warp reduces 2 rows (warp*2 + 0..1) across the 4 warps w=0..3
        for (int sub = 0; sub < 2; ++sub) {
            const int r8 = warp * 2 + sub; // 0..7
            const int r_local = half_r * 8 + r8;
            if (q_base + r_local >= M_Q) continue;
            float fm = -1e30f;
            for (int w = 0; w < 4; ++w) {
                if (wsmem.s_red[w][r8][129] > 0.0f) {
                    fm = fmaxf(fm, wsmem.s_red[w][r8][128]);
                }
            }
            float fl = 0.0f;
            float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (int w = 0; w < 4; ++w) {
                const float wl = wsmem.s_red[w][r8][129];
                if (wl <= 0.0f) continue;
                const float es = expf(wsmem.s_red[w][r8][128] - fm);
                fl += wl * es;
                const float4 wo = ((const float4 *) &wsmem.s_red[w][r8][0])[lane];
                acc.x += wo.x * es; acc.y += wo.y * es;
                acc.z += wo.z * es; acc.w += wo.w * es;
            }
            if (n_splits == 1) {
                write_final_row(r_local, fm, fl, acc);
            } else {
                float * sp_ptr = scratch + ((size_t) (head_slot * MAX_SPLITS + sp) * 16 + r_local) * 132;
                if (fl > 0.0f) {
                    ((float4 *) sp_ptr)[lane] = acc;
                }
                if (lane == 0) {
                    sp_ptr[128] = fm;
                    sp_ptr[129] = fl;
                }
            }
        }
    }

    if (n_splits == 1) {
        return;
    }

    __syncthreads();
    if (tid == 0) {
        __threadfence();
        const int done = atomicAdd(&cnt[head_slot], 1) + 1;
        is_last = (done == n_splits);
        if (is_last) {
            cnt[head_slot] = 0;
        }
    }
    __syncthreads();
    if (!is_last) {
        return;
    }

    // Last split reduces across s = 0..n_splits-1 (4 rows per warp)
    for (int sub = 0; sub < 4; ++sub) {
        const int r_local = warp * 4 + sub;
        if (q_base + r_local >= M_Q) continue;
        float fm = -1e30f;
        for (int s = 0; s < n_splits; ++s) {
            const float * sp_p = scratch + ((size_t) (head_slot * MAX_SPLITS + s) * 16 + r_local) * 132;
            if (sp_p[129] > 0.0f) {
                fm = fmaxf(fm, sp_p[128]);
            }
        }
        float fl = 0.0f;
        float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int s = 0; s < n_splits; ++s) {
            const float * sp_p = scratch + ((size_t) (head_slot * MAX_SPLITS + s) * 16 + r_local) * 132;
            const float sl = sp_p[129];
            if (sl <= 0.0f) continue;
            const float es = expf(sp_p[128] - fm);
            fl += sl * es;
            const float4 so = ((const float4 *) sp_p)[lane];
            acc.x += so.x * es; acc.y += so.y * es;
            acc.z += so.z * es; acc.w += so.w * es;
        }
        write_final_row(r_local, fm, fl, acc);
    }
#endif
}

template <bool K_Q8, bool V_Q8, bool MASK_F16, int MAX_SPLITS>
static inline void inkling_launch_rel_pos_fattn_batched_mma(
        const int T,
        const dim3 grid,
        cudaStream_t stream,
        const float * Q, const char * K_b, const char * V_b,
        const float * R, const float * rel_bias, const float * q_sinks,
        const int32_t * k_pos, const int32_t * q_pos, const void * raw_mask,
        float * dst_f32, float * scratch, int32_t * cnt,
        int n_kv, int n_extent,
        size_t q_tok_f, size_t q_head_f,
        size_t k_head_b, size_t k_tok_b,
        size_t v_head_b, size_t v_tok_b,
        float kq_scale, int n_splits) {
    const dim3 blk(32, 4, 1);
    switch (T) {
        case 2: inkling_rel_pos_fattn_batched_mma_kpos<K_Q8, V_Q8, MASK_F16, 2, MAX_SPLITS><<<grid, blk, 0, stream>>>(Q, K_b, V_b, R, rel_bias, q_sinks, k_pos, q_pos, raw_mask, dst_f32, scratch, cnt, n_kv, n_extent, q_tok_f, q_head_f, k_head_b, k_tok_b, v_head_b, v_tok_b, kq_scale, n_splits); break;
        case 3: inkling_rel_pos_fattn_batched_mma_kpos<K_Q8, V_Q8, MASK_F16, 3, MAX_SPLITS><<<grid, blk, 0, stream>>>(Q, K_b, V_b, R, rel_bias, q_sinks, k_pos, q_pos, raw_mask, dst_f32, scratch, cnt, n_kv, n_extent, q_tok_f, q_head_f, k_head_b, k_tok_b, v_head_b, v_tok_b, kq_scale, n_splits); break;
        case 4: inkling_rel_pos_fattn_batched_mma_kpos<K_Q8, V_Q8, MASK_F16, 4, MAX_SPLITS><<<grid, blk, 0, stream>>>(Q, K_b, V_b, R, rel_bias, q_sinks, k_pos, q_pos, raw_mask, dst_f32, scratch, cnt, n_kv, n_extent, q_tok_f, q_head_f, k_head_b, k_tok_b, v_head_b, v_tok_b, kq_scale, n_splits); break;
        case 5: inkling_rel_pos_fattn_batched_mma_kpos<K_Q8, V_Q8, MASK_F16, 5, MAX_SPLITS><<<grid, blk, 0, stream>>>(Q, K_b, V_b, R, rel_bias, q_sinks, k_pos, q_pos, raw_mask, dst_f32, scratch, cnt, n_kv, n_extent, q_tok_f, q_head_f, k_head_b, k_tok_b, v_head_b, v_tok_b, kq_scale, n_splits); break;
        case 6: inkling_rel_pos_fattn_batched_mma_kpos<K_Q8, V_Q8, MASK_F16, 6, MAX_SPLITS><<<grid, blk, 0, stream>>>(Q, K_b, V_b, R, rel_bias, q_sinks, k_pos, q_pos, raw_mask, dst_f32, scratch, cnt, n_kv, n_extent, q_tok_f, q_head_f, k_head_b, k_tok_b, v_head_b, v_tok_b, kq_scale, n_splits); break;
        case 7: inkling_rel_pos_fattn_batched_mma_kpos<K_Q8, V_Q8, MASK_F16, 7, MAX_SPLITS><<<grid, blk, 0, stream>>>(Q, K_b, V_b, R, rel_bias, q_sinks, k_pos, q_pos, raw_mask, dst_f32, scratch, cnt, n_kv, n_extent, q_tok_f, q_head_f, k_head_b, k_tok_b, v_head_b, v_tok_b, kq_scale, n_splits); break;
        case 8: inkling_rel_pos_fattn_batched_mma_kpos<K_Q8, V_Q8, MASK_F16, 8, MAX_SPLITS><<<grid, blk, 0, stream>>>(Q, K_b, V_b, R, rel_bias, q_sinks, k_pos, q_pos, raw_mask, dst_f32, scratch, cnt, n_kv, n_extent, q_tok_f, q_head_f, k_head_b, k_tok_b, v_head_b, v_tok_b, kq_scale, n_splits); break;
    }
}

template <bool K_Q8, bool V_Q8, bool MASK_F16, int MAX_SPLITS>
static inline void inkling_launch_rel_pos_fattn_batched(
        const int T,
        const dim3 grid,
        cudaStream_t stream,
        const float * Q, const char * K_b, const char * V_b,
        const float * R, const float * rel_bias, const float * q_sinks,
        const int32_t * k_pos, const int32_t * q_pos, const void * raw_mask,
        float * dst_f32, float * scratch, int32_t * cnt,
        int n_kv, int n_extent,
        size_t q_tok_f, size_t q_head_f,
        size_t k_head_b, size_t k_tok_b,
        size_t v_head_b, size_t v_tok_b,
        float kq_scale, int n_splits) {
    const dim3 blk(32, 3 * T, 1);
    switch (T) {
        case 2: inkling_rel_pos_fattn_batched_kpos<K_Q8, V_Q8, MASK_F16, 2, MAX_SPLITS><<<grid, blk, 0, stream>>>(Q, K_b, V_b, R, rel_bias, q_sinks, k_pos, q_pos, raw_mask, dst_f32, scratch, cnt, n_kv, n_extent, q_tok_f, q_head_f, k_head_b, k_tok_b, v_head_b, v_tok_b, kq_scale, n_splits); break;
        case 3: inkling_rel_pos_fattn_batched_kpos<K_Q8, V_Q8, MASK_F16, 3, MAX_SPLITS><<<grid, blk, 0, stream>>>(Q, K_b, V_b, R, rel_bias, q_sinks, k_pos, q_pos, raw_mask, dst_f32, scratch, cnt, n_kv, n_extent, q_tok_f, q_head_f, k_head_b, k_tok_b, v_head_b, v_tok_b, kq_scale, n_splits); break;
        case 4: inkling_rel_pos_fattn_batched_kpos<K_Q8, V_Q8, MASK_F16, 4, MAX_SPLITS><<<grid, blk, 0, stream>>>(Q, K_b, V_b, R, rel_bias, q_sinks, k_pos, q_pos, raw_mask, dst_f32, scratch, cnt, n_kv, n_extent, q_tok_f, q_head_f, k_head_b, k_tok_b, v_head_b, v_tok_b, kq_scale, n_splits); break;
        case 5: inkling_rel_pos_fattn_batched_kpos<K_Q8, V_Q8, MASK_F16, 5, MAX_SPLITS><<<grid, blk, 0, stream>>>(Q, K_b, V_b, R, rel_bias, q_sinks, k_pos, q_pos, raw_mask, dst_f32, scratch, cnt, n_kv, n_extent, q_tok_f, q_head_f, k_head_b, k_tok_b, v_head_b, v_tok_b, kq_scale, n_splits); break;
        case 6: inkling_rel_pos_fattn_batched_kpos<K_Q8, V_Q8, MASK_F16, 6, MAX_SPLITS><<<grid, blk, 0, stream>>>(Q, K_b, V_b, R, rel_bias, q_sinks, k_pos, q_pos, raw_mask, dst_f32, scratch, cnt, n_kv, n_extent, q_tok_f, q_head_f, k_head_b, k_tok_b, v_head_b, v_tok_b, kq_scale, n_splits); break;
        case 7: inkling_rel_pos_fattn_batched_kpos<K_Q8, V_Q8, MASK_F16, 7, MAX_SPLITS><<<grid, blk, 0, stream>>>(Q, K_b, V_b, R, rel_bias, q_sinks, k_pos, q_pos, raw_mask, dst_f32, scratch, cnt, n_kv, n_extent, q_tok_f, q_head_f, k_head_b, k_tok_b, v_head_b, v_tok_b, kq_scale, n_splits); break;
        case 8: inkling_rel_pos_fattn_batched_kpos<K_Q8, V_Q8, MASK_F16, 8, MAX_SPLITS><<<grid, blk, 0, stream>>>(Q, K_b, V_b, R, rel_bias, q_sinks, k_pos, q_pos, raw_mask, dst_f32, scratch, cnt, n_kv, n_extent, q_tok_f, q_head_f, k_head_b, k_tok_b, v_head_b, v_tok_b, kq_scale, n_splits); break;
    }
}

// Unified Wave 9A/9B RelPos Flash Attention dispatcher for GGML_OP_FLASH_ATTN_EXT with src[6..9]
static inline void ggml_cuda_op_inkling_rel_pos_fattn_ext(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        const ggml_tensor * Gcur = nullptr,
        const ggml_tensor * wo   = nullptr,
        ggml_tensor * wo_dst     = nullptr) {
    const ggml_tensor * Q        = dst->src[0];
    const ggml_tensor * K        = dst->src[1];
    const ggml_tensor * V        = dst->src[2];
    const ggml_tensor * raw_mask = dst->src[3];
    const ggml_tensor * q_sinks  = dst->src[4];
    const ggml_tensor * R        = dst->src[6];
    const ggml_tensor * rel_bias = dst->src[7];
    const ggml_tensor * k_pos    = dst->src[8];
    const ggml_tensor * q_pos    = dst->src[9];

    const float kq_scale = ggml_get_op_params_f32(dst, 0);
    const int T_s        = (int) Q->ne[1];
    const int S          = (int) Q->ne[3];
    const int n_kv       = (int) K->ne[1];
    const int n_extent   = (int) rel_bias->ne[1];

    const bool k_q8     = (K->type == GGML_TYPE_Q8_0);
    const bool v_q8     = (V->type == GGML_TYPE_Q8_0);
    const bool mask_f16 = (raw_mask->type == GGML_TYPE_F16);
    const bool fuse_wo  = (Gcur != nullptr && wo != nullptr && wo_dst != nullptr && T_s == 1 && S == 1);

    const size_t q_tok_stride_f    = Q->nb[1] / sizeof(float);
    const size_t q_head_stride_f   = Q->nb[2] / sizeof(float);
    const size_t q_stream_stride_f = Q->nb[3] / sizeof(float);
    const size_t k_tok_stride_b    = K->nb[1];
    const size_t k_head_stride_b   = K->nb[2];
    const size_t k_stream_stride_b = K->nb[3];
    const size_t v_tok_stride_b    = V->nb[1];
    const size_t v_head_stride_b   = V->nb[2];
    const size_t v_stream_stride_b = V->nb[3];

    cudaStream_t stream = ctx.stream();
    const int kv_mode = (k_q8 ? 4 : 0) | (v_q8 ? 2 : 0) | (mask_f16 ? 1 : 0);

    // Path 1: T == 1, S == 1 (8-warp 0-SMEM register-streaming decode, 82-SM wave-aligned n_splits!)
    if (T_s == 1 && S == 1) {
        constexpr int MAX_SPLITS = 64;
        const int max_useful = (n_kv + 15) / 16;
        int n_splits = (n_kv <= 512) ? 10 : ((n_kv <= 2048) ? 20 : ((n_kv <= 16384) ? 30 : 40));
        if (n_splits > max_useful) n_splits = max_useful;
        if (n_splits < 1) n_splits = 1;

        ggml_cuda_pool_alloc<block_q8_1> y_q8(ctx.pool());
        block_q8_1 * y_q8_ptr = nullptr;
        int nrows = 0, nb = 0;
        if (fuse_wo) {
            nrows = (int) wo->ne[1];
            nb    = (int) (wo->ne[0] / QK8_0);
            y_q8.alloc((size_t) nb);
            y_q8_ptr = y_q8.get();
        }
        ggml_cuda_pool_alloc<float> scratch(ctx.pool());
        float * scratch_ptr = nullptr;
        if (n_splits > 1) {
            scratch.alloc((size_t) 8 * MAX_SPLITS * 3 * 132);
            scratch_ptr = scratch.get();
        }
        int32_t * d_cnt = inkling_get_fattn_counters(ctx.device, ctx.curr_stream_no);
        const dim3 fa_grid(8, n_splits, 1);
        const dim3 fa_block(32, 8, 1);

#define LAUNCH_T1(KQ, VQ, MF, FW) \
        inkling_rel_pos_fattn_t1_kpos<KQ, VQ, MF, FW, MAX_SPLITS><<<fa_grid, fa_block, 0, stream>>>( \
            (const float *) Q->data, (const char *) K->data, (const char *) V->data, \
            (const float *) R->data, (const float *) rel_bias->data, (const float *) q_sinks->data, \
            (const int32_t *) k_pos->data, (const int32_t *) q_pos->data, raw_mask->data, \
            Gcur ? (const float *) Gcur->data : nullptr, y_q8_ptr, (float *) dst->data, \
            scratch_ptr, d_cnt, n_kv, n_extent, \
            q_head_stride_f, k_head_stride_b, k_tok_stride_b, v_head_stride_b, v_tok_stride_b, \
            kq_scale, n_splits)

        if (fuse_wo) {
            switch (kv_mode) {
                case 0: LAUNCH_T1(false, false, false, true); break;
                case 1: LAUNCH_T1(false, false, true,  true); break;
                case 2: LAUNCH_T1(false, true,  false, true); break;
                case 3: LAUNCH_T1(false, true,  true,  true); break;
                case 4: LAUNCH_T1(true,  false, false, true); break;
                case 5: LAUNCH_T1(true,  false, true,  true); break;
                case 6: LAUNCH_T1(true,  true,  false, true); break;
                case 7: LAUNCH_T1(true,  true,  true,  true); break;
            }
            constexpr int NWARPS = 4;
            const dim3 wo_block(32, NWARPS, 1);
            const dim3 wo_grid((nrows + NWARPS - 1) / NWARPS, 1, 1);
            if (nb == 96 && (((uintptr_t) wo->data) & 3) == 0) {
                inkling_q8_0_mmv_align32_f32<NWARPS, 96><<<wo_grid, wo_block, 0, stream>>>(
                    (const block_q8_0 *) wo->data, y_q8_ptr, (float *) wo_dst->data, nrows, 96);
            } else if (nb == 96) {
                inkling_q8_0_mmv_f32<NWARPS, 96><<<wo_grid, wo_block, 0, stream>>>(
                    (const block_q8_0 *) wo->data, y_q8_ptr, (float *) wo_dst->data, nrows, 96);
            } else {
                inkling_q8_0_mmv_f32<NWARPS, 0><<<wo_grid, wo_block, 0, stream>>>(
                    (const block_q8_0 *) wo->data, y_q8_ptr, (float *) wo_dst->data, nrows, nb);
            }
        } else {
            switch (kv_mode) {
                case 0: LAUNCH_T1(false, false, false, false); break;
                case 1: LAUNCH_T1(false, false, true,  false); break;
                case 2: LAUNCH_T1(false, true,  false, false); break;
                case 3: LAUNCH_T1(false, true,  true,  false); break;
                case 4: LAUNCH_T1(true,  false, false, false); break;
                case 5: LAUNCH_T1(true,  false, true,  false); break;
                case 6: LAUNCH_T1(true,  true,  false, false); break;
                case 7: LAUNCH_T1(true,  true,  true,  false); break;
            }
        }
#undef LAUNCH_T1
        return;
    }

    // Path 2: T = 2..8, S == 1 (speculative verification)
    if (T_s >= 2 && T_s <= 8 && S == 1) {
        static const bool w9_no_batched_mma = (getenv("W9_NO_BATCHED_MMA") != nullptr);
        if (!w9_no_batched_mma) {
            constexpr int MAX_SPLITS = 32;
            const int nqt        = (3 * T_s + 15) / 16; // 1 for T=2..5, 2 for T=6..8
            const int n_tiles    = (n_kv + 15) / 16;
            const int max_useful = (n_tiles + 3) / 4;
            int n_splits;
            if (nqt == 1) {
                n_splits = (n_kv <= 512) ? 10 : ((n_kv <= 2048) ? 20 : 30);
            } else {
                n_splits = (n_kv <= 512) ? 5 : ((n_kv <= 2048) ? 10 : 15);
            }
            if (n_splits > max_useful) n_splits = max_useful;
            if (n_splits < 1) n_splits = 1;

            ggml_cuda_pool_alloc<float> scratch(ctx.pool());
            float * scratch_ptr = nullptr;
            if (n_splits > 1) {
                scratch.alloc((size_t) 8 * nqt * MAX_SPLITS * 16 * 132);
                scratch_ptr = scratch.get();
            }
            int32_t * d_cnt = inkling_get_fattn_counters(ctx.device, ctx.curr_stream_no);
            const dim3 fa_grid(8, n_splits, nqt);

#define LAUNCH_TBM(KQ, VQ, MF) \
            inkling_launch_rel_pos_fattn_batched_mma<KQ, VQ, MF, MAX_SPLITS>( \
                T_s, fa_grid, stream, \
                (const float *) Q->data, (const char *) K->data, (const char *) V->data, \
                (const float *) R->data, (const float *) rel_bias->data, (const float *) q_sinks->data, \
                (const int32_t *) k_pos->data, (const int32_t *) q_pos->data, raw_mask->data, \
                (float *) dst->data, scratch_ptr, d_cnt, n_kv, n_extent, \
                q_tok_stride_f, q_head_stride_f, k_head_stride_b, k_tok_stride_b, v_head_stride_b, v_tok_stride_b, \
                kq_scale, n_splits)

            switch (kv_mode) {
                case 0: LAUNCH_TBM(false, false, false); break;
                case 1: LAUNCH_TBM(false, false, true);  break;
                case 2: LAUNCH_TBM(false, true,  false); break;
                case 3: LAUNCH_TBM(false, true,  true);  break;
                case 4: LAUNCH_TBM(true,  false, false); break;
                case 5: LAUNCH_TBM(true,  false, true);  break;
                case 6: LAUNCH_TBM(true,  true,  false); break;
                case 7: LAUNCH_TBM(true,  true,  true);  break;
            }
#undef LAUNCH_TBM
            return;
        }

        constexpr int MAX_SPLITS = 48;
        const int n_tiles = (n_kv + 15) / 16;
        int n_splits = (n_kv <= 256) ? 10 : ((n_kv <= 2048) ? 20 : ((n_kv <= 16384) ? 20 : ((T_s <= 4) ? 40 : 20)));
        if (n_splits > n_tiles) n_splits = n_tiles;
        if (n_splits < 1) n_splits = 1;

        ggml_cuda_pool_alloc<float> scratch(ctx.pool());
        float * scratch_ptr = nullptr;
        if (n_splits > 1) {
            scratch.alloc((size_t) 8 * MAX_SPLITS * (3 * T_s) * 132);
            scratch_ptr = scratch.get();
        }
        int32_t * d_cnt = inkling_get_fattn_counters(ctx.device, ctx.curr_stream_no);
        const dim3 fa_grid(8, n_splits, 1);

#define LAUNCH_TB(KQ, VQ, MF) \
        inkling_launch_rel_pos_fattn_batched<KQ, VQ, MF, MAX_SPLITS>( \
            T_s, fa_grid, stream, \
            (const float *) Q->data, (const char *) K->data, (const char *) V->data, \
            (const float *) R->data, (const float *) rel_bias->data, (const float *) q_sinks->data, \
            (const int32_t *) k_pos->data, (const int32_t *) q_pos->data, raw_mask->data, \
            (float *) dst->data, scratch_ptr, d_cnt, n_kv, n_extent, \
            q_tok_stride_f, q_head_stride_f, k_head_stride_b, k_tok_stride_b, v_head_stride_b, v_tok_stride_b, \
            kq_scale, n_splits)

        switch (kv_mode) {
            case 0: LAUNCH_TB(false, false, false); break;
            case 1: LAUNCH_TB(false, false, true);  break;
            case 2: LAUNCH_TB(false, true,  false); break;
            case 3: LAUNCH_TB(false, true,  true);  break;
            case 4: LAUNCH_TB(true,  false, false); break;
            case 5: LAUNCH_TB(true,  false, true);  break;
            case 6: LAUNCH_TB(true,  true,  false); break;
            case 7: LAUNCH_TB(true,  true,  true);  break;
        }
#undef LAUNCH_TB
        return;
    }

    // Path 3: Prefill (T_s > 8) or multi-stream (S > 1) — SM 8.6 Tensor Core MMA
    const dim3 pre_grid(24, (T_s + 63) / 64, S);
    const dim3 pre_block(32, 4, 1);
#define LAUNCH_PRE(KQ, VQ, MF) \
    inkling_rel_pos_fattn_prefill_mma_kpos<KQ, VQ, MF><<<pre_grid, pre_block, 0, stream>>>( \
        (const float *) Q->data, (const char *) K->data, (const char *) V->data, \
        (const float *) R->data, (const float *) rel_bias->data, (const float *) q_sinks->data, \
        (const int32_t *) k_pos->data, (const int32_t *) q_pos->data, raw_mask->data, \
        (float *) dst->data, T_s, n_kv, n_extent, \
        q_tok_stride_f, q_head_stride_f, q_stream_stride_f, \
        k_head_stride_b, k_tok_stride_b, k_stream_stride_b, \
        v_head_stride_b, v_tok_stride_b, v_stream_stride_b, \
        kq_scale)

    switch (kv_mode) {
        case 0: LAUNCH_PRE(false, false, false); break;
        case 1: LAUNCH_PRE(false, false, true);  break;
        case 2: LAUNCH_PRE(false, true,  false); break;
        case 3: LAUNCH_PRE(false, true,  true);  break;
        case 4: LAUNCH_PRE(true,  false, false); break;
        case 5: LAUNCH_PRE(true,  false, true);  break;
        case 6: LAUNCH_PRE(true,  true,  false); break;
        case 7: LAUNCH_PRE(true,  true,  true);  break;
    }
#undef LAUNCH_PRE
}
