#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, float* __restrict__ C);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, float* __restrict__ C) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  void* A_shared = ((void*)((char*)buf_dyn_shmem + 0));
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  void* B_shared = ((void*)((char*)buf_dyn_shmem + 16384));
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  float C_local[128];
#line 21 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  for (int w = 0; w < 16; ++w) {
#line 21 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    if (((w * 33) + (((int)blockIdx.x) >> 1)) < 512) {
#line 30 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
      #pragma unroll
#line 30 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
      for (int i = 0; i < 32; ++i) {
#line 30 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
        float broadcast_var = 0x0p+0f/*0.000000e+00*/;
#line 30 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
        *(float4*)(C_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
#line 30 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
      }
#line 33 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
      for (int ko = 0; ko < 64; ++ko) {
#line 33 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
        __syncthreads();
#line 35 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
        #pragma unroll
#line 35 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
        for (int i_1 = 0; i_1 < 8; ++i_1) {
#line 35 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
          *(uint4*)(((half_t*)A_shared) + (((((i_1 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(A + (((((((((w * 66) + ((int)blockIdx.x)) % 256) / 8) * 524288) + (i_1 * 65536)) + ((((int)threadIdx.x) >> 3) * 4096)) + (ko * 64)) + ((((int)threadIdx.x) & 7) * 8)));
#line 35 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
        }
#line 38 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
        #pragma unroll
#line 38 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
        for (int i_2 = 0; i_2 < 8; ++i_2) {
#line 38 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
          *(uint4*)(((half_t*)B_shared) + (((((i_2 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(B + (((((((((w * 66) + ((int)blockIdx.x)) / 256) * 4194304) + ((((w * 66) + ((int)blockIdx.x)) % 8) * 524288)) + (i_2 * 65536)) + ((((int)threadIdx.x) >> 3) * 4096)) + (ko * 64)) + ((((int)threadIdx.x) & 7) * 8)));
#line 38 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
        }
#line 33 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
        {
#line 33 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
          half_t A_local[32];
#line 33 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
          half_t B_local[32];
#line 33 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
          __syncthreads();
#line 121 "/home/kael/dev/src/tilelang/tilelang/cuda/op/gemm/gemm_mma.py"
          for (int ki = 0; ki < 4; ++ki) {
#line 376 "/home/kael/dev/src/tilelang/tilelang/cuda/intrinsics/macro/mma_macro_generator.py"
            #pragma unroll
#line 376 "/home/kael/dev/src/tilelang/tilelang/cuda/intrinsics/macro/mma_macro_generator.py"
            for (int i_3 = 0; i_3 < 4; ++i_3) {
#line 387 "/home/kael/dev/src/tilelang/tilelang/cuda/intrinsics/macro/mma_macro_generator.py"
              tl::ptx_ldmatrix_x4((&(((half_t*)A_shared)[((((((((((int)threadIdx.x) & 63) >> 5) * 4096) + (i_3 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local[(i_3 * 8)])));
#line 376 "/home/kael/dev/src/tilelang/tilelang/cuda/intrinsics/macro/mma_macro_generator.py"
            }
#line 491 "/home/kael/dev/src/tilelang/tilelang/cuda/intrinsics/macro/mma_macro_generator.py"
            #pragma unroll
#line 491 "/home/kael/dev/src/tilelang/tilelang/cuda/intrinsics/macro/mma_macro_generator.py"
            for (int i_4 = 0; i_4 < 4; ++i_4) {
#line 505 "/home/kael/dev/src/tilelang/tilelang/cuda/intrinsics/macro/mma_macro_generator.py"
              tl::ptx_ldmatrix_x4((&(((half_t*)B_shared)[((((((((((int)threadIdx.x) >> 6) * 4096) + (i_4 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[(i_4 * 8)])));
#line 491 "/home/kael/dev/src/tilelang/tilelang/cuda/intrinsics/macro/mma_macro_generator.py"
            }
#line 530 "/home/kael/dev/src/tilelang/tilelang/cuda/intrinsics/macro/mma_macro_generator.py"
            for (int i_5 = 0; i_5 < 4; ++i_5) {
#line 530 "/home/kael/dev/src/tilelang/tilelang/cuda/intrinsics/macro/mma_macro_generator.py"
              for (int j = 0; j < 4; ++j) {
#line 600 "/home/kael/dev/src/tilelang/tilelang/cuda/intrinsics/macro/mma_macro_generator.py"
                tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((i_5 * 32) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_5 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
#line 617 "/home/kael/dev/src/tilelang/tilelang/cuda/intrinsics/macro/mma_macro_generator.py"
                tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (((i_5 * 32) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_5 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
#line 530 "/home/kael/dev/src/tilelang/tilelang/cuda/intrinsics/macro/mma_macro_generator.py"
              }
#line 530 "/home/kael/dev/src/tilelang/tilelang/cuda/intrinsics/macro/mma_macro_generator.py"
            }
#line 121 "/home/kael/dev/src/tilelang/tilelang/cuda/op/gemm/gemm_mma.py"
          }
#line 33 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
        }
#line 33 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
      }
#line 43 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
      #pragma unroll
#line 43 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
      for (int i_6 = 0; i_6 < 64; ++i_6) {
#line 43 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
        *(float2*)(C + ((((((((((((((w * 66) + ((int)blockIdx.x)) % 256) / 8) * 524288) + (((((int)threadIdx.x) & 63) >> 5) * 262144)) + ((i_6 >> 4) * 65536)) + ((i_6 & 1) * 32768)) + (((((int)threadIdx.x) & 31) >> 2) * 4096)) + ((((w * 66) + ((int)blockIdx.x)) / 256) * 1024)) + ((((w * 66) + ((int)blockIdx.x)) % 8) * 128)) + ((((int)threadIdx.x) >> 6) * 64)) + (((i_6 & 15) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(C_local + (i_6 * 2));
#line 43 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
      }
#line 21 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    }
#line 21 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
}

