// tilelang target: {"kind":"c","tag":"","keys":["cpu"]}
#define TVM_EXPORTS
#include "tvm/runtime/base.h"
#include "tvm/runtime/c_backend_api.h"
#include "tvm/ffi/c_api.h"
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#if defined(_MSC_VER)
#define TL_ALIGN(N) __declspec(align(N))
#else
#define TL_ALIGN(N) __attribute__((aligned(N)))
#endif
#ifdef __OBJC__
#include "tvm/runtime/device_api.h"
#include "tvm/ffi/function.h"
#include <Metal/Metal.h>
#include <Foundation/Foundation.h>
#include <torch/mps.h>
#endif
void* __tvm_ffi__library_ctx = NULL;
static void* __tvm_set_device_packed = NULL;
static void* main_kernel_packed = NULL;
#ifdef __cplusplus
extern "C"
#endif
int32_t __tvm_ffi_main(void* self_handle, void* args, int32_t num_args, void* result);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
#ifdef __cplusplus
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
extern "C"
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
#endif
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
int32_t __tvm_ffi_main(void* self_handle, void* args, int32_t num_args, void* result) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  TL_ALIGN(128) TVMFFIAny stack[9];
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  void* stack_ffi_any = stack;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if (!((num_args == 3))) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    char __tvm_assert_msg_buf[512];
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "main: num_args should be 3", (long long)(num_args), (long long)(3));
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    return -1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if (!(!(args == NULL))) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main: args pointer is NULL");
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    return -1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  int32_t A_handle_type_index = (((TVMFFIAny*)args)[0].type_index);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if (!(((((A_handle_type_index == 0) || (A_handle_type_index == 4)) || (A_handle_type_index == 7)) || (64 <= A_handle_type_index)))) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input A expected pointer or tensor handle");
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    return -1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  int32_t B_handle_type_index = (((TVMFFIAny*)args)[1].type_index);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if (!(((((B_handle_type_index == 0) || (B_handle_type_index == 4)) || (B_handle_type_index == 7)) || (64 <= B_handle_type_index)))) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input B expected pointer or tensor handle");
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    return -1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  int32_t C_handle_type_index = (((TVMFFIAny*)args)[2].type_index);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if (!(((((C_handle_type_index == 0) || (C_handle_type_index == 4)) || (C_handle_type_index == 7)) || (64 <= C_handle_type_index)))) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input C expected pointer or tensor handle");
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    return -1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  void* A_handle = ((A_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[0].v_ptr) + 24)) : (((TVMFFIAny*)args)[0].v_ptr));
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  void* B_handle = ((B_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[1].v_ptr) + 24)) : (((TVMFFIAny*)args)[1].v_ptr));
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  void* C_handle = ((C_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[2].v_ptr) + 24)) : (((TVMFFIAny*)args)[2].v_ptr));
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  bool main_A_is_null = (A_handle == NULL);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if (!(!main_A_is_null)) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.A is expected to have non-NULL pointer");
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    return -1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  bool main_B_is_null = (B_handle == NULL);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if (!(!main_B_is_null)) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.B is expected to have non-NULL pointer");
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    return -1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  bool main_C_is_null = (C_handle == NULL);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if (!(!main_C_is_null)) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.C is expected to have non-NULL pointer");
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    return -1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  void* main_A_shape = (((DLTensor*)A_handle)[0].shape);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  void* main_B_shape = (((DLTensor*)B_handle)[0].shape);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  void* main_C_shape = (((DLTensor*)C_handle)[0].shape);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if (!(((((DLTensor*)A_handle)[0].ndim) == 2))) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    char __tvm_assert_msg_buf[512];
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input A ndim mismatch, expected 2", (long long)((((DLTensor*)A_handle)[0].ndim)), (long long)(2));
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    return -1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  void* main_A_strides = (((DLTensor*)A_handle)[0].strides);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  int32_t dev_id = (((DLTensor*)A_handle)[0].device.device_id);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  void* A = (((DLTensor*)A_handle)[0].data);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if (!(((((DLTensor*)B_handle)[0].ndim) == 2))) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    char __tvm_assert_msg_buf[512];
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input B ndim mismatch, expected 2", (long long)((((DLTensor*)B_handle)[0].ndim)), (long long)(2));
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    return -1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  void* main_B_strides = (((DLTensor*)B_handle)[0].strides);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  void* B = (((DLTensor*)B_handle)[0].data);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if (!(((((DLTensor*)C_handle)[0].ndim) == 2))) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    char __tvm_assert_msg_buf[512];
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input C ndim mismatch, expected 2", (long long)((((DLTensor*)C_handle)[0].ndim)), (long long)(2));
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    return -1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  void* main_C_strides = (((DLTensor*)C_handle)[0].strides);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  void* C = (((DLTensor*)C_handle)[0].data);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if (!(((((((DLTensor*)A_handle)[0].dtype.code) == (uint8_t)2) && ((((DLTensor*)A_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)A_handle)[0].dtype.lanes) == (uint16_t)1)))) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input A dtype mismatch, expected float16");
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    return -1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if (!((((int32_t)((int64_t*)main_A_shape)[0]) == 4096))) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    char __tvm_assert_msg_buf[512];
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input A shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_A_shape)[0])), (long long)(4096));
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    return -1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if (!((((int32_t)((int64_t*)main_A_shape)[1]) == 4096))) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    char __tvm_assert_msg_buf[512];
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input A shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_A_shape)[1])), (long long)(4096));
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    return -1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  int32_t condval;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if ((main_A_strides == NULL)) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    condval = 1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  } else {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    condval = ((int32_t)((int64_t*)main_A_strides)[1]);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if (!((condval == 1))) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    int32_t condval_1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    if ((main_A_strides == NULL)) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
      condval_1 = 1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    } else {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
      condval_1 = ((int32_t)((int64_t*)main_A_strides)[1]);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    char __tvm_assert_msg_buf[512];
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input A strides[1] violates packed ABI constraint", (long long)(condval_1), (long long)(1));
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    return -1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  int32_t condval_2;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if ((main_A_strides == NULL)) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    condval_2 = 1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  } else {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    condval_2 = ((int32_t)((int64_t*)main_A_strides)[0]);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if (!((condval_2 == 4096))) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    int32_t condval_3;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    if ((main_A_strides == NULL)) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
      condval_3 = 1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    } else {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
      condval_3 = ((int32_t)((int64_t*)main_A_strides)[0]);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    char __tvm_assert_msg_buf[512];
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input A strides[0] violates packed ABI constraint", (long long)(condval_3), (long long)(4096));
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    return -1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if (!(((uint64_t)0 == (((DLTensor*)A_handle)[0].byte_offset)))) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    char __tvm_assert_msg_buf[512];
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input A byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)A_handle)[0].byte_offset)));
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    return -1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if (!(((((DLTensor*)A_handle)[0].device.device_type) == 2))) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    char __tvm_assert_msg_buf[512];
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input A device_type mismatch, expected cuda", (long long)((((DLTensor*)A_handle)[0].device.device_type)), (long long)(2));
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    return -1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if (!(!(A == NULL))) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input A data pointer is NULL");
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    return -1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if (!(((((((DLTensor*)B_handle)[0].dtype.code) == (uint8_t)2) && ((((DLTensor*)B_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)B_handle)[0].dtype.lanes) == (uint16_t)1)))) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input B dtype mismatch, expected float16");
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    return -1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if (!((((int32_t)((int64_t*)main_B_shape)[0]) == 4096))) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    char __tvm_assert_msg_buf[512];
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input B shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_B_shape)[0])), (long long)(4096));
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    return -1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if (!((((int32_t)((int64_t*)main_B_shape)[1]) == 4096))) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    char __tvm_assert_msg_buf[512];
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input B shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_B_shape)[1])), (long long)(4096));
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    return -1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  int32_t condval_4;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if ((main_B_strides == NULL)) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    condval_4 = 1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  } else {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    condval_4 = ((int32_t)((int64_t*)main_B_strides)[1]);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if (!((condval_4 == 1))) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    int32_t condval_5;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    if ((main_B_strides == NULL)) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
      condval_5 = 1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    } else {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
      condval_5 = ((int32_t)((int64_t*)main_B_strides)[1]);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    char __tvm_assert_msg_buf[512];
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input B strides[1] violates packed ABI constraint", (long long)(condval_5), (long long)(1));
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    return -1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  int32_t condval_6;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if ((main_B_strides == NULL)) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    condval_6 = 1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  } else {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    condval_6 = ((int32_t)((int64_t*)main_B_strides)[0]);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if (!((condval_6 == 4096))) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    int32_t condval_7;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    if ((main_B_strides == NULL)) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
      condval_7 = 1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    } else {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
      condval_7 = ((int32_t)((int64_t*)main_B_strides)[0]);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    char __tvm_assert_msg_buf[512];
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input B strides[0] violates packed ABI constraint", (long long)(condval_7), (long long)(4096));
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    return -1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if (!(((uint64_t)0 == (((DLTensor*)B_handle)[0].byte_offset)))) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    char __tvm_assert_msg_buf[512];
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input B byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)B_handle)[0].byte_offset)));
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    return -1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if (!(((((DLTensor*)B_handle)[0].device.device_id) == (((DLTensor*)A_handle)[0].device.device_id)))) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    char __tvm_assert_msg_buf[512];
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input B device_id violates packed ABI constraint", (long long)((((DLTensor*)B_handle)[0].device.device_id)), (long long)((((DLTensor*)A_handle)[0].device.device_id)));
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    return -1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if (!(((((DLTensor*)B_handle)[0].device.device_type) == 2))) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    char __tvm_assert_msg_buf[512];
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input B device_type mismatch, expected cuda", (long long)((((DLTensor*)B_handle)[0].device.device_type)), (long long)(2));
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    return -1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if (!(!(B == NULL))) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input B data pointer is NULL");
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    return -1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if (!(((((((DLTensor*)C_handle)[0].dtype.code) == (uint8_t)2) && ((((DLTensor*)C_handle)[0].dtype.bits) == (uint8_t)32)) && ((((DLTensor*)C_handle)[0].dtype.lanes) == (uint16_t)1)))) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input C dtype mismatch, expected float32");
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    return -1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if (!((((int32_t)((int64_t*)main_C_shape)[0]) == 4096))) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    char __tvm_assert_msg_buf[512];
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input C shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_C_shape)[0])), (long long)(4096));
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    return -1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if (!((((int32_t)((int64_t*)main_C_shape)[1]) == 4096))) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    char __tvm_assert_msg_buf[512];
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input C shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_C_shape)[1])), (long long)(4096));
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    return -1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  int32_t condval_8;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if ((main_C_strides == NULL)) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    condval_8 = 1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  } else {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    condval_8 = ((int32_t)((int64_t*)main_C_strides)[1]);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if (!((condval_8 == 1))) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    int32_t condval_9;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    if ((main_C_strides == NULL)) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
      condval_9 = 1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    } else {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
      condval_9 = ((int32_t)((int64_t*)main_C_strides)[1]);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    char __tvm_assert_msg_buf[512];
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input C strides[1] violates packed ABI constraint", (long long)(condval_9), (long long)(1));
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    return -1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  int32_t condval_10;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if ((main_C_strides == NULL)) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    condval_10 = 1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  } else {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    condval_10 = ((int32_t)((int64_t*)main_C_strides)[0]);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if (!((condval_10 == 4096))) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    int32_t condval_11;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    if ((main_C_strides == NULL)) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
      condval_11 = 1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    } else {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
      condval_11 = ((int32_t)((int64_t*)main_C_strides)[0]);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    char __tvm_assert_msg_buf[512];
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input C strides[0] violates packed ABI constraint", (long long)(condval_11), (long long)(4096));
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    return -1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if (!(((uint64_t)0 == (((DLTensor*)C_handle)[0].byte_offset)))) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    char __tvm_assert_msg_buf[512];
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input C byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)C_handle)[0].byte_offset)));
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    return -1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if (!(((((DLTensor*)C_handle)[0].device.device_id) == (((DLTensor*)A_handle)[0].device.device_id)))) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    char __tvm_assert_msg_buf[512];
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input C device_id violates packed ABI constraint", (long long)((((DLTensor*)C_handle)[0].device.device_id)), (long long)((((DLTensor*)A_handle)[0].device.device_id)));
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    return -1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if (!(((((DLTensor*)C_handle)[0].device.device_type) == 2))) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    char __tvm_assert_msg_buf[512];
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input C device_type mismatch, expected cuda", (long long)((((DLTensor*)C_handle)[0].device.device_type)), (long long)(2));
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    return -1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if (!(!(C == NULL))) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input C data pointer is NULL");
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    return -1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = ((int64_t)2);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = ((int64_t)dev_id);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 0;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)0;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if (__tvm_set_device_packed == NULL) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_set_device", &__tvm_set_device_packed) != 0) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
      return -1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  TVMFFIAny result_1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  result_1.type_index = kTVMFFINone;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  result_1.zero_padding = 0;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  result_1.v_int64 = 0;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if (TVMFFIFunctionCall(__tvm_set_device_packed, (TVMFFIAny*) stack_ffi_any, 2, &result_1) != 0) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    return -1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  ((int32_t)result_1.v_int64);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if (A == NULL) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 0;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  } else {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 4;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = A;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if (B == NULL) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 0;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  } else {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 4;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = B;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if (C == NULL) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 0;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  } else {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 4;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = C;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)66);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)128);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = ((int64_t)1);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = ((int64_t)1);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = ((int64_t)32768);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 0;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = (int64_t)0;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if (main_kernel_packed == NULL) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "main_kernel", &main_kernel_packed) != 0) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
      return -1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  TVMFFIAny result_2;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  result_2.type_index = kTVMFFINone;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  result_2.zero_padding = 0;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  result_2.v_int64 = 0;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  if (TVMFFIFunctionCall(main_kernel_packed, (TVMFFIAny*) stack_ffi_any, 8, &result_2) != 0) {
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
    return -1;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  }
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  ((int32_t)result_2.v_int64);
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
  return 0;
#line 7 "/home/kael/src/github.com/osisos/TIRx-DSL-Demo/tilelang-demo/tilelang_gemm.py"
}

