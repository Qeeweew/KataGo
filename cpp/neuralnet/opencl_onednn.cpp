#include <CL/cl.h>
#include <CL/cl_platform.h>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_ocl.hpp>
#include <iostream>
#include <vector>
#include <memory>
#include <mutex>

#include "../neuralnet/opencl_onednn.h"

namespace OneDNNHelpers {

using namespace dnnl;

using tag = memory::format_tag;
using dt = memory::data_type;

template <bool transA, bool transB, bool transC, bool use_f16, bool broadcast_weight = false>
cl_int gemm_batched_impl(
    cl_command_queue commandQueue,
    cl_mem A, cl_mem B, cl_mem C,
    int M, int N, int K,
    int numBatchElts,
    cl_event* event_buf
) {
    cl_context oclContext;
    cl_device_id oclDevice;
    clGetCommandQueueInfo(commandQueue, CL_QUEUE_CONTEXT, sizeof(cl_context), &oclContext, nullptr);
    clGetCommandQueueInfo(commandQueue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &oclDevice, nullptr);

    engine eng = ocl_interop::make_engine(oclDevice, oclContext);
    stream strm = ocl_interop::make_stream(eng, commandQueue);

    constexpr memory::data_type dtype = use_f16 ? dt::f16 : dt::f32;
    memory::dims a_dims = {numBatchElts, M, K};
    memory::dims b_dims = {broadcast_weight ? 1 : numBatchElts, K, N};
    memory::dims c_dims = {numBatchElts, M, N};

    auto a_md = memory::desc(a_dims, dtype, transA ? tag::acb : tag::abc);
    auto b_md = memory::desc(b_dims, dtype, transB ? tag::acb : tag::abc);
    auto c_md = memory::desc(c_dims, dtype, transC ? tag::acb : tag::abc);

    memory mem_A = ocl_interop::make_memory(a_md, eng, A);
    memory mem_B = ocl_interop::make_memory(b_md, eng, B);
    memory mem_C = ocl_interop::make_memory(c_md, eng, C);

    primitive_attr attr;
    auto matmul_pd = matmul::primitive_desc(eng, a_md, b_md, c_md, attr);
    auto matmul_prim = matmul(matmul_pd);

    cl_event event = ocl_interop::execute(matmul_prim, strm, 
        {{DNNL_ARG_SRC, mem_A},
            {DNNL_ARG_WEIGHTS, mem_B},
            {DNNL_ARG_DST, mem_C}});
    
    if (event_buf) *event_buf = event;
    return CL_SUCCESS;
}

} // namespace OneDNNHelpers

template<bool transA, bool transB, bool transC, bool use_f16>
cl_int OneDNNHelpers::doBatchedXGemm(
  cl_command_queue commandQueue,
  cl_mem A, cl_mem B, cl_mem C,
  int M, int N, int K,
  int numBatchElts,
  cl_event* event_buf) {
  return gemm_batched_impl<transA, transB, transC, use_f16>(commandQueue, A, B, C, M, N, K, numBatchElts, event_buf);
}

template<bool use_f16>
cl_int OneDNNHelpers::doBatchedXGemm1x1Conv(
  cl_command_queue commandQueue,
  cl_mem A, cl_mem B, cl_mem C,
  int M, int N, int K,
  int numBatchElts,
  cl_event* event_buf) {
  return gemm_batched_impl<true, false, true, use_f16, true>(commandQueue, A, B, C, M, N, K, numBatchElts, event_buf);
}

// 显式实例化
template cl_int OneDNNHelpers::doBatchedXGemm<true, false, true, false>(
    cl_command_queue commandQueue,
    cl_mem A, cl_mem B, cl_mem C,
    int M, int N, int K,
    int numBatchElts,
    cl_event* event_buf);

template cl_int OneDNNHelpers::doBatchedXGemm<true, false, true, true>(
    cl_command_queue commandQueue,
    cl_mem A, cl_mem B, cl_mem C,
    int M, int N, int K,
    int numBatchElts,
    cl_event* event_buf);

template cl_int OneDNNHelpers::doBatchedXGemm1x1Conv<true>(
  cl_command_queue commandQueue,
  cl_mem A, cl_mem B, cl_mem C,
  int M, int N, int K,
  int numBatchElts,
  cl_event* event_buf);

template cl_int OneDNNHelpers::doBatchedXGemm1x1Conv<false>(
  cl_command_queue commandQueue,
  cl_mem A, cl_mem B, cl_mem C,
  int M, int N, int K,
  int numBatchElts,
  cl_event* event_buf);