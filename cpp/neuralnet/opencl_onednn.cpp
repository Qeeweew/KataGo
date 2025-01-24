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

template <bool use_f16>
cl_int conv2dNCHW(
  cl_command_queue commandQueue,
  cl_mem input, cl_mem filter, cl_mem output,
  int batchSize, int inChannels, int outChannels,
  int inputHeight, int inputWidth,
  int filterXRadius, int filterYRadius,
  cl_event* event_buf) {

  cl_int err = CL_SUCCESS;
  cl_context oclContext;
  cl_device_id oclDevice;
  err |= clGetCommandQueueInfo(commandQueue, CL_QUEUE_CONTEXT, sizeof(cl_context), &oclContext, nullptr);
  err |= clGetCommandQueueInfo(commandQueue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &oclDevice, nullptr);
  if (err != CL_SUCCESS) return err;

    // 创建引擎和流
    engine eng = ocl_interop::make_engine(oclDevice, oclContext);
    stream strm = ocl_interop::make_stream(eng, commandQueue);

    // 数据类型确定
    constexpr dt dtype = use_f16 ? dt::f16 : dt::f32;

    // 计算卷积核尺寸
    const int kernel_w = 2 * filterXRadius + 1;
    const int kernel_h = 2 * filterYRadius + 1;

    // 张量维度定义
    memory::dims src_dims = {batchSize, inChannels, inputHeight, inputWidth};
    memory::dims weights_dims = {outChannels, inChannels, kernel_h, kernel_w};
    memory::dims dst_dims = {batchSize, outChannels, inputHeight, inputWidth};

    // 用户内存描述符（NCHW和OIHW格式）
    auto conv_src_md = memory::desc(src_dims, dtype, tag::nchw);
    auto conv_weights_md = memory::desc(weights_dims, dtype, tag::oihw);
    auto conv_dst_md = memory::desc(dst_dims, dtype, tag::nchw);

    // 创建用户内存对象
    memory conv_src_mem = ocl_interop::make_memory(conv_src_md, eng, input);
    memory conv_weights_mem = ocl_interop::make_memory(conv_weights_md, eng, filter);
    memory conv_dst_mem = ocl_interop::make_memory(conv_dst_md, eng, output);

    // 卷积参数
    memory::dims strides = {1, 1};
    memory::dims padding_l = {filterYRadius, filterXRadius};
    memory::dims padding_r = {filterYRadius, filterXRadius};

    // 使用any标签让oneDNN选择最优布局

    // 创建卷积原语描述符
    auto conv_pd = convolution_forward::primitive_desc(
        eng,
        prop_kind::forward_inference,
        algorithm::convolution_direct,
        conv_src_md,
        conv_weights_md,
        conv_dst_md,
        strides,
        padding_l,
        padding_r);

    // 创建并执行卷积原语
    convolution_forward conv_prim(conv_pd);
    cl_event conv_event = ocl_interop::execute(conv_prim, strm,
        {{DNNL_ARG_SRC, conv_src_mem},
          {DNNL_ARG_WEIGHTS, conv_weights_mem},
          {DNNL_ARG_DST, conv_dst_mem}});

    if (event_buf) *event_buf = conv_event;
    return CL_SUCCESS;
}

} // namespace OneDNNHelpers

template<bool use_f16>
cl_int OneDNNHelpers::doBatchedXGemm3x3or5x5Conv(
  cl_command_queue commandQueue,
  cl_mem A, cl_mem B, cl_mem C,
  int M, int N, int K,
  int numBatchElts,
  cl_event* event_buf) {
  return gemm_batched_impl<true,false, true, use_f16>(commandQueue, A, B, C, M, N, K, numBatchElts, event_buf);
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
template cl_int OneDNNHelpers::doBatchedXGemm3x3or5x5Conv<true>(
    cl_command_queue commandQueue,
    cl_mem A, cl_mem B, cl_mem C,
    int M, int N, int K,
    int numBatchElts,
    cl_event* event_buf);

template cl_int OneDNNHelpers::doBatchedXGemm3x3or5x5Conv<false>(
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

template cl_int OneDNNHelpers::conv2dNCHW<true>(
  cl_command_queue commandQueue,
  cl_mem input, cl_mem filter, cl_mem output,
  int batchSize, int inChannels, int outChannels,
  int inputHeight, int inputWidth,
  int filterXRadius, int filterYRadius,
  cl_event* event_buf);

template cl_int OneDNNHelpers::conv2dNCHW<false>(
  cl_command_queue commandQueue,
  cl_mem input, cl_mem filter, cl_mem output,
  int batchSize, int inChannels, int outChannels,
  int inputHeight, int inputWidth,
  int filterXRadius, int filterYRadius,
  cl_event* event_buf);