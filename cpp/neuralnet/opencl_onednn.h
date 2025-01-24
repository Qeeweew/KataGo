#include "../neuralnet/openclincludes.h"
namespace OneDNNHelpers {
  template<bool use_f16>
  cl_int doBatchedXGemm1x1Conv(
    cl_command_queue commandQueue,
    cl_mem A, cl_mem B, cl_mem C,
    int M, int N, int K,
    int numBatchElts,
    cl_event* event_buf
  );

  template<bool use_f16>
  cl_int doBatchedXGemm3x3or5x5Conv(
    cl_command_queue commandQueue,
    cl_mem A, cl_mem B, cl_mem C,
    int M, int N, int K,
    int numBatchElts,
    cl_event* event_buf
  );
  template <bool use_f16>
  cl_int conv2dNCHW(
    cl_command_queue commandQueue,
    cl_mem input, cl_mem filter, cl_mem output,
    int batchSize, int inChannels, int outChannels,
    int inputHeight, int inputWidth,
    int filterXRadius, int filterYRadius,
    cl_event* event_buf);
}