#include "../neuralnet/openclincludes.h"
namespace OneDNNHelpers {
  template<bool transA, bool transB, bool transC, bool use_f16>
  cl_int doBatchedXGemm(
    cl_command_queue commandQueue,
    cl_mem A, cl_mem B, cl_mem C,
    int M, int N, int K,
    int numBatchElts,
    cl_event* event_buf
  );
}