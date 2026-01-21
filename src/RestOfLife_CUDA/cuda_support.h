#ifndef CUDA_SUPPORT_H
#define CUDA_SUPPORT_H

// Central CUDA runtime include used throughout the CUDA port
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

inline void cuda_check(cudaError_t result, const char *file, int line)
{
  if (result != cudaSuccess)
  {
    fprintf(stderr, "CUDA error %d at %s:%d -> %s\n",
            static_cast<int>(result), file, line,
            cudaGetErrorString(result));
    std::abort();
  }
}

#define CUDA_CHECK(expr) cuda_check((expr), __FILE__, __LINE__)

#endif // CUDA_SUPPORT_H
