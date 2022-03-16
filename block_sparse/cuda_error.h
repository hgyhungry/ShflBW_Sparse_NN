#pragma once
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>


#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__                             \
                << " of file " << __FILE__ << std::endl;                \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

inline bool isCudaSuccess(cudaError_t status) {
    cudaError_t error = status;
    if (error != cudaSuccess) {
        std::cerr << "Got bad cuda status: " << cudaGetErrorString(error)
              << std::endl;
        return false;
    }
    return true;
}
