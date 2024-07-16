#include <stdio.h>

#include <iostream>

#include "cuda_malloc.h"

void malloc_test_wrap() {
  size_t maxSize = 0;
  size_t increment = 1 << 28;  // Start with 256MB increments
  while (true) {
    void* d_ptr;
    cudaError_t err = cudaMalloc(&d_ptr, maxSize + increment);
    if (err == cudaSuccess) {
      cudaFree(d_ptr);
      maxSize += increment;
    } else {
      if (increment == 1)
        break;          // Stop if we can't allocate a single byte more
      increment >>= 1;  // Reduce the increment
    }
  }

  std::cout << "Maximum allocatable size: " << maxSize << " bytes" << std::endl;
}
