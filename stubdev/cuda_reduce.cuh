#pragma once

#include "cuda_reduce.h"

__global__ void ReduceByProblemKernel(double* d_vec_in, double* d_vec_out,
                                      int num_equations,
                                      int items_per_equation);