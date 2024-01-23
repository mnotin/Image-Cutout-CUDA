#ifndef CONVOLUTION_HPP
#define CONVOLUTION_HPP

#include "../main.hpp"

__global__ void convolution_kernel(unsigned char *input_matrix, int *output_matrix, dim3 matrix_dim, const float *kernel, int kernel_size);
__device__ __host__ int convolution_core(int2 index, unsigned char *input_matrix, dim3 matrix_dim, const float *kernel, int kernel_size);

#endif // CONVOLUTION_HPP
