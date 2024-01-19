#ifndef CONVOLUTION_HPP
#define CONVOLUTION_HPP

#include "../main.hpp"

__global__ void convolution_kernel(unsigned char *input_matrix, int *output_matrix, Dim matrix_dim, float *kernel, int kernel_size);
__device__ __host__ int convolution_core(Vec2 index, unsigned char *input_matrix, int *output_matrix, Dim matrix_dim, float *kernel, int kernel_size);

#endif // CONVOLUTION_HPP
