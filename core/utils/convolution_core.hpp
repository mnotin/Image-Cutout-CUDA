#ifndef CONVOLUTION_CORE_HPP
#define CONVOLUTION_CORE_HPP

__device__ __host__ int convolution_core(int2 index, unsigned char *input_matrix, dim3 matrix_dim, const float *kernel, int kernel_size);

#endif // CONVOLUTION_CORE_HPP
