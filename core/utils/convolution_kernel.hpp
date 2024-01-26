#ifndef CONVOLUTION_KERNEL_HPP
#define CONVOLUTION_KERNEL_HPP

__global__ void convolution_kernel(unsigned char *input_matrix, int *output_matrix, dim3 matrix_dim, const float *kernel, int kernel_size);

#endif // CONVOLUTION_KERNEL_HPP
