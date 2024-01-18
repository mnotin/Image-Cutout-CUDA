#ifndef UTILS_HPP
#define UTILS_HPP

#include "../main.hpp"

__global__ void convolution_kernel(unsigned char *input_matrix, int *output_matrix, int matrix_width, int matrix_height, float *kernel, int kernel_size);
__device__ __host__ int convolution_core(Vec2 index, unsigned char *input_matrix, int *output_matrix, int matrix_width, int matrix_height, float *kernel, int kernel_size);


namespace ProcessingUnitDevice {
  void gaussian_blur(unsigned char *h_matrix, int matrix_width, int matrix_height);
}


namespace ProcessingUnitHost {
  void gaussian_blur(unsigned char *matrix, int matrix_width, int matrix_height);
}

#endif // UTILS_HPP
