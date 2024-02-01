#include "convolution_core.hpp"

/**
 * Applies convolution over an element of a matrix using a given kernel.
 **/
__device__ __host__ int convolution_core(int2 index, unsigned char *input_matrix, dim3 matrix_dim, const float *kernel, int kernel_size) {
  int convolution_result = 0;

  for (int i = 0; i < kernel_size; i++) {
    for (int j = 0; j < kernel_size; j++) {
      int vertical_offset = index.y - floor(kernel_size / (float) 2.0) + i;
      int horizontal_offset = index.x - floor(kernel_size / (float) 2.0) + j;
      int tmp_index = vertical_offset*matrix_dim.x + horizontal_offset;

      convolution_result += input_matrix[matrix_dim.x +1 + tmp_index] * kernel[i*kernel_size + j];
    }
  }

  if (255 < abs(convolution_result)) {
    convolution_result = convolution_result < 0 ? -255 : 255;
  }

  return convolution_result;
}
