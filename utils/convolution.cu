#include <iostream>

#include "convolution.hpp"
#include "../main.hpp"

__global__ void convolution_kernel(unsigned char *input_matrix, int *output_matrix, Dim matrix_dim, float *kernel, int kernel_size) {
  int globalIdxX = threadIdx.x + (blockIdx.x * blockDim.x);
  int globalIdxY = threadIdx.y + (blockIdx.y * blockDim.y);
  int localIdxX = threadIdx.x;
  int localIdxY = threadIdx.y;
  
  int current_matrix_index = globalIdxY*matrix_dim.width + globalIdxX;
  int current_shared_matrix_index = MATRIX_SIZE_PER_BLOCK+2+1+ localIdxY*(MATRIX_SIZE_PER_BLOCK+2) + localIdxX;

  __shared__ unsigned char shared_matrix[(MATRIX_SIZE_PER_BLOCK+2)*(MATRIX_SIZE_PER_BLOCK+2)];

  /*
   * x x x x x x MATRIX_SIZE_PER_BLOCK + 2
   * x o o o o x
   * x o o o o x
   * x o o o o x
   * x o o o o x
   * x x x x x x
   */
  shared_matrix[current_shared_matrix_index] = input_matrix[current_matrix_index];

  // Handle the borders of each block
  if (localIdxX == 0 && localIdxY == 0) {
    // Fill the edges
    for (int i = 0; i < MATRIX_SIZE_PER_BLOCK+2; i++) {
      // First line
      int first_line_offset = -1;
      if (0 == globalIdxY) {
        first_line_offset = 0;
      }
      shared_matrix[i] = input_matrix[(globalIdxY+first_line_offset)*matrix_dim.width + globalIdxX + i - 1];
      
      // Last line
      int last_line_offset = 0;
      if (globalIdxY+MATRIX_SIZE_PER_BLOCK == matrix_dim.height) {
        last_line_offset = -1;
      }
      shared_matrix[(MATRIX_SIZE_PER_BLOCK+2)*(MATRIX_SIZE_PER_BLOCK+1)+i] =
        input_matrix[(globalIdxY+MATRIX_SIZE_PER_BLOCK+last_line_offset)*matrix_dim.width + globalIdxX + i - 1];
    }

    for (int i = 0; i < MATRIX_SIZE_PER_BLOCK; i++) {
      // Left side
      int left_side_offset = -1;
      if (0 == globalIdxX) {
        left_side_offset = 0;
      }
      shared_matrix[MATRIX_SIZE_PER_BLOCK+2 + i*(MATRIX_SIZE_PER_BLOCK+2)] = 
        input_matrix[(globalIdxY+i)*matrix_dim.width + globalIdxX + left_side_offset];

      // Right side
      int right_side_offset = 0;
      if (globalIdxX+MATRIX_SIZE_PER_BLOCK == matrix_dim.width) {
        right_side_offset = -1;
      }
      shared_matrix[MATRIX_SIZE_PER_BLOCK+2 + (i+1)*(MATRIX_SIZE_PER_BLOCK+2) - 1] =
        input_matrix[(globalIdxY+i)*matrix_dim.width + globalIdxX+MATRIX_SIZE_PER_BLOCK + right_side_offset];
    }
  }
  __syncthreads();

  Vec2 index;
  index.x = localIdxX;
  index.y = localIdxY;
  Dim shared_matrix_dim;
  shared_matrix_dim.width = MATRIX_SIZE_PER_BLOCK+2;
  shared_matrix_dim.height = MATRIX_SIZE_PER_BLOCK+2;
  output_matrix[current_matrix_index] = convolution_core(index,
    shared_matrix,
    output_matrix,
    shared_matrix_dim,
    kernel,
    kernel_size);
}

/**
 * Applies discrete convolution over a matrix using a given kernel.
 * This kernel should be called using appropriate number of grids, blocks and threads to match the resolution of the image.
 **/
__device__ __host__ int convolution_core(Vec2 index, unsigned char *input_matrix, int *output_matrix,
  Dim matrix_dim, float *kernel, int kernel_size
) {
  int convolution_result = 0;

  for (int i = 0; i < kernel_size; i++) {
    for (int j = 0; j < kernel_size; j++) {
      int vertical_offset = ((index.y + i) - (int)floor(kernel_size/2.0));
      int horizontal_offset = (index.x + j) - (int)floor(kernel_size/2.0);
      int tmp_index = vertical_offset*matrix_dim.width + horizontal_offset;
      
      convolution_result += input_matrix[matrix_dim.width+1 + tmp_index] * kernel[i*kernel_size + j];
    }
  }

  if (255 < abs(convolution_result)) {
    convolution_result = convolution_result < 0 ? -255 : 255;
  }
  
  return convolution_result;
}
