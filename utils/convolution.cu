#include <iostream>

#include "convolution.hpp"
#include "../main.hpp"

__global__ void convolution_kernel(unsigned char *input_matrix, int *output_matrix, Dim matrix_dim, const float *kernel, int kernel_size) {
  int2 global_index = make_int2(threadIdx.x + (blockIdx.x * blockDim.x), threadIdx.y + (blockIdx.y * blockDim.y));
  int2 local_index = make_int2(threadIdx.x, threadIdx.y);
  
  int current_mat_idx = global_index.y*matrix_dim.width + global_index.x;
  int current_shared_mat_idx = MATRIX_SIZE_PER_BLOCK+2+1+ local_index.y*(MATRIX_SIZE_PER_BLOCK+2) + local_index.x;

  Dim shared_matrix_dim;
  shared_matrix_dim.width = MATRIX_SIZE_PER_BLOCK+2;
  shared_matrix_dim.height = MATRIX_SIZE_PER_BLOCK+2;
  __shared__ unsigned char shared_matrix[(MATRIX_SIZE_PER_BLOCK+2)*(MATRIX_SIZE_PER_BLOCK+2)];

  /*
   * x x x x x x MATRIX_SIZE_PER_BLOCK + 2
   * x o o o o x
   * x o o o o x
   * x o o o o x
   * x o o o o x
   * x x x x x x
   */
  shared_matrix[current_shared_mat_idx] = input_matrix[current_mat_idx];

  // Handle borders of the block
  if (local_index.y == 0) {
    // First line
    if (global_index.y == 0) {
      shared_matrix[current_shared_mat_idx - shared_matrix_dim.width] = input_matrix[current_mat_idx];
    } else {
      shared_matrix[current_shared_mat_idx - shared_matrix_dim.width] =
        input_matrix[current_mat_idx - matrix_dim.width];
    }
  } else if (local_index.y == MATRIX_SIZE_PER_BLOCK-1) {
    // Last line
    if (global_index.y == matrix_dim.height-1) {
      shared_matrix[current_shared_mat_idx + shared_matrix_dim.width] = input_matrix[current_mat_idx];
    } else {
      shared_matrix[current_shared_mat_idx + shared_matrix_dim.width] =
        input_matrix[current_mat_idx + matrix_dim.width];
    }
  }
  
  if (local_index.x == 0) {
    // Left side
    if (global_index.x == 0) {
      shared_matrix[current_shared_mat_idx - 1] = input_matrix[current_mat_idx];
    } else {
      shared_matrix[current_shared_mat_idx - 1] =
        input_matrix[current_mat_idx - 1];
    }
  } else if (local_index.x == MATRIX_SIZE_PER_BLOCK-1) {
    // Right side
    if (global_index.x == matrix_dim.width-1) {
      shared_matrix[current_shared_mat_idx + 1] = input_matrix[current_mat_idx];
    } else {
      shared_matrix[current_shared_mat_idx + 1] =
        input_matrix[current_mat_idx + 1];
    }
  }

  // Handle corners of the block
  if (local_index.x == 0 && local_index.y == 0) {
    // Top left
    if (global_index.x == 0 && global_index.y == 0) {
      shared_matrix[0] = input_matrix[current_mat_idx];
    } else if (global_index.x == 0) {
      shared_matrix[0] = input_matrix[current_mat_idx - matrix_dim.width];
    } else if (global_index.y == 0) {
      shared_matrix[0] = input_matrix[current_mat_idx - 1];
    } else {
      shared_matrix[0] = input_matrix[current_mat_idx - matrix_dim.width - 1];
    }
  } else if (local_index.x == MATRIX_SIZE_PER_BLOCK-1 && local_index.y == 0) {
    // Top right
    if (global_index.x == matrix_dim.width-1 && global_index.y == 0) {
      shared_matrix[MATRIX_SIZE_PER_BLOCK+1] = input_matrix[current_mat_idx];
    } else if (global_index.x == matrix_dim.width-1) {
      shared_matrix[MATRIX_SIZE_PER_BLOCK+1] = input_matrix[current_mat_idx - matrix_dim.width];
    } else if (global_index.y == 0) {
      shared_matrix[MATRIX_SIZE_PER_BLOCK+1] = input_matrix[current_mat_idx + 1];
    } else {
      shared_matrix[MATRIX_SIZE_PER_BLOCK+1] = input_matrix[current_mat_idx - matrix_dim.width + 1];
    }
  } else if (local_index.x == 0 && local_index.y == MATRIX_SIZE_PER_BLOCK-1) {
    // Bottom left
    if (global_index.x == 0 && global_index.y == matrix_dim.height-1) {
      shared_matrix[current_shared_mat_idx + shared_matrix_dim.width - 1] = input_matrix[current_mat_idx];
    } else if (global_index.x == 0) {
      shared_matrix[current_shared_mat_idx + shared_matrix_dim.width - 1] = input_matrix[current_mat_idx + matrix_dim.width];
    } else if (global_index.y == matrix_dim.height-1) {
      shared_matrix[current_shared_mat_idx + shared_matrix_dim.width - 1] = input_matrix[current_mat_idx - 1];
    } else {
      shared_matrix[current_shared_mat_idx + shared_matrix_dim.width - 1] = input_matrix[current_mat_idx + matrix_dim.width - 1];
    }
  } else if (local_index.x == MATRIX_SIZE_PER_BLOCK-1 && local_index.y == MATRIX_SIZE_PER_BLOCK-1) {
    // Bottom right
    if (global_index.x == matrix_dim.width-1 && global_index.y == matrix_dim.height-1) {
      shared_matrix[current_shared_mat_idx + shared_matrix_dim.width + 1] = input_matrix[current_mat_idx];
    } else if (global_index.x == matrix_dim.width-1) {
      shared_matrix[current_shared_mat_idx + shared_matrix_dim.width + 1] = input_matrix[current_mat_idx + matrix_dim.width];
    } else if (global_index.y == matrix_dim.height-1) {
      shared_matrix[current_shared_mat_idx + shared_matrix_dim.width + 1] = input_matrix[current_mat_idx + 1];
    } else {
      shared_matrix[current_shared_mat_idx + shared_matrix_dim.width + 1] = input_matrix[current_mat_idx + matrix_dim.width + 1];
    }
  }

  __syncthreads();

  output_matrix[current_mat_idx] = convolution_core(
    local_index,
    shared_matrix,
    shared_matrix_dim,
    kernel,
    kernel_size);
}

/**
 * Applies discrete convolution over a matrix using a given kernel.
 * This kernel should be called using appropriate number of grids, blocks and threads to match the resolution of the image.
 **/
__device__ __host__ int convolution_core(int2 index, unsigned char *input_matrix, Dim matrix_dim, const float *kernel, int kernel_size) {
  int convolution_result = 0;

  for (int i = 0; i < kernel_size; i++) {
    for (int j = 0; j < kernel_size; j++) {
      int vertical_offset = index.y - floor(kernel_size / (float) 2.0) + i;
      int horizontal_offset = index.x - floor(kernel_size / (float) 2.0) + j;
      int tmp_index = vertical_offset*matrix_dim.width + horizontal_offset;
      
      convolution_result += input_matrix[matrix_dim.width+1 + tmp_index] * kernel[i*kernel_size + j];
    }
  }

  if (255 < abs(convolution_result)) {
    convolution_result = convolution_result < 0 ? -255 : 255;
  }
  
  return convolution_result;
}
