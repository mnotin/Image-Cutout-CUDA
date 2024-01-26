#include "convolution_kernel.hpp"

#include "../../core/utils/convolution_core.hpp"
#include "../../main.hpp"

__global__ void convolution_kernel(unsigned char *input_matrix, int *output_matrix, dim3 matrix_dim, const float *kernel, int kernel_size) {
  int2 global_index = make_int2(threadIdx.x + (blockIdx.x * blockDim.x), threadIdx.y + (blockIdx.y * blockDim.y));
  int2 local_index = make_int2(threadIdx.x, threadIdx.y);
  
  int current_mat_idx = global_index.y*matrix_dim.x + global_index.x;
  int current_shared_mat_idx = MATRIX_SIZE_PER_BLOCK+2+1+ local_index.y*(MATRIX_SIZE_PER_BLOCK+2) + local_index.x;

  dim3 shared_matrix_dim(MATRIX_SIZE_PER_BLOCK+2, MATRIX_SIZE_PER_BLOCK+2);
  __shared__ unsigned char shared_matrix[(MATRIX_SIZE_PER_BLOCK+2)*(MATRIX_SIZE_PER_BLOCK+2)];

  /*
   * x x x x x x MATRIX_SIZE_PER_BLOCK + 2
   * x o o o o x
   * x o o o o x
   * x o o o o x
   * x o o o o x
   * x x x x x x
   */

  if (global_index.x < matrix_dim.x && global_index.y < matrix_dim.y) {
    shared_matrix[current_shared_mat_idx] = input_matrix[current_mat_idx];

    // Handle borders of the block
    if (local_index.y == 0) {
      // First line
      if (global_index.y == 0) {
        shared_matrix[current_shared_mat_idx - shared_matrix_dim.x] = input_matrix[current_mat_idx];
      } else {
        shared_matrix[current_shared_mat_idx - shared_matrix_dim.x] = input_matrix[current_mat_idx - matrix_dim.x];
      }
    } else if (local_index.y == MATRIX_SIZE_PER_BLOCK-1 || global_index.y == matrix_dim.y-1) {
      // Last line
      if (global_index.y == matrix_dim.y-1) {
        shared_matrix[current_shared_mat_idx + shared_matrix_dim.x] = input_matrix[current_mat_idx];
      } else {
        shared_matrix[current_shared_mat_idx + shared_matrix_dim.x] = input_matrix[current_mat_idx + matrix_dim.x];
      }
    }
  
    if (local_index.x == 0) {
      // Left side
      if (global_index.x == 0) {
        shared_matrix[current_shared_mat_idx - 1] = input_matrix[current_mat_idx];
      } else {
        shared_matrix[current_shared_mat_idx - 1] = input_matrix[current_mat_idx - 1];
      }
    } else if (local_index.x == MATRIX_SIZE_PER_BLOCK-1 || global_index.x == matrix_dim.x-1) {
      // Right side
      if (global_index.x == matrix_dim.x-1) {
        shared_matrix[current_shared_mat_idx + 1] = input_matrix[current_mat_idx];
      } else {
        shared_matrix[current_shared_mat_idx + 1] = input_matrix[current_mat_idx + 1];
      }
    }

    // Handle corners of the block
    if (local_index.x == 0 && local_index.y == 0) {
      // Top left
      if (global_index.x == 0 && global_index.y == 0) {
        shared_matrix[0] = input_matrix[current_mat_idx];
      } else if (global_index.x == 0) {
        shared_matrix[0] = input_matrix[current_mat_idx - matrix_dim.x];
      } else if (global_index.y == 0) {
        shared_matrix[0] = input_matrix[current_mat_idx - 1];
      } else {
        shared_matrix[0] = input_matrix[current_mat_idx - matrix_dim.x - 1];
      }
    } else if ((local_index.x == MATRIX_SIZE_PER_BLOCK-1 || global_index.x == matrix_dim.x-1) && local_index.y == 0) {
      // Top right
      if (global_index.x == matrix_dim.x-1 && global_index.y == 0) {
        shared_matrix[local_index.x+2] = input_matrix[current_mat_idx];
      } else if (global_index.x == matrix_dim.x-1) {
        shared_matrix[local_index.x+2] = input_matrix[current_mat_idx - matrix_dim.x];
      } else if (global_index.y == 0) {
        shared_matrix[local_index.x+2] = input_matrix[current_mat_idx + 1];
      } else {
        shared_matrix[local_index.x+2] = input_matrix[current_mat_idx - matrix_dim.x + 1];
      }
    } else if (local_index.x == 0 && (local_index.y == MATRIX_SIZE_PER_BLOCK-1 || global_index.y == matrix_dim.y-1)) {
      // Bottom left
      if (global_index.x == 0 && global_index.y == matrix_dim.y-1) {
        shared_matrix[current_shared_mat_idx + shared_matrix_dim.x - 1] = input_matrix[current_mat_idx];
      } else if (global_index.x == 0) {
        shared_matrix[current_shared_mat_idx + shared_matrix_dim.x - 1] = input_matrix[current_mat_idx + matrix_dim.x];
      } else if (global_index.y == matrix_dim.y-1) {
        shared_matrix[current_shared_mat_idx + shared_matrix_dim.x - 1] = input_matrix[current_mat_idx - 1];
      } else {
        shared_matrix[current_shared_mat_idx + shared_matrix_dim.x - 1] = input_matrix[current_mat_idx + matrix_dim.x - 1];
      }
    } else if ((local_index.x == MATRIX_SIZE_PER_BLOCK-1 || global_index.x == matrix_dim.x-1) &&
               (local_index.y == MATRIX_SIZE_PER_BLOCK-1 || global_index.y == matrix_dim.x-1)) {
      // Bottom right
      if (global_index.x == matrix_dim.x-1 && global_index.y == matrix_dim.y-1) {
        shared_matrix[(local_index.y+2)*shared_matrix_dim.x + local_index.x + 2] = input_matrix[current_mat_idx];
      } else if (global_index.x == matrix_dim.x-1) {
        shared_matrix[(local_index.y+2)*shared_matrix_dim.x + local_index.x + 2] = input_matrix[current_mat_idx + matrix_dim.x];
      } else if (global_index.y == matrix_dim.y-1) {
        shared_matrix[(local_index.y+2)*shared_matrix_dim.x + local_index.x + 2] = input_matrix[current_mat_idx + 1];
      } else {
        shared_matrix[(local_index.y+2)*shared_matrix_dim.x + local_index.x + 2] = input_matrix[current_mat_idx + matrix_dim.x + 1];
      }
    }
  }

  __syncthreads();

  if (global_index.x < matrix_dim.x && global_index.y < matrix_dim.y) {
    output_matrix[current_mat_idx] = convolution_core(
      local_index,
      shared_matrix,
      shared_matrix_dim,
      kernel,
      kernel_size);
  }
}
