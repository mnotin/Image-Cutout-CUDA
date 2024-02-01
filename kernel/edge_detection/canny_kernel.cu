#include "canny_kernel.hpp"

#include "../../core/edge_detection/canny_core.hpp"
#include "../../main.hpp"

__global__ void non_maximum_suppression_kernel(unsigned char *gradient_matrix, float *angle_matrix, dim3 matrix_dim) {
  int2 global_index = make_int2(threadIdx.x + (blockIdx.x * blockDim.x), threadIdx.y + (blockIdx.y * blockDim.y));
  unsigned char final_value = 0;
 
  if (global_index.x < matrix_dim.x && global_index.y < matrix_dim.y) {
    final_value = non_maximum_suppression_core(global_index, gradient_matrix, angle_matrix, matrix_dim);
  }
  
  // Avoid race condition with gradient_matrix wich is read before and written after
  __syncthreads(); 
  
  if (global_index.x < matrix_dim.x && global_index.y < matrix_dim.y) {
    gradient_matrix[global_index.y*matrix_dim.x + global_index.x] = final_value;
  }
}

__global__ void histeresis_thresholding_init_kernel(unsigned char *gradient_matrix, char *ht_matrix, dim3 matrix_dim, int canny_min, int canny_max) {
  int2 global_index = make_int2(threadIdx.x + (blockIdx.x * blockDim.x), threadIdx.y + (blockIdx.y * blockDim.y));

  if (global_index.x < matrix_dim.x && global_index.y < matrix_dim.y) {
    ht_matrix[global_index.y*matrix_dim.x + global_index.x] =
      histeresis_thresholding_init_core(global_index, gradient_matrix, matrix_dim, canny_min, canny_max);
  }
}

__global__ void histeresis_thresholding_loop_kernel(char *ht_matrix, dim3 matrix_dim, int *done) {
  int2 global_index = make_int2(threadIdx.x + (blockIdx.x * blockDim.x), threadIdx.y + (blockIdx.y * blockDim.y));
  int2 local_index = make_int2(threadIdx.x, threadIdx.y);
 
  __shared__ bool right_block;
  __shared__ bool bottom_block;
  if (local_index.x == 0 && local_index.y == 0) {
    right_block = false;
    bottom_block = false;
  }
  __syncthreads();
  if (matrix_dim.x <= global_index.x) {
    right_block = true;
  }
  if (matrix_dim.y <= global_index.y) {
    bottom_block = true;
  }
  __syncthreads();
  
  dim3 shared_matrix_dim(MATRIX_SIZE_PER_BLOCK, MATRIX_SIZE_PER_BLOCK);
  int2 read_limit = make_int2(shared_matrix_dim.x, shared_matrix_dim.y);
  if (right_block) {
    read_limit.x = matrix_dim.x % MATRIX_SIZE_PER_BLOCK;
  }
  if (bottom_block) {
    read_limit.y = matrix_dim.y % MATRIX_SIZE_PER_BLOCK;
  }

  __shared__ int shared_done;
  __shared__ char shared_ht_matrix[MATRIX_SIZE_PER_BLOCK*MATRIX_SIZE_PER_BLOCK];

  if (global_index.x < matrix_dim.x && global_index.y < matrix_dim.y) {
    shared_ht_matrix[local_index.y*MATRIX_SIZE_PER_BLOCK + local_index.x] = ht_matrix[global_index.y*matrix_dim.x + global_index.x];
  }

  if (local_index.x == 0 && local_index.y == 0) {
    shared_done = 0;
  }

  __syncthreads();
  
  while (shared_done == 0) {
    __syncthreads();

    if (local_index.x == 0 && local_index.y == 0) {
      shared_done = 1; // Let's assume the process is finished
    }
    
    __syncthreads();
    
    if (global_index.x < matrix_dim.x && global_index.y < matrix_dim.y) {
      histeresis_thresholding_loop_core(local_index, shared_ht_matrix, shared_matrix_dim, read_limit, &shared_done);
    }

    __syncthreads();

  if (local_index.x == 0 && local_index.y == 0 && shared_done == 0) {
      // At least one block had to update a pixel so we will need
      // to rerun them all at least once
      *done = 0;
    }
  }

  __syncthreads();

  if (global_index.x < matrix_dim.x && global_index.y < matrix_dim.y) {
    // Write the result back to global memory
    ht_matrix[global_index.y*matrix_dim.x + global_index.x] =
      shared_ht_matrix[local_index.y*MATRIX_SIZE_PER_BLOCK + local_index.x];
  }
}

__global__ void ProcessingUnitDevice::Canny::transfer_edges_between_blocks_kernel(char *ht_matrix, dim3 matrix_dim, int *done) {
  int2 global_index = make_int2(threadIdx.x + (blockIdx.x * blockDim.x), threadIdx.y + (blockIdx.y * blockDim.y));
  int2 local_index = make_int2(threadIdx.x, threadIdx.y);
  int2 read_limit = make_int2(matrix_dim.x, matrix_dim.y);

  __shared__ bool right_block;
  __shared__ bool bottom_block;
  if (local_index.x == 0 && local_index.y == 0) {
    right_block = false;
    bottom_block = false;
  }
  __syncthreads();
  if (matrix_dim.x <= global_index.x) {
    right_block = true;
  }
  if (matrix_dim.y <= global_index.y) {
    bottom_block = true;
  }
  __syncthreads();
  
  if (right_block) {
    read_limit.x = matrix_dim.x % MATRIX_SIZE_PER_BLOCK;
  }
  if (bottom_block) {
    read_limit.y = matrix_dim.y % MATRIX_SIZE_PER_BLOCK;
  }
  
  if (global_index.x < matrix_dim.x && global_index.y < matrix_dim.y) {
    if (local_index.y == 0 && 0 < global_index.y ||
        local_index.y == MATRIX_SIZE_PER_BLOCK-1 && global_index.y < matrix_dim.y-1 ||
        local_index.x == 0 && 0 < global_index.x ||
        local_index.x == MATRIX_SIZE_PER_BLOCK-1 && global_index.x < matrix_dim.x-1) {
        histeresis_thresholding_loop_core(global_index, ht_matrix, matrix_dim, read_limit, done);
    }
  }
}

__global__ void histeresis_thresholding_end_kernel(unsigned char *gradient_matrix, char *ht_matrix, dim3 matrix_dim) {
  int2 global_index = make_int2(threadIdx.x + (blockIdx.x * blockDim.x), threadIdx.y + (blockIdx.y * blockDim.y));
  
  if (global_index.x < matrix_dim.x && global_index.y < matrix_dim.y) {
    gradient_matrix[global_index.y*matrix_dim.x + global_index.x] = histeresis_thresholding_end_core(global_index, ht_matrix, matrix_dim);
  }
}
