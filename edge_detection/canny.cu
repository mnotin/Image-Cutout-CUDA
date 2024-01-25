#include <math.h>
#include <stdio.h>

#include "../main.hpp"
#include "canny.hpp"

void ProcessingUnitDevice::canny(unsigned char *h_gradient_matrix, float *h_angle_matrix, dim3 matrix_dim, int canny_min, int canny_max) {
  dim3 threads(MATRIX_SIZE_PER_BLOCK, MATRIX_SIZE_PER_BLOCK);
  dim3 blocks(ceil((float) matrix_dim.x/MATRIX_SIZE_PER_BLOCK), ceil((float) matrix_dim.y/MATRIX_SIZE_PER_BLOCK));
  int h_done = 0;

  unsigned char *d_gradient_matrix;
  float *d_angle_matrix;
  char *d_ht_matrix;
  int *d_done;

  cudaMalloc(&d_gradient_matrix, matrix_dim.x * matrix_dim.y * sizeof(unsigned char));
  cudaMalloc(&d_angle_matrix, matrix_dim.x * matrix_dim.y * sizeof(float));
  cudaMalloc(&d_ht_matrix, matrix_dim.x * matrix_dim.y * sizeof(char));
  cudaMalloc(&d_done, sizeof(int));
  
  cudaMemcpy(d_gradient_matrix, h_gradient_matrix, matrix_dim.x * matrix_dim.y * sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_angle_matrix, h_angle_matrix, matrix_dim.x * matrix_dim.y * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_done, &h_done, sizeof(int), cudaMemcpyHostToDevice);

  non_maximum_suppression_kernel<<<blocks, threads>>>(d_gradient_matrix, d_angle_matrix, matrix_dim);
  histeresis_thresholding_init_kernel<<<blocks, threads>>>(d_gradient_matrix, d_ht_matrix, matrix_dim, canny_min, canny_max);
  while (h_done == 0) {
    h_done = 1;
    cudaMemcpy(d_done, &h_done, sizeof(int), cudaMemcpyHostToDevice);
    histeresis_thresholding_loop_kernel<<<blocks, threads>>>(d_ht_matrix, matrix_dim, d_done);
    
    Canny::transfer_edges_between_blocks_kernel<<<blocks, threads>>>(d_ht_matrix, matrix_dim, d_done);
    cudaMemcpy(&h_done, d_done, sizeof(int), cudaMemcpyDeviceToHost);
  }
  histeresis_thresholding_end_kernel<<<blocks, threads>>>(d_gradient_matrix, d_ht_matrix, matrix_dim);

  cudaMemcpy(h_gradient_matrix, d_gradient_matrix, matrix_dim.x * matrix_dim.y * sizeof(unsigned char), cudaMemcpyDeviceToHost);

  cudaFree(d_gradient_matrix);
  cudaFree(d_angle_matrix);
  cudaFree(d_ht_matrix);
  cudaFree(d_done);
}

void ProcessingUnitHost::canny(unsigned char *gradient_matrix, float *angle_matrix, dim3 matrix_dim, int canny_min, int canny_max) {
  int done = 0;
  char ht_matrix[matrix_dim.x * matrix_dim.y];
  unsigned char gradient_matrix_buffer[matrix_dim.x * matrix_dim.y];
  int2 index;
  
  for (index.y = 0; index.y < matrix_dim.y; index.y++) {
    for (index.x = 0; index.x < matrix_dim.x; index.x++) {
      gradient_matrix_buffer[index.y*matrix_dim.x + index.x] = gradient_matrix[index.y*matrix_dim.x + index.x];
    }
  }
  for (index.y = 0; index.y < matrix_dim.y; index.y++) {
    for (index.x = 0; index.x < matrix_dim.x; index.x++) {
      gradient_matrix_buffer[index.y*matrix_dim.x + index.x] = non_maximum_suppression_core(index, gradient_matrix, angle_matrix, matrix_dim);
    }
  }
  for (index.y = 0; index.y < matrix_dim.y; index.y++) {
    for (index.x = 0; index.x < matrix_dim.x; index.x++) {
      gradient_matrix[index.y*matrix_dim.x + index.x] = gradient_matrix_buffer[index.y*matrix_dim.x + index.x];
    }
  }
  
  for (index.y = 0; index.y < matrix_dim.y; index.y++) {
    for (index.x = 0; index.x < matrix_dim.x; index.x++) {
      ht_matrix[index.y*matrix_dim.x + index.x] = histeresis_thresholding_init_core(index, gradient_matrix, matrix_dim, canny_min, canny_max);
    }
  }

  while (done == 0) {
    done = 1;

    for (index.y = 0; index.y < matrix_dim.y; index.y++) {
      for (index.x = 0; index.x < matrix_dim.x; index.x++) {
        histeresis_thresholding_loop_core(index, ht_matrix, matrix_dim, make_int2(matrix_dim.x, matrix_dim.y), &done);
      }
    }
  }

  for (index.y = 0; index.y < matrix_dim.y; index.y++) {
    for (index.x = 0; index.x < matrix_dim.x; index.x++) {
      gradient_matrix[index.y*matrix_dim.x + index.x] = histeresis_thresholding_end_core(index, ht_matrix, matrix_dim);
    }
  }
}

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

__device__ __host__ unsigned char non_maximum_suppression_core(int2 index, unsigned char *gradient_matrix, float *angle_matrix, dim3 matrix_dim) {
  const int INT_INDEX = index.y*matrix_dim.x + index.x;
  const float ANGLE = angle_matrix[INT_INDEX] + M_PI_2;
  unsigned char final_value = gradient_matrix[INT_INDEX];

  if (get_color_canny(ANGLE) == 'Y') {
    // Vertical gradient direction : Yellow
    if (0 < index.y && gradient_matrix[INT_INDEX] < gradient_matrix[INT_INDEX - matrix_dim.x] && 
          get_color_canny(angle_matrix[INT_INDEX - matrix_dim.x] + M_PI_2) == 'Y') {
      final_value = 0;
    } else if (index.y < matrix_dim.y - 1 && gradient_matrix[INT_INDEX] < gradient_matrix[INT_INDEX + matrix_dim.x] &&
        get_color_canny(angle_matrix[INT_INDEX + matrix_dim.x] + M_PI_2) == 'Y') {
      final_value = 0;
    }
  } else if (get_color_canny(ANGLE) == 'G') {
    // Top right gradient direction : Green
    if (index.x < matrix_dim.x-1 && 0 < index.y && gradient_matrix[INT_INDEX] < gradient_matrix[INT_INDEX - matrix_dim.x + 1] &&
      get_color_canny(angle_matrix[INT_INDEX - matrix_dim.x + 1] + M_PI_2) == 'G') {
      final_value = 0;
    } else if (0 < index.x && index.y < matrix_dim.y-1 && gradient_matrix[INT_INDEX] < gradient_matrix[INT_INDEX + matrix_dim.x - 1] &&
      get_color_canny(angle_matrix[INT_INDEX + matrix_dim.x - 1] + M_PI_2) == 'G') {
      final_value = 0;
    }
  } else if (get_color_canny(ANGLE) == 'R') {
    // Top left gradient direction : Red
    if (0 < index.x && 0 < index.y && gradient_matrix[INT_INDEX] < gradient_matrix[INT_INDEX - matrix_dim.x - 1] &&
      get_color_canny(angle_matrix[INT_INDEX - matrix_dim.x - 1] + M_PI_2) == 'R') {
      final_value = 0;
    } else if (index.x < matrix_dim.x-1 && index.y < matrix_dim.y-1 &&
      gradient_matrix[INT_INDEX] < gradient_matrix[INT_INDEX + matrix_dim.x + 1] &&
      get_color_canny(angle_matrix[INT_INDEX + matrix_dim.x + 1] + M_PI_2) == 'R') { 
      final_value = 0;
    }
  } else if (get_color_canny(ANGLE) == 'B')  {
    // Horizontal gradient direction : Blue
    if (0 < index.x && gradient_matrix[INT_INDEX] < gradient_matrix[INT_INDEX - 1] &&
      get_color_canny(angle_matrix[INT_INDEX - 1] + M_PI_2) == 'B') {
      final_value = 0;
    } else if (index.x < matrix_dim.x - 1 && gradient_matrix[INT_INDEX] < gradient_matrix[INT_INDEX + 1] &&
      get_color_canny(angle_matrix[INT_INDEX + 1] + M_PI_2) == 'B') {
      final_value = 0;
    }
  }

  return final_value; 
}

__global__ void histeresis_thresholding_init_kernel(unsigned char *gradient_matrix, char *ht_matrix, dim3 matrix_dim, int canny_min, int canny_max) {
  int2 global_index = make_int2(threadIdx.x + (blockIdx.x * blockDim.x), threadIdx.y + (blockIdx.y * blockDim.y));

  if (global_index.x < matrix_dim.x && global_index.y < matrix_dim.y) {
    ht_matrix[global_index.y*matrix_dim.x + global_index.x] =
      histeresis_thresholding_init_core(global_index, gradient_matrix, matrix_dim, canny_min, canny_max);
  }
}

__device__ __host__ char histeresis_thresholding_init_core(int2 index, unsigned char *gradient_matrix, dim3 matrix_dim, int canny_min, int canny_max) {
  const int INT_INDEX = index.y*matrix_dim.x + index.x;
  char result;

  if (gradient_matrix[INT_INDEX] < canny_min) {
    result = 'D'; // Discarded
  } else if (canny_max < gradient_matrix[INT_INDEX]) {
    result = 'M'; // Marked
  } else {
    result = 'P'; // Pending
  }

  return result;
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

/**
 * Mark pending pixels connected to a marked pixel.
 */
__device__ __host__ void histeresis_thresholding_loop_core(int2 index, char *ht_matrix, dim3 matrix_dim, int2 read_limit, int *done) {
  const int INT_INDEX = index.y*matrix_dim.x + index.x; 

  if (ht_matrix[INT_INDEX] == 'P') {
    // Pending pixel

    if (0 < index.x && 0 < index.y && ht_matrix[(index.y-1)*matrix_dim.x + index.x-1] == 'M') {
      // Top Left
      ht_matrix[INT_INDEX] = 'M';
      *done = 0;
    } else if (0 < index.y && ht_matrix[(index.y-1)*matrix_dim.x + index.x] == 'M') {
      // Top
      ht_matrix[INT_INDEX] = 'M';
      *done = 0;
    } else if (index.x < read_limit.x-1 && 0 < index.y && ht_matrix[(index.y-1)*matrix_dim.x + index.x+1] == 'M') {
      // Top Right
      ht_matrix[INT_INDEX] = 'M';
      *done = 0;
    } else if (0 < index.x && ht_matrix[index.y*matrix_dim.x + index.x-1] == 'M') {
      // Left 
      ht_matrix[INT_INDEX] = 'M';
      *done = 0;
    } else if (index.x < read_limit.x-1 && ht_matrix[index.y*matrix_dim.x+ index.x+1] == 'M') {
      // Right
      ht_matrix[INT_INDEX] = 'M';
      *done = 0;
    } else if (0 < index.x && index.y < read_limit.y-1 && ht_matrix[(index.y+1)*matrix_dim.x + index.x-1] == 'M') {
      // Bottom Left
      ht_matrix[INT_INDEX] = 'M';
      *done = 0;
    } else if (index.y < read_limit.y-1 && ht_matrix[(index.y+1)*matrix_dim.x + index.x] == 'M') {
      // Bottom
      ht_matrix[INT_INDEX] = 'M';
      *done = 0;
    } else if (index.x < read_limit.x-1 && index.y < read_limit.y-1 && ht_matrix[(index.y+1)*matrix_dim.x + index.x+1] == 'M') {
      // Bottom Right
      ht_matrix[INT_INDEX] = 'M';
      *done = 0;
    }
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

__device__ __host__ unsigned char histeresis_thresholding_end_core(int2 index, char *ht_matrix, dim3 matrix_dim) {
  const int INT_INDEX = index.y*matrix_dim.x + index.x;   
  unsigned char result;

  if (ht_matrix[INT_INDEX] == 'P') {
    // All the still pending pixels are discarded
    ht_matrix[INT_INDEX] = 'D';
  }

  if (ht_matrix[INT_INDEX] == 'D') {
    // Final step, we set every discarded pixel to 0 in the gradient matrix
    result = 0;
  } else {
    result = 255;
  }

  return result;
}

__device__ __host__ char get_color_canny(float angle) {
  char color = ' ';

  if (angle < M_PI / 8.0 || (M_PI / 8.0) * 7 < angle) {
    // Horizontal gradient direction : Yellow
    color = 'Y';
  } else if (M_PI / 8.0 < angle && angle < (M_PI / 8.0) * 3) {
    // Top right gradient direction : Green
    color = 'G';
  } else if ((M_PI / 8.0) * 5 < angle && angle < (M_PI / 8.0) * 7) {
    // Top left gradient direction : Red
    color = 'R';
  } else {
    // Vertical gradient direction : Blue
    color = 'B';
  }

  return color;
}
