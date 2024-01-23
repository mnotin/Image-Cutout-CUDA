#include <math.h>
#include <stdio.h>

#include "../main.hpp"
#include "canny.hpp"

void ProcessingUnitDevice::canny(unsigned char *h_gradient_matrix, float *h_angle_matrix, Dim matrix_dim, int canny_min, int canny_max) {
  int h_done = 0;

  unsigned char *d_gradient_matrix;
  float *d_angle_matrix;
  unsigned char *d_ht_matrix;
  int *d_done;

  cudaMalloc(&d_gradient_matrix, matrix_dim.width * matrix_dim.height * sizeof(unsigned char));
  cudaMalloc(&d_angle_matrix, matrix_dim.width * matrix_dim.height * sizeof(float));
  cudaMalloc(&d_ht_matrix, matrix_dim.width * matrix_dim.height * sizeof(unsigned char));
  cudaMalloc(&d_done, sizeof(int));
  
  cudaMemcpy(d_gradient_matrix, h_gradient_matrix, matrix_dim.width * matrix_dim.height * sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_angle_matrix, h_angle_matrix, matrix_dim.width * matrix_dim.height * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_done, &h_done, sizeof(int), cudaMemcpyHostToDevice);

  dim3 threads = dim3(MATRIX_SIZE_PER_BLOCK, MATRIX_SIZE_PER_BLOCK);
  dim3 blocks = dim3(matrix_dim.width/MATRIX_SIZE_PER_BLOCK, matrix_dim.height/MATRIX_SIZE_PER_BLOCK);
  non_maximum_suppression_kernel<<<blocks, threads>>>(d_gradient_matrix, d_angle_matrix, matrix_dim);
  histeresis_thresholding_init_kernel<<<blocks, threads>>>(d_gradient_matrix, d_ht_matrix, matrix_dim, canny_min, canny_max);
  while (h_done == 0) {
    h_done = 1;
    cudaMemcpy(d_done, &h_done, sizeof(int), cudaMemcpyHostToDevice);
    histeresis_thresholding_loop_kernel<<<blocks, threads>>>(d_ht_matrix, matrix_dim, d_done);
    cudaMemcpy(&h_done, d_done, sizeof(int), cudaMemcpyDeviceToHost);
  }
  histeresis_thresholding_end_kernel<<<blocks, threads>>>(d_gradient_matrix, d_ht_matrix, matrix_dim);

  cudaMemcpy(h_gradient_matrix, d_gradient_matrix, matrix_dim.width * matrix_dim.height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

  cudaFree(d_gradient_matrix);
  cudaFree(d_angle_matrix);
  cudaFree(d_done);
}

void ProcessingUnitHost::canny(unsigned char *gradient_matrix, float *angle_matrix, Dim matrix_dim, int canny_min, int canny_max) {
  int done = 0;
  unsigned char ht_matrix[matrix_dim.width * matrix_dim.height];
  unsigned char gradient_matrix_buffer[matrix_dim.width * matrix_dim.height];
  int2 index;
  
  for (index.y = 0; index.y < matrix_dim.height; index.y++) {
    for (index.x = 0; index.x < matrix_dim.width; index.x++) {
      gradient_matrix_buffer[index.y*matrix_dim.width + index.x] = gradient_matrix[index.y*matrix_dim.width + index.x];
    }
  }
  for (index.y = 0; index.y < matrix_dim.height; index.y++) {
    for (index.x = 0; index.x < matrix_dim.width; index.x++) {
      gradient_matrix_buffer[index.y*matrix_dim.width + index.x] = non_maximum_suppression_core(index, gradient_matrix, angle_matrix, matrix_dim);
    }
  }
  for (index.y = 0; index.y < matrix_dim.height; index.y++) {
    for (index.x = 0; index.x < matrix_dim.width; index.x++) {
      gradient_matrix[index.y*matrix_dim.width + index.x] = gradient_matrix_buffer[index.y*matrix_dim.width + index.x];
    }
  }
  
  for (index.y = 0; index.y < matrix_dim.height; index.y++) {
    for (index.x = 0; index.x < matrix_dim.width; index.x++) {
      ht_matrix[index.y*matrix_dim.width + index.x] = histeresis_thresholding_init_core(index, gradient_matrix, matrix_dim, canny_min, canny_max);
    }
  }

  while (done == 0) {
    done = 1;

    for (index.y = 0; index.y < matrix_dim.height; index.y++) {
      for (index.x = 0; index.x < matrix_dim.width; index.x++) {
        histeresis_thresholding_loop_core(index, ht_matrix, matrix_dim, &done);
      }
    }
  }

  for (index.y = 0; index.y < matrix_dim.height; index.y++) {
    for (index.x = 0; index.x < matrix_dim.width; index.x++) {
      gradient_matrix[index.y*matrix_dim.width + index.x] = histeresis_thresholding_end_core(index, ht_matrix, matrix_dim);
    }
  }
}

__global__ void non_maximum_suppression_kernel(unsigned char *gradient_matrix, float *angle_matrix, Dim matrix_dim) {
  int2 index = make_int2(threadIdx.x + (blockIdx.x * blockDim.x), threadIdx.y + (blockIdx.y * blockDim.y));
  
  unsigned char final_value = non_maximum_suppression_core(index, gradient_matrix, angle_matrix, matrix_dim);
  
  // Avoid race condition with gradient_matrix wich is read before and written after
  __syncthreads(); 
  
  gradient_matrix[index.y*matrix_dim.width + index.x] = final_value; 
}

__device__ __host__ unsigned char non_maximum_suppression_core(int2 index, unsigned char *gradient_matrix, float *angle_matrix, Dim matrix_dim) {
  const int INT_INDEX = index.y*matrix_dim.width + index.x;
  const float ANGLE = angle_matrix[INT_INDEX] + M_PI_2;
  unsigned char final_value = gradient_matrix[INT_INDEX];

  if (get_color_canny(ANGLE) == 'Y') {
    // Vertical gradient direction : Yellow
    if (0 < index.y && gradient_matrix[INT_INDEX] < gradient_matrix[INT_INDEX - matrix_dim.width] && 
          get_color_canny(angle_matrix[INT_INDEX - matrix_dim.width] + M_PI_2) == 'Y') {
      final_value = 0;
    } else if (index.y < matrix_dim.width - 1 && gradient_matrix[INT_INDEX] < gradient_matrix[INT_INDEX + matrix_dim.width] &&
        get_color_canny(angle_matrix[INT_INDEX + matrix_dim.width] + M_PI_2) == 'Y') {
      final_value = 0;
    }
  } else if (get_color_canny(ANGLE) == 'G') {
    // Top right gradient direction : Green
    if (index.x < matrix_dim.width-1 && 0 < index.y && gradient_matrix[INT_INDEX] < gradient_matrix[INT_INDEX - matrix_dim.width + 1] &&
      get_color_canny(angle_matrix[INT_INDEX - matrix_dim.width + 1] + M_PI_2) == 'G') {
      final_value = 0;
    } else if (0 < index.x && index.y < matrix_dim.height-1 && gradient_matrix[INT_INDEX] < gradient_matrix[INT_INDEX + matrix_dim.width - 1] &&
      get_color_canny(angle_matrix[INT_INDEX - matrix_dim.width - 1] + M_PI_2) == 'G') {
      final_value = 0;
    }
  } else if (get_color_canny(ANGLE) == 'R') {
    // Top left gradient direction : Red
    if (0 < index.x && 0 < index.y && gradient_matrix[INT_INDEX] < gradient_matrix[INT_INDEX - matrix_dim.width - 1] &&
      get_color_canny(angle_matrix[INT_INDEX - matrix_dim.width - 1] + M_PI_2) == 'R') {
      final_value = 0;
    } else if (index.x < matrix_dim.width-1 && index.y < matrix_dim.height-1 &&
      gradient_matrix[INT_INDEX] < gradient_matrix[INT_INDEX + matrix_dim.width + 1] &&
      get_color_canny(angle_matrix[INT_INDEX + matrix_dim.width + 1] + M_PI_2) == 'R') { 
      final_value = 0;
    }
  } else if (get_color_canny(ANGLE) == 'B')  {
    // Horizontal gradient direction : Blue
    if (0 < index.x && gradient_matrix[INT_INDEX] < gradient_matrix[INT_INDEX - 1] &&
      get_color_canny(angle_matrix[INT_INDEX - 1] + M_PI_2) == 'B') {
      final_value = 0;
    } else if (index.x < matrix_dim.width - 1 && gradient_matrix[INT_INDEX] < gradient_matrix[INT_INDEX + 1] &&
      get_color_canny(angle_matrix[INT_INDEX + 1] + M_PI_2) == 'B') {
      final_value = 0;
    }
  }

  return final_value; 
}

__global__ void histeresis_thresholding_init_kernel(unsigned char *gradient_matrix, unsigned char *ht_matrix, Dim matrix_dim, int canny_min, int canny_max) {
  int2 index = make_int2(threadIdx.x + (blockIdx.x * blockDim.x), threadIdx.y + (blockIdx.y * blockDim.y));

  ht_matrix[index.y*matrix_dim.width + index.x] =
    histeresis_thresholding_init_core(index, gradient_matrix, matrix_dim, canny_min, canny_max);
}

__device__ __host__ unsigned char histeresis_thresholding_init_core(int2 index, unsigned char *gradient_matrix, Dim matrix_dim, int canny_min, int canny_max) {
  const int INT_INDEX = index.y*matrix_dim.width + index.x;
  unsigned char result;

  if (gradient_matrix[INT_INDEX] < canny_min) {
    result = 'D'; // Discarded
  } else if (canny_max < gradient_matrix[INT_INDEX]) {
    result = 'M'; // Marked
  } else {
    result = 'P'; // Pending
  }

  return result;
}

__global__ void histeresis_thresholding_loop_kernel(unsigned char *ht_matrix, Dim matrix_dim, int *done) {
  Vec2 global_index;
  Vec2 local_index;
  global_index.x = threadIdx.x + (blockIdx.x * blockDim.x);
  global_index.y = threadIdx.y + (blockIdx.y * blockDim.y);
  local_index.x = threadIdx.x;
  local_index.y = threadIdx.y;

  __shared__ int shared_done;

  if (local_index.x == 0 && local_index.y == 0) {
    shared_done = 1; // Initialize the variable of the block
  }
  
  __syncthreads();

  histeresis_thresholding_loop_core(global_index, ht_matrix, matrix_dim, &shared_done);

  __syncthreads();
   
  if (local_index.x == 0 && local_index.y == 0 && shared_done == 0) {
    *done = 0;
  }
}

/**
 * Mark pending pixels connected to a marked pixel.
 */
__device__ __host__ void histeresis_thresholding_loop_core(int2 index, unsigned char *ht_matrix, Dim matrix_dim, int *done) {
  const int INT_INDEX = index.y*matrix_dim.width + index.x; 

  if (ht_matrix[INT_INDEX] == 'P') {
    // Pending pixel

    if (0 < index.x && 0 < index.y && ht_matrix[(index.y-1)*matrix_dim.width + index.x-1] == 'M') {
      // Top Left
      ht_matrix[INT_INDEX] = 'M';
      *done = 0;
    } else if (0 < index.y && ht_matrix[(index.y-1)*matrix_dim.width + index.x] == 'M') {
      // Top
      ht_matrix[INT_INDEX] = 'M';
      *done = 0;
    } else if (index.x < matrix_dim.width-1 && 0 < index.y && ht_matrix[(index.y-1)*matrix_dim.width + index.x+1] == 'M') {
      // Top Right
      ht_matrix[INT_INDEX] = 'M';
      *done = 0;
    } else if (0 < index.x && ht_matrix[index.y*matrix_dim.width + index.x-1] == 'M') {
      // Left 
      ht_matrix[INT_INDEX] = 'M';
      *done = 0;
    } else if (index.x < matrix_dim.width-1 && ht_matrix[index.y*matrix_dim.width + index.x+1] == 'M') {
      // Right
      ht_matrix[INT_INDEX] = 'M';
      *done = 0;
    } else if (0 < index.x && index.y < matrix_dim.height-1 && ht_matrix[(index.y+1)*matrix_dim.width + index.x-1] == 'M') {
      // Bottom Left
      ht_matrix[INT_INDEX] = 'M';
      *done = 0;
    } else if (index.y < matrix_dim.height-1 && ht_matrix[(index.y+1)*matrix_dim.width + index.x] == 'M') {
      // Bottom
      ht_matrix[INT_INDEX] = 'M';
      *done = 0;
    } else if (index.x < matrix_dim.width-1 && index.y < matrix_dim.height-1 && ht_matrix[(index.y+1)*matrix_dim.width + index.x+1] == 'M') {
      // Bottom Left
      ht_matrix[INT_INDEX] = 'M';
      *done = 0;
    }
  }
}

__global__ void histeresis_thresholding_end_kernel(unsigned char *gradient_matrix, unsigned char *ht_matrix, Dim matrix_dim) {
  int2 index = make_int2(threadIdx.x + (blockIdx.x * blockDim.x), threadIdx.y + (blockIdx.y * blockDim.y));

  gradient_matrix[index.y*matrix_dim.width + index.x] = histeresis_thresholding_end_core(index, ht_matrix, matrix_dim);
}

__device__ __host__ unsigned char histeresis_thresholding_end_core(int2 index, unsigned char *ht_matrix, Dim matrix_dim) {
  const int INT_INDEX = index.y*matrix_dim.width + index.x;   
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
