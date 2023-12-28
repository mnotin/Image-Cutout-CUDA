#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#include "../main.h"
#include "canny.h"

void canny(unsigned char *h_gradient_matrix, float *h_angle_matrix, int matrix_width, int matrix_height) {
  int h_done = 0;

  unsigned char *d_gradient_matrix;
  float *d_angle_matrix;
  unsigned char *d_ht_matrix;
  int *d_done;

  cudaMalloc((void **) &d_gradient_matrix, matrix_width * matrix_height * sizeof(unsigned char));
  cudaMalloc((void **) &d_angle_matrix, matrix_width * matrix_height * sizeof(float));
  cudaMalloc((void **) &d_ht_matrix, matrix_width * matrix_height * sizeof(unsigned char));
  cudaMalloc((void **) &d_done, sizeof(int));
  
  cudaMemcpy(d_gradient_matrix, h_gradient_matrix, matrix_width * matrix_height * sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_angle_matrix, h_angle_matrix, matrix_width * matrix_height * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_done, &h_done, sizeof(int), cudaMemcpyHostToDevice);

  dim3 threads = dim3(MATRIX_SIZE_PER_BLOCK, MATRIX_SIZE_PER_BLOCK);
  dim3 blocks = dim3(matrix_width/MATRIX_SIZE_PER_BLOCK, matrix_height/MATRIX_SIZE_PER_BLOCK);
  non_maximum_suppression<<<blocks, threads>>>(d_gradient_matrix, d_angle_matrix, matrix_width, matrix_height);
  histeresis_thresholding_init<<<blocks, threads>>>(d_gradient_matrix, d_ht_matrix, matrix_width, matrix_height);
  while (h_done == 0) {
    h_done = 1;
    cudaMemcpy(d_done, &h_done, sizeof(int), cudaMemcpyHostToDevice);
    histeresis_thresholding_loop<<<blocks, threads>>>(d_ht_matrix, matrix_width, matrix_height, d_done);
    cudaMemcpy(&h_done, d_done, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
  }
  histeresis_thresholding_end<<<blocks, threads>>>(d_gradient_matrix, d_ht_matrix, matrix_width, matrix_height);

  cudaMemcpy(h_gradient_matrix, d_gradient_matrix, matrix_width * matrix_height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

  cudaFree(d_gradient_matrix);
  cudaFree(d_angle_matrix);
  cudaFree(d_done);
}

__global__ void non_maximum_suppression(unsigned char *gradient_matrix, float *angle_matrix, int matrix_width, int matrix_height) {
  int globalIdxX = threadIdx.x + (blockIdx.x * blockDim.x);
  int globalIdxY = threadIdx.y + (blockIdx.y * blockDim.y);
  const int GLOBAL_IDX = globalIdxY*matrix_width + globalIdxX;

  const float ANGLE = angle_matrix[GLOBAL_IDX] + M_PI_2;
  unsigned char final_value = gradient_matrix[GLOBAL_IDX];

  if (get_color_canny(ANGLE) == 'Y') {
    // Vertical gradient direction : Yellow
    if (gradient_matrix[GLOBAL_IDX] < gradient_matrix[GLOBAL_IDX - matrix_width] && 
          get_color_canny(angle_matrix[GLOBAL_IDX - matrix_width] + M_PI_2) == 'Y' || 
        gradient_matrix[GLOBAL_IDX] < gradient_matrix[GLOBAL_IDX + matrix_width] &&
          get_color_canny(angle_matrix[GLOBAL_IDX + matrix_width] + M_PI_2) == 'Y') {
      final_value = 0;
    }
  } else if (get_color_canny(ANGLE) == 'G') {
    // Top right gradient direction : Green
    if (gradient_matrix[GLOBAL_IDX] < gradient_matrix[GLOBAL_IDX - matrix_width + 1] &&
          get_color_canny(angle_matrix[GLOBAL_IDX - matrix_width + 1] + M_PI_2) == 'G' || 
        gradient_matrix[GLOBAL_IDX] < gradient_matrix[GLOBAL_IDX + matrix_width - 1] &&
          get_color_canny(angle_matrix[GLOBAL_IDX - matrix_width - 1] + M_PI_2) == 'G') {
      final_value = 0;
    }
  } else if (get_color_canny(ANGLE) == 'R') {
    // Top left gradient direction : Red
    if (gradient_matrix[GLOBAL_IDX] < gradient_matrix[GLOBAL_IDX - matrix_width - 1] &&
          get_color_canny(angle_matrix[GLOBAL_IDX - matrix_width - 1] + M_PI_2) == 'R' || 
        gradient_matrix[GLOBAL_IDX] < gradient_matrix[GLOBAL_IDX + matrix_width + 1] &&
          get_color_canny(angle_matrix[GLOBAL_IDX + matrix_width + 1] + M_PI_2) == 'R') { 
      final_value = 0;
    }
  } else {
    // Horizontal gradient direction : Blue
    if (gradient_matrix[GLOBAL_IDX] < gradient_matrix[GLOBAL_IDX - 1] &&
          get_color_canny(angle_matrix[GLOBAL_IDX - 1] + M_PI_2) == 'B' || 
        gradient_matrix[GLOBAL_IDX] < gradient_matrix[GLOBAL_IDX + 1] &&
          get_color_canny(angle_matrix[GLOBAL_IDX + 1] + M_PI_2) == 'B') {
      final_value = 0;
    }
  }

  // Avoid race condition with gradient_matrix wich is read before and written after
  __syncthreads(); 
  
  gradient_matrix[GLOBAL_IDX] = final_value; 
}

__global__ void histeresis_thresholding_init(unsigned char *gradient_matrix, unsigned char *ht_matrix, int matrix_width, int matrix_height) {
  int globalIdxX = threadIdx.x + (blockIdx.x * blockDim.x);
  int globalIdxY = threadIdx.y + (blockIdx.y * blockDim.y);
  const int GLOBAL_IDX = globalIdxY*matrix_width + globalIdxX;

  int min_val = 30;
  int max_val = 100;
  
  if (gradient_matrix[GLOBAL_IDX] < min_val) {
    ht_matrix[GLOBAL_IDX] = 'D'; // Discarded
  } else if (max_val < gradient_matrix[GLOBAL_IDX]) {
    ht_matrix[GLOBAL_IDX] = 'M'; // Marked
  } else {
    ht_matrix[GLOBAL_IDX] = 'P'; // Pending
  }
}

__global__ void histeresis_thresholding_loop(unsigned char *ht_matrix, int matrix_width, int matrix_height, int *done) {
  int globalIdxX = threadIdx.x + (blockIdx.x * blockDim.x);
  int globalIdxY = threadIdx.y + (blockIdx.y * blockDim.y);
  int localIdxX = threadIdx.x;
  int localIdxY = threadIdx.y;

  const int GLOBAL_IDX = globalIdxY*matrix_width + globalIdxX; 

  __shared__ int shared_done;

  if (localIdxX == 0 && localIdxY == 0) {
    shared_done = 1; // Initialize the variable of the block
  }
  
  __syncthreads();

  if (ht_matrix[GLOBAL_IDX] == 'P') {
    // Pending pixel

    if (ht_matrix[(globalIdxY-1)*matrix_width + globalIdxX-1] == 'M') {
      ht_matrix[GLOBAL_IDX] = 'M';
      shared_done = 0;
    } else if (ht_matrix[(globalIdxY-1)*matrix_width + globalIdxX] == 'M') {
      ht_matrix[GLOBAL_IDX] = 'M';
      shared_done = 0;
    } else if (ht_matrix[(globalIdxY-1)*matrix_width + globalIdxX+1] == 'M') {
      ht_matrix[GLOBAL_IDX] = 'M';
      shared_done = 0;
    } else if (ht_matrix[globalIdxY*matrix_width + globalIdxX-1] == 'M') {
      ht_matrix[GLOBAL_IDX] = 'M';
      shared_done = 0;
    } else if (ht_matrix[globalIdxY*matrix_width + globalIdxX+1] == 'M') {
      ht_matrix[GLOBAL_IDX] = 'M';
      shared_done = 0;
    } else if (ht_matrix[(globalIdxY+1)*matrix_width + globalIdxX-1] == 'M') {
      ht_matrix[GLOBAL_IDX] = 'M';
      shared_done = 0;
    } else if (ht_matrix[(globalIdxY+1)*matrix_width + globalIdxX] == 'M') {
      ht_matrix[GLOBAL_IDX] = 'M';
      shared_done = 0;
    } else if (ht_matrix[(globalIdxY+1)*matrix_width + globalIdxX+1] == 'M') {
      ht_matrix[GLOBAL_IDX] = 'M';
      shared_done = 0;
    } 
  }

  __syncthreads();
   
  if (localIdxX == 0 && localIdxY == 0 && shared_done == 0) {
    *done = 0;
  }
}

__global__ void histeresis_thresholding_end(unsigned char *gradient_matrix, unsigned char *ht_matrix, int matrix_width, int matrix_height) {
  int globalIdxX = threadIdx.x + (blockIdx.x * blockDim.x);
  int globalIdxY = threadIdx.y + (blockIdx.y * blockDim.y);

  const int GLOBAL_IDX = globalIdxY*matrix_width + globalIdxX;   

  if (ht_matrix[GLOBAL_IDX] == 'P') {
    // All the still pending pixels are discarded
    ht_matrix[GLOBAL_IDX] = 'D';
  }

  __syncthreads();

  if (ht_matrix[GLOBAL_IDX] == 'D') {
    // Final step, we set every discarded pixel to 0 in the gradient matrix
    gradient_matrix[GLOBAL_IDX] = 0;
  }
}

__device__ char get_color_canny(float angle) {
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
