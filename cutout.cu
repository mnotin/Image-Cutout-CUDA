#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#include "cutout.h"
#include "main.h"
#include "utils.h"

void cutout(unsigned char *h_rgb_image, unsigned char *h_edge_matrix, int matrix_width, int matrix_height, int start_pixel_x, int start_pixel_y, int threshold) {
  int h_done = 0;
  unsigned char h_cutout_matrix[matrix_height][matrix_width];
 
  for (int i = 0; i < matrix_height; i++) {
    for (int j = 0; j < matrix_width; j++) {
      h_cutout_matrix[i][j] = 'D';
    }
  }

  unsigned char *d_rgb_image;
  unsigned char *d_edge_matrix;
  unsigned char *d_cutout_matrix;
  int *d_done;

  cudaMalloc((void **) &d_rgb_image, 3 * matrix_width * matrix_height * sizeof(unsigned char));
  cudaMalloc((void **) &d_edge_matrix, matrix_width * matrix_height * sizeof(unsigned char));
  cudaMalloc((void **) &d_cutout_matrix, matrix_width * matrix_height * sizeof(unsigned char));
  cudaMalloc((void **) &d_done, sizeof(int));

  cudaMemcpy(d_rgb_image, h_rgb_image, 3 * matrix_width * matrix_height * sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_edge_matrix, h_edge_matrix, matrix_width * matrix_height * sizeof(unsigned char), cudaMemcpyHostToDevice);
  for (int i = 0; i < matrix_height; i++) {
    cudaMemcpy(d_cutout_matrix+i*matrix_width, h_cutout_matrix[i], matrix_width * sizeof(unsigned char), cudaMemcpyHostToDevice);
  }
  
  dim3 threads = dim3(MATRIX_SIZE_PER_BLOCK, MATRIX_SIZE_PER_BLOCK);
  dim3 blocks = dim3(matrix_width/MATRIX_SIZE_PER_BLOCK, matrix_height/MATRIX_SIZE_PER_BLOCK);
  draw_edges_on_cutout_matrix<<<blocks, threads>>>(d_edge_matrix, d_cutout_matrix, matrix_width, matrix_height, start_pixel_x, start_pixel_y, threshold);

  while (h_done == 0) {
    h_done = 1;
    cudaMemcpy(d_done, &h_done, sizeof(int), cudaMemcpyHostToDevice);
    cutout_algorithm<<<blocks, threads>>>(d_cutout_matrix, matrix_width, matrix_height, d_done);
    cudaMemcpy(&h_done, d_done, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
  }
  apply_cutout<<<blocks, threads>>>(d_cutout_matrix, d_rgb_image, matrix_width, matrix_height, start_pixel_x, start_pixel_y);

  cudaMemcpy(h_rgb_image, d_rgb_image, 3 * matrix_width * matrix_height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

  cudaFree(d_rgb_image);
  cudaFree(d_edge_matrix);
  cudaFree(d_cutout_matrix);
  cudaFree(d_done);
}

/**
 * First step of the cutout process.
 * Each gradient pixel with a value above the threshold is considered a border.
 **/
__global__ void draw_edges_on_cutout_matrix(unsigned char *edge_matrix, unsigned char *cutout_matrix, int matrix_width, int matrix_height, int start_pixel_x, int start_pixel_y, int threshold) {
  int globalIdxX = threadIdx.x + (blockIdx.x * blockDim.x);
  int globalIdxY = threadIdx.y + (blockIdx.y * blockDim.y);
  const int GLOBAL_IDX = globalIdxY * matrix_width + globalIdxX;

  if (globalIdxX < matrix_width && globalIdxY < matrix_height && threshold < edge_matrix[GLOBAL_IDX]) {
    cutout_matrix[GLOBAL_IDX] = 'B'; 
  }
  
  if (start_pixel_x == globalIdxX && start_pixel_y == globalIdxY) {
    cutout_matrix[start_pixel_y*matrix_width + start_pixel_x] = 'A';
  }
}

/**
 * Main part of the cutout process.
 * Loops over a cutout matrix from the start pixel to fill the shape it is in.
 **/
__global__ void cutout_algorithm(unsigned char *cutout_matrix, int matrix_width, int matrix_height, int *done) {
  int globalIdxX = threadIdx.x + (blockIdx.x * blockDim.x);
  int globalIdxY = threadIdx.y + (blockIdx.y * blockDim.y);
  int localIdxX = threadIdx.x;
  int localIdxY = threadIdx.y;

  __shared__ int shared_done;

  if (localIdxX == 0 && localIdxY == 0) {
    shared_done = 1; // Initialize the variable of the block
  }
  
  __syncthreads();

  // Process
  if (cutout_matrix[globalIdxY*matrix_width + globalIdxX] == 'A') {
    // Active pixel
    if (cutout_matrix[globalIdxY*matrix_width + globalIdxX-1] == 'D') {
      cutout_matrix[globalIdxY*matrix_width + globalIdxX-1] = 'A';
      shared_done = 0;
    }
    
    if (cutout_matrix[globalIdxY*matrix_width + globalIdxX+1] == 'D') {
      cutout_matrix[globalIdxY*matrix_width + globalIdxX+1] = 'A';
      shared_done = 0;
    }
    
    if (cutout_matrix[(globalIdxY-1)*matrix_width + globalIdxX] == 'D') {
      cutout_matrix[(globalIdxY-1)*matrix_width + globalIdxX] = 'A';
      shared_done = 0;
    }
    
    if (cutout_matrix[(globalIdxY+1)*matrix_width + globalIdxX] == 'D') {
      cutout_matrix[(globalIdxY+1)*matrix_width + globalIdxX] = 'A';
      shared_done = 0;
    }
    
    cutout_matrix[globalIdxY*matrix_width + globalIdxX] = 'M';
  }
 
  if (localIdxX == 0 && localIdxY == 0 && shared_done == 0) {
    *done = 0;
  }
}

__global__ void apply_cutout(unsigned char *cutout_matrix, unsigned char *output_image, int image_width, int image_height, int start_pixel_x, int start_pixel_y) { 
  int globalIdxX = threadIdx.x + (blockIdx.x * blockDim.x);
  int globalIdxY = threadIdx.y + (blockIdx.y * blockDim.y);
  const int GLOBAL_IDX = globalIdxY * image_width + globalIdxX;
 
  if (globalIdxX == start_pixel_x && globalIdxY == start_pixel_y) {
    output_image[3 * (GLOBAL_IDX)] = 255;
    output_image[3 * (GLOBAL_IDX) + 1] = 0; 
    output_image[3 * (GLOBAL_IDX) + 2] = 0; 
  } else if (cutout_matrix[globalIdxY*image_width + globalIdxX] != 'M') {
    output_image[3 * (GLOBAL_IDX)] = 0; 
    output_image[3 * (GLOBAL_IDX) + 1] = 0; 
    output_image[3 * (GLOBAL_IDX) + 2] = 0; 
  }
}

