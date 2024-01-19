#include <math.h>
#include <unistd.h>

#include "cutout.hpp"
#include "main.hpp"

void cutout(unsigned char *h_rgb_image, unsigned char *h_edge_matrix, Dim matrix_dim, Vec2 start_pixel, int threshold) {
  int h_done = 0;
  unsigned char h_cutout_matrix[matrix_dim.height][matrix_dim.width];
  
  for (int i = 0; i < matrix_dim.height; i++) {
    for (int j = 0; j < matrix_dim.width; j++) {
      h_cutout_matrix[i][j] = 'D';
    }
  }

  unsigned char *d_rgb_image;
  unsigned char *d_edge_matrix;
  unsigned char *d_cutout_matrix;
  int *d_done;

  cudaMalloc((void **) &d_rgb_image, 3 * matrix_dim.width * matrix_dim.height * sizeof(unsigned char));
  cudaMalloc((void **) &d_edge_matrix, matrix_dim.width * matrix_dim.height * sizeof(unsigned char));
  cudaMalloc((void **) &d_cutout_matrix, matrix_dim.width * matrix_dim.height * sizeof(unsigned char));
  cudaMalloc((void **) &d_done, sizeof(int));

  cudaMemcpy(d_rgb_image, h_rgb_image, 3 * matrix_dim.width * matrix_dim.height * sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_edge_matrix, h_edge_matrix, matrix_dim.width * matrix_dim.height * sizeof(unsigned char), cudaMemcpyHostToDevice);
  for (int i = 0; i < matrix_dim.height; i++) {
    cudaMemcpy(d_cutout_matrix+i*matrix_dim.width, h_cutout_matrix[i], matrix_dim.width * sizeof(unsigned char), cudaMemcpyHostToDevice);
  }
  
  dim3 threads = dim3(MATRIX_SIZE_PER_BLOCK, MATRIX_SIZE_PER_BLOCK);
  dim3 blocks = dim3(matrix_dim.width/MATRIX_SIZE_PER_BLOCK, matrix_dim.height/MATRIX_SIZE_PER_BLOCK);
  draw_edges_on_cutout_matrix<<<blocks, threads>>>(d_edge_matrix, d_cutout_matrix, matrix_dim, start_pixel, threshold);

  while (h_done == 0) {
    h_done = 1;
    cudaMemcpy(d_done, &h_done, sizeof(int), cudaMemcpyHostToDevice);
    cutout_algorithm<<<blocks, threads>>>(d_cutout_matrix, matrix_dim, d_done);
    cudaMemcpy(&h_done, d_done, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
  }
  apply_cutout<<<blocks, threads>>>(d_cutout_matrix, d_rgb_image, matrix_dim, start_pixel);

  cudaMemcpy(h_rgb_image, d_rgb_image, 3 * matrix_dim.width * matrix_dim.height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

  cudaFree(d_rgb_image);
  cudaFree(d_edge_matrix);
  cudaFree(d_cutout_matrix);
  cudaFree(d_done);
}

/**
 * First step of the cutout process.
 * Each gradient pixel with a value above the threshold is considered a border.
 **/
__global__ void draw_edges_on_cutout_matrix(unsigned char *edge_matrix, unsigned char *cutout_matrix, Dim matrix_dim, Vec2 start_pixel, int threshold) {
  int globalIdxX = threadIdx.x + (blockIdx.x * blockDim.x);
  int globalIdxY = threadIdx.y + (blockIdx.y * blockDim.y);
  const int GLOBAL_IDX = globalIdxY * matrix_dim.width + globalIdxX;

  if (globalIdxX < matrix_dim.width && globalIdxY < matrix_dim.height && threshold < edge_matrix[GLOBAL_IDX]) {
    cutout_matrix[GLOBAL_IDX] = 'B'; 
  }
  
  if (start_pixel.x == globalIdxX && start_pixel.y == globalIdxY) {
    cutout_matrix[start_pixel.y*matrix_dim.width + start_pixel.x] = 'A';
  }
}

/**
 * Main part of the cutout process.
 * Loops over a cutout matrix from the start pixel to fill the shape it is in.
 **/
__global__ void cutout_algorithm(unsigned char *cutout_matrix, Dim matrix_dim, int *done) {
  int globalIdxX = threadIdx.x + (blockIdx.x * blockDim.x);
  int globalIdxY = threadIdx.y + (blockIdx.y * blockDim.y);
  int localIdxX = threadIdx.x;
  int localIdxY = threadIdx.y;
  const int GLOBAL_IDX = globalIdxY*matrix_dim.width + globalIdxX;

  __shared__ int shared_done;

  if (localIdxX == 0 && localIdxY == 0) {
    shared_done = 1; // Initialize the variable of the block
  }
  
  __syncthreads();

  // Process
  if (cutout_matrix[GLOBAL_IDX] == 'A') {
    // Active pixel
    if (0 < globalIdxX && cutout_matrix[GLOBAL_IDX-1] == 'D') {
      cutout_matrix[GLOBAL_IDX-1] = 'A';
      shared_done = 0;
    }
    
    if (globalIdxX < matrix_dim.width-1 && cutout_matrix[GLOBAL_IDX+1] == 'D') {
      cutout_matrix[GLOBAL_IDX+1] = 'A';
      shared_done = 0;
    }
    
    if (0 < globalIdxY && cutout_matrix[GLOBAL_IDX - matrix_dim.width] == 'D') {
      cutout_matrix[GLOBAL_IDX - matrix_dim.width] = 'A';
      shared_done = 0;
    }
    
    if (globalIdxY < matrix_dim.height-1 && cutout_matrix[GLOBAL_IDX + matrix_dim.width] == 'D') {
      cutout_matrix[GLOBAL_IDX + matrix_dim.width] = 'A';
      shared_done = 0;
    }
    
    cutout_matrix[GLOBAL_IDX] = 'M';
  }

  // The first local thread has to wait for all the threads of the bloc to finish
  __syncthreads();
 
  if (localIdxX == 0 && localIdxY == 0 && shared_done == 0) {
    *done = 0;
  }
}

__global__ void apply_cutout(unsigned char *cutout_matrix, unsigned char *output_image, Dim image_dim, Vec2 start_pixel) { 
  int globalIdxX = threadIdx.x + (blockIdx.x * blockDim.x);
  int globalIdxY = threadIdx.y + (blockIdx.y * blockDim.y);
  const int GLOBAL_IDX = globalIdxY * image_dim.width + globalIdxX;
 
  if (globalIdxX == start_pixel.x && globalIdxY == start_pixel.y) {
    output_image[3 * (GLOBAL_IDX)] = 255;
    output_image[3 * (GLOBAL_IDX) + 1] = 0; 
    output_image[3 * (GLOBAL_IDX) + 2] = 0; 
  } else if (cutout_matrix[globalIdxY*image_dim.width + globalIdxX] == 'M') {
    output_image[3 * (GLOBAL_IDX)] = 0; 
    output_image[3 * (GLOBAL_IDX) + 1] = 0; 
    output_image[3 * (GLOBAL_IDX) + 2] = 0; 
  }
}

