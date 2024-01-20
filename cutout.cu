#include <math.h>
#include <unistd.h>

#include "cutout.hpp"
#include "main.hpp"

void ProcessingUnitDevice::cutout(unsigned char *h_rgb_image, unsigned char *h_edge_matrix, Dim matrix_dim, Vec2 start_pixel, int threshold) {
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
  draw_edges_on_cutout_matrix_kernel<<<blocks, threads>>>(d_edge_matrix, d_cutout_matrix, matrix_dim, start_pixel, threshold);

  while (h_done == 0) {
    h_done = 1;
    cudaMemcpy(d_done, &h_done, sizeof(int), cudaMemcpyHostToDevice);
    cutout_algorithm_kernel<<<blocks, threads>>>(d_cutout_matrix, matrix_dim, d_done);
    cudaMemcpy(&h_done, d_done, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
  }
  apply_cutout_kernel<<<blocks, threads>>>(d_cutout_matrix, d_rgb_image, matrix_dim, start_pixel);

  cudaMemcpy(h_rgb_image, d_rgb_image, 3 * matrix_dim.width * matrix_dim.height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

  cudaFree(d_rgb_image);
  cudaFree(d_edge_matrix);
  cudaFree(d_cutout_matrix);
  cudaFree(d_done);
}

void ProcessingUnitHost::cutout(unsigned char *rgb_image, unsigned char *edge_matrix, Dim matrix_dim, Vec2 start_pixel, int threshold) {
  int done = 0;
  unsigned char cutout_matrix[matrix_dim.height * matrix_dim.width];
  
  for (int i = 0; i < matrix_dim.height; i++) {
    for (int j = 0; j < matrix_dim.width; j++) {
      cutout_matrix[i*matrix_dim.width + j] = 'D';
    }
  }
  
  Vec2 index;
  for (index.y = 0; index.y < matrix_dim.height; index.y++) {
    for (index.x = 0; index.x < matrix_dim.width; index.x++) {
      cutout_matrix[index.y*matrix_dim.width + index.x] = draw_edges_on_cutout_matrix_core(index, edge_matrix, matrix_dim, start_pixel, threshold);
    }
  }

  while (done == 0) {
    done = 1;
    for (index.y = 0; index.y < matrix_dim.width; index.y++) {
      for (index.x = 0; index.x < matrix_dim.height; index.x++) {
        cutout_algorithm_core(index, cutout_matrix, matrix_dim, &done);
      }
    }
  }
  
  for (index.y = 0; index.y < matrix_dim.height; index.y++) {
    for (index.x = 0; index.x < matrix_dim.width; index.x++) {
      apply_cutout_core(index, cutout_matrix, rgb_image, matrix_dim, start_pixel);
    }
  }
}

/**
 * First step of the cutout process.
 * Each gradient pixel with a value above the threshold is considered a border.
 **/
__global__ void draw_edges_on_cutout_matrix_kernel(unsigned char *edge_matrix, unsigned char *cutout_matrix, Dim matrix_dim, Vec2 start_pixel, int threshold) {
  Vec2 index;
  index.x = threadIdx.x + (blockIdx.x * blockDim.x);
  index.y = threadIdx.y + (blockIdx.y * blockDim.y);
  
  cutout_matrix[index.y*matrix_dim.width + index.x] = draw_edges_on_cutout_matrix_core(index, edge_matrix, matrix_dim, start_pixel, threshold);
}

__device__ __host__ unsigned char draw_edges_on_cutout_matrix_core(Vec2 index, unsigned char *edge_matrix, Dim matrix_dim, Vec2 start_pixel, int threshold) {
  unsigned char result = 'D';

  if (index.x < matrix_dim.width && index.y < matrix_dim.height && threshold < edge_matrix[index.y*matrix_dim.width + index.x]) {
    result = 'B'; 
  }
  
  if (start_pixel.x == index.x && start_pixel.y == index.y) {
    result = 'A';
  }

  return result;
}

/**
 * Main part of the cutout process.
 * Loops over a cutout matrix from the start pixel to fill the shape it is in.
 **/
__global__ void cutout_algorithm_kernel(unsigned char *cutout_matrix, Dim matrix_dim, int *done) {
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

  cutout_algorithm_core(global_index, cutout_matrix, matrix_dim, &shared_done);

  // The first local thread has to wait for all the threads of the bloc to finish
  __syncthreads();
 
  if (local_index.x == 0 && local_index.y == 0 && shared_done == 0) {
    *done = 0;
  }
}

__device__ __host__ void cutout_algorithm_core(Vec2 index, unsigned char *cutout_matrix, Dim matrix_dim, int *done) {
  const int INT_INDEX = index.y*matrix_dim.width + index.x;
  if (cutout_matrix[INT_INDEX] == 'A') {
    // Active pixel
    if (0 < index.x && cutout_matrix[INT_INDEX-1] == 'D') {
      cutout_matrix[INT_INDEX-1] = 'A';
      *done = 0;
    }
    
    if (index.x < matrix_dim.width-1 && cutout_matrix[INT_INDEX+1] == 'D') {
      cutout_matrix[INT_INDEX+1] = 'A';
      *done = 0;
    }
    
    if (0 < index.y && cutout_matrix[INT_INDEX - matrix_dim.width] == 'D') {
      cutout_matrix[INT_INDEX - matrix_dim.width] = 'A';
      *done = 0;
    }
    
    if (index.y < matrix_dim.height-1 && cutout_matrix[INT_INDEX + matrix_dim.width] == 'D') {
      cutout_matrix[INT_INDEX + matrix_dim.width] = 'A';
      *done = 0;
    }
      
    cutout_matrix[INT_INDEX] = 'M'; // At the end of the loop, current pixel is marked
  }
}

__global__ void apply_cutout_kernel(unsigned char *cutout_matrix, unsigned char *output_image, Dim image_dim, Vec2 start_pixel) { 
  Vec2 index;
  index.x = threadIdx.x + (blockIdx.x * blockDim.x);
  index.y = threadIdx.y + (blockIdx.y * blockDim.y);
  
  apply_cutout_core(index, cutout_matrix, output_image, image_dim, start_pixel);
}

__device__ __host__ void apply_cutout_core(Vec2 index, unsigned char *cutout_matrix, unsigned char *output_image, Dim image_dim, Vec2 start_pixel) { 
  const int INT_INDEX = index.y*image_dim.width + index.x;

  if (index.x == start_pixel.x && index.y == start_pixel.y) {
    output_image[3 * (INT_INDEX)] = 255;
    output_image[3 * (INT_INDEX) + 1] = 0; 
    output_image[3 * (INT_INDEX) + 2] = 0; 
  } else if (cutout_matrix[INT_INDEX] == 'M') {
    output_image[3 * (INT_INDEX)] = 0; 
    output_image[3 * (INT_INDEX) + 1] = 0; 
    output_image[3 * (INT_INDEX) + 2] = 0; 
  }
}
