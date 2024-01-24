#include <math.h>
#include <unistd.h>
#include <iostream>

#include "cutout.hpp"
#include "main.hpp"

void ProcessingUnitDevice::cutout(unsigned char *h_rgb_image, unsigned char *h_edge_matrix, dim3 matrix_dim, int2 start_pixel, int threshold) {
  dim3 threads = dim3(MATRIX_SIZE_PER_BLOCK, MATRIX_SIZE_PER_BLOCK);
  dim3 blocks = dim3(matrix_dim.x/MATRIX_SIZE_PER_BLOCK, matrix_dim.y/MATRIX_SIZE_PER_BLOCK);

  int h_done = 0;
  dim3 macro_matrix_dim(blocks.x, blocks.y);

  unsigned char *d_rgb_image;
  unsigned char *d_edge_matrix;
  char *d_macro_cutout_matrix;
  char *d_micro_cutout_matrix;
  int *d_done;

  cudaMalloc(&d_rgb_image, 3 * matrix_dim.x * matrix_dim.y * sizeof(unsigned char));
  cudaMalloc(&d_edge_matrix, matrix_dim.x * matrix_dim.y * sizeof(unsigned char));
  cudaMalloc(&d_macro_cutout_matrix, blocks.x * blocks.y * sizeof(char));
  cudaMalloc(&d_micro_cutout_matrix, matrix_dim.x * matrix_dim.y * sizeof(char));
  cudaMalloc(&d_done, sizeof(int));

  cudaMemcpy(d_edge_matrix, h_edge_matrix, matrix_dim.x * matrix_dim.y * sizeof(unsigned char), cudaMemcpyHostToDevice);
  
  draw_edges_on_cutout_matrix_kernel<<<blocks, threads>>>(
    d_edge_matrix,
    d_micro_cutout_matrix,
    matrix_dim,
    start_pixel,
    threshold,
    d_macro_cutout_matrix);

  // Macro
  while (h_done == 0) {
    h_done = 1; // Let's assume that the process is done
    cudaMemcpy(d_done, &h_done, sizeof(int), cudaMemcpyHostToDevice);
    cutout_algorithm_kernel<<<blocks, threads>>>(d_macro_cutout_matrix, macro_matrix_dim, d_done);
    
    Cutout::transfer_edges_between_blocks_kernel<<<blocks, threads>>>(d_macro_cutout_matrix, macro_matrix_dim, d_done);
    cudaMemcpy(&h_done, d_done, sizeof(int), cudaMemcpyDeviceToHost);
  }
  
  apply_macro_to_micro_cutout_matrix_kernel<<<blocks, threads>>>(d_macro_cutout_matrix, d_micro_cutout_matrix, macro_matrix_dim, matrix_dim);
  h_done = 0;

  // Micro
  while (h_done == 0) {
    h_done = 1; // Let's assume that the process is done
    cudaMemcpy(d_done, &h_done, sizeof(int), cudaMemcpyHostToDevice);
    cutout_algorithm_kernel<<<blocks, threads>>>(d_micro_cutout_matrix, matrix_dim, d_done);
    
    Cutout::transfer_edges_between_blocks_kernel<<<blocks, threads>>>(d_micro_cutout_matrix, matrix_dim, d_done);
    cudaMemcpy(&h_done, d_done, sizeof(int), cudaMemcpyDeviceToHost);
  }
  
  cudaMemcpy(d_rgb_image, h_rgb_image, 3 * matrix_dim.x * matrix_dim.y * sizeof(unsigned char), cudaMemcpyHostToDevice);
  apply_cutout_kernel<<<blocks, threads>>>(d_micro_cutout_matrix, d_rgb_image, matrix_dim, start_pixel);
  cudaMemcpy(h_rgb_image, d_rgb_image, 3 * matrix_dim.x * matrix_dim.y * sizeof(unsigned char), cudaMemcpyDeviceToHost);

  cudaFree(d_rgb_image);
  cudaFree(d_edge_matrix);
  cudaFree(d_macro_cutout_matrix);
  cudaFree(d_micro_cutout_matrix);
  cudaFree(d_done);
}

void ProcessingUnitHost::cutout(unsigned char *rgb_image, unsigned char *edge_matrix, dim3 matrix_dim, int2 start_pixel, int threshold) {
  int done = 0;
  char cutout_matrix[matrix_dim.y * matrix_dim.x];
  
  for (int i = 0; i < matrix_dim.y; i++) {
    for (int j = 0; j < matrix_dim.x; j++) {
      cutout_matrix[i*matrix_dim.x + j] = 'D';
    }
  }
  
  int2 index;
  for (index.y = 0; index.y < matrix_dim.y; index.y++) {
    for (index.x = 0; index.x < matrix_dim.x; index.x++) {
      cutout_matrix[index.y*matrix_dim.x + index.x] = draw_edges_on_cutout_matrix_core(index, edge_matrix, matrix_dim, start_pixel, threshold);
    }
  }

  while (done == 0) {
    done = 1;
    for (index.y = 0; index.y < matrix_dim.y; index.y++) {
      for (index.x = 0; index.x < matrix_dim.x; index.x++) {
        cutout_algorithm_core(index, cutout_matrix, matrix_dim, &done);
      }
    }
  }
  
  for (index.y = 0; index.y < matrix_dim.y; index.y++) {
    for (index.x = 0; index.x < matrix_dim.x; index.x++) {
      apply_cutout_core(index, cutout_matrix, rgb_image, matrix_dim, start_pixel);
    }
  }
}

/**
 * First step of the cutout process.
 * Each gradient pixel with a value above the threshold is considered a border.
 **/
__global__ void draw_edges_on_cutout_matrix_kernel(unsigned char *edge_matrix, char *micro_cutout_matrix, dim3 matrix_dim, int2 start_pixel, int threshold, char *macro_cutout_matrix) {
  int2 global_index = make_int2(threadIdx.x + (blockIdx.x * blockDim.x), threadIdx.y + (blockIdx.y * blockDim.y));
  int2 local_index = make_int2(threadIdx.x, threadIdx.y);
  __shared__ bool block_contains_edge;
  __shared__ bool block_contains_start_pixel;
  
  if (local_index.x == 0 && local_index.y == 0) {
    block_contains_edge = false;
    block_contains_start_pixel = false;
  }
  __syncthreads();
  
  char result = draw_edges_on_cutout_matrix_core(global_index, edge_matrix, matrix_dim, start_pixel, threshold);
  micro_cutout_matrix[global_index.y*matrix_dim.x + global_index.x] = result;

  if (result == 'B') {
    // This block contains at least one border
    block_contains_edge = true;
  } else if (result == 'M') {
    block_contains_start_pixel = true;
  }
  __syncthreads();
  if (local_index.x == 0 && local_index.y == 0) {
    if (block_contains_edge) {
      macro_cutout_matrix[blockIdx.y*gridDim.x + blockIdx.x] = 'B';
    } else if (block_contains_start_pixel) {
      macro_cutout_matrix[blockIdx.y*gridDim.x + blockIdx.x] = 'M';
    } else {
      macro_cutout_matrix[blockIdx.y*gridDim.x + blockIdx.x] = 'D';
    }
  }
}

__device__ __host__ char draw_edges_on_cutout_matrix_core(int2 index, unsigned char *edge_matrix, dim3 matrix_dim, int2 start_pixel, int threshold) {
  char result = 'D'; // Discard

  if (index.x < matrix_dim.x && index.y < matrix_dim.y && threshold < edge_matrix[index.y*matrix_dim.x + index.x]) {
    result = 'B'; // Border
  }
  
  if (start_pixel.x == index.x && start_pixel.y == index.y) {
    result = 'M'; // Marked
  }

  return result;
}

/**
 * Main part of the cutout process.
 * Loops over a cutout matrix from the start pixel to fill the shape it is in.
 **/
__global__ void cutout_algorithm_kernel(char *cutout_matrix, dim3 matrix_dim, int *done) {
  int2 global_index = make_int2(threadIdx.x + (blockIdx.x * blockDim.x), threadIdx.y + (blockIdx.y * blockDim.y));
  int2 local_index = make_int2(threadIdx.x, threadIdx.y);

  dim3 shared_matrix_dim(MATRIX_SIZE_PER_BLOCK, MATRIX_SIZE_PER_BLOCK);

  __shared__ int shared_done;
  __shared__ char shared_cutout_matrix[MATRIX_SIZE_PER_BLOCK*MATRIX_SIZE_PER_BLOCK];
  shared_cutout_matrix[local_index.y*MATRIX_SIZE_PER_BLOCK + local_index.x] = cutout_matrix[global_index.y*matrix_dim.x + global_index.x];

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

    cutout_algorithm_core(local_index, shared_cutout_matrix, shared_matrix_dim, &shared_done);

    __syncthreads();

    if (local_index.x == 0 && local_index.y == 0 && shared_done == 0) {
      // At least one block had to update a pixel so we will need
      // to rerun them all at least once
      *done = 0;
    }
  }

  __syncthreads();
 
  // Write the result back to global memory
  cutout_matrix[global_index.y*matrix_dim.x + global_index.x] =
    shared_cutout_matrix[local_index.y*MATRIX_SIZE_PER_BLOCK + local_index.x];
}

__device__ __host__ void cutout_algorithm_core(int2 index, char *cutout_matrix, dim3 matrix_dim, int *done) {
  const int INT_INDEX = index.y*matrix_dim.x + index.x;

  if (cutout_matrix[INT_INDEX] == 'D') {
    if (0 < index.x && cutout_matrix[INT_INDEX-1] == 'M' || 
        index.x < matrix_dim.x-1 && cutout_matrix[INT_INDEX+1] == 'M' ||
        0 < index.y && cutout_matrix[INT_INDEX - matrix_dim.x] == 'M' ||
        index.y < matrix_dim.y-1 && cutout_matrix[INT_INDEX + matrix_dim.x] == 'M') {
      cutout_matrix[INT_INDEX] = 'M';
      *done = 0;
    }
  }
}

__global__ void ProcessingUnitDevice::Cutout::transfer_edges_between_blocks_kernel(char *cutout_matrix, dim3 matrix_dim, int *done) {
  int2 global_index = make_int2(threadIdx.x + (blockIdx.x * blockDim.x), threadIdx.y + (blockIdx.y * blockDim.y));
  int2 local_index = make_int2(threadIdx.x, threadIdx.y);

  if (local_index.y == 0 && 0 < global_index.y ||
      local_index.y == MATRIX_SIZE_PER_BLOCK-1 && global_index.y < matrix_dim.y-1 ||
      local_index.x == 0 && 0 < global_index.x ||
      local_index.x == MATRIX_SIZE_PER_BLOCK-1 && global_index.x < matrix_dim.x-1) {
      cutout_algorithm_core(global_index, cutout_matrix, matrix_dim, done);
  }
}

__global__ void apply_macro_to_micro_cutout_matrix_kernel(char *macro_cutout_matrix, char *micro_cutout_matrix, dim3 macro_matrix_dim, dim3 micro_matrix_dim) {
  int2 global_index = make_int2(threadIdx.x + (blockIdx.x * blockDim.x), threadIdx.y + (blockIdx.y * blockDim.y));
  int2 local_index = make_int2(threadIdx.x, threadIdx.y);

  __shared__ char macro_matrix_content;

  if (local_index.x == 0 && local_index.y == 0) {
    macro_matrix_content = macro_cutout_matrix[blockIdx.y*macro_matrix_dim.x + blockIdx.x];
  }

  __syncthreads();

  if (macro_matrix_content == 'M') {
    micro_cutout_matrix[global_index.y*micro_matrix_dim.x + global_index.x] = 'M';
  }
}

__global__ void apply_cutout_kernel(char *cutout_matrix, unsigned char *output_image, dim3 image_dim, int2 start_pixel) { 
  int2 index  = make_int2(threadIdx.x + (blockIdx.x * blockDim.x), threadIdx.y + (blockIdx.y * blockDim.y));
  
  apply_cutout_core(index, cutout_matrix, output_image, image_dim, start_pixel);
}

__device__ __host__ void apply_cutout_core(int2 index, char *cutout_matrix, unsigned char *output_image, dim3 image_dim, int2 start_pixel) {
  const int INT_INDEX = index.y*image_dim.x + index.x;

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
