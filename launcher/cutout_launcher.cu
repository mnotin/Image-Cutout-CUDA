#include <math.h>
#include <iostream>

#include "cutout_launcher.hpp"

#include "../main.hpp"
#include "../core/cutout_core.hpp"
#include "../kernel/cutout_kernel.hpp"

void ProcessingUnitDevice::cutout(unsigned char *h_rgb_image, unsigned char *h_edge_matrix, dim3 matrix_dim, int2 start_pixel, int threshold) {
  dim3 block_dim(MATRIX_SIZE_PER_BLOCK, MATRIX_SIZE_PER_BLOCK);
  dim3 grid_dim(ceil((float) matrix_dim.x/MATRIX_SIZE_PER_BLOCK), ceil((float) matrix_dim.y/MATRIX_SIZE_PER_BLOCK));
  
  int h_done = 0;
  dim3 macro_matrix_dim(grid_dim.x, grid_dim.y);

  unsigned char *d_rgb_image;
  unsigned char *d_edge_matrix;
  char *d_macro_cutout_matrix;
  char *d_micro_cutout_matrix;
  int *d_done;

  cudaMalloc(&d_rgb_image, 3 * matrix_dim.x * matrix_dim.y * sizeof(unsigned char));
  cudaMalloc(&d_edge_matrix, matrix_dim.x * matrix_dim.y * sizeof(unsigned char));
  cudaMalloc(&d_macro_cutout_matrix, macro_matrix_dim.x * macro_matrix_dim.y * sizeof(char));
  cudaMalloc(&d_micro_cutout_matrix, matrix_dim.x * matrix_dim.y * sizeof(char));
  cudaMalloc(&d_done, sizeof(int));

  cudaMemcpy(d_edge_matrix, h_edge_matrix, matrix_dim.x * matrix_dim.y * sizeof(unsigned char), cudaMemcpyHostToDevice);
  
  draw_edges_on_cutout_matrix_kernel<<<grid_dim, block_dim>>>(
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
    cutout_algorithm_kernel<<<grid_dim, block_dim>>>(d_macro_cutout_matrix, macro_matrix_dim, d_done);
    
    Cutout::transfer_edges_between_blocks_kernel<<<grid_dim, block_dim>>>(d_macro_cutout_matrix, macro_matrix_dim, d_done);
    cudaMemcpy(&h_done, d_done, sizeof(int), cudaMemcpyDeviceToHost);
  }
  
  apply_macro_to_micro_cutout_matrix_kernel<<<grid_dim, block_dim>>>(d_macro_cutout_matrix, d_micro_cutout_matrix, macro_matrix_dim, matrix_dim);
  h_done = 0;

  // Micro
  while (h_done == 0) {
    h_done = 1; // Let's assume that the process is done
    cudaMemcpy(d_done, &h_done, sizeof(int), cudaMemcpyHostToDevice);
    cutout_algorithm_kernel<<<grid_dim, block_dim>>>(d_micro_cutout_matrix, matrix_dim, d_done);
    
    Cutout::transfer_edges_between_blocks_kernel<<<grid_dim, block_dim>>>(d_micro_cutout_matrix, matrix_dim, d_done);
    cudaMemcpy(&h_done, d_done, sizeof(int), cudaMemcpyDeviceToHost);
  }
  
  cudaMemcpy(d_rgb_image, h_rgb_image, 3 * matrix_dim.x * matrix_dim.y * sizeof(unsigned char), cudaMemcpyHostToDevice);
  apply_cutout_kernel<<<grid_dim, block_dim>>>(d_micro_cutout_matrix, d_rgb_image, matrix_dim, start_pixel);
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
        cutout_algorithm_core(index, cutout_matrix, matrix_dim, make_int2(matrix_dim.x, matrix_dim.y), &done);
      }
    }
  }
  
  for (index.y = 0; index.y < matrix_dim.y; index.y++) {
    for (index.x = 0; index.x < matrix_dim.x; index.x++) {
      apply_cutout_core(index, cutout_matrix, rgb_image, matrix_dim, start_pixel);
    }
  }
}
