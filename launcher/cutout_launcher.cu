#include <math.h>
#include <iostream>

#include "cutout_launcher.hpp"

#include "../main.hpp"
#include "../core/cutout_core.hpp"
#include "../kernel/cutout_kernel.hpp"

void ProcessingUnitDevice::cutout(unsigned char *h_rgb_image, unsigned char *h_edge_matrix,
                                  dim3 matrix_dim, int2 cutout_start_pixel,
                                  int2 *tracking_start_pixel, int threshold
) {
  dim3 block_dim(MATRIX_SIZE_PER_BLOCK, MATRIX_SIZE_PER_BLOCK);
  dim3 grid_dim(ceil((float) matrix_dim.x/MATRIX_SIZE_PER_BLOCK), ceil((float) matrix_dim.y/MATRIX_SIZE_PER_BLOCK));
  
  int h_done = 0;
  dim3 macro_matrix_dim(grid_dim.x, grid_dim.y);
  char h_cutout_looking_pixels[2] = { 'D', '\0' };
  char h_tracking_looking_pixels[3] = { 'D', 'B', '\0' };
  int2 h_tracking_top_left = *tracking_start_pixel;
  int2 h_tracking_bottom_right = *tracking_start_pixel;

  unsigned char *d_rgb_image;
  unsigned char *d_edge_matrix;
  char *d_macro_cutout_matrix;
  char *d_micro_cutout_matrix;
  int *d_done;
  int2 *d_top_left_tracking;
  int2 *d_bottom_right_tracking;
  char *d_cutout_looking_pixels;
  char *d_tracking_looking_pixels;
  int2 *d_tracking_top_left;
  int2 *d_tracking_bottom_right;

  cudaMalloc(&d_rgb_image, 3 * matrix_dim.x * matrix_dim.y * sizeof(unsigned char));
  cudaMalloc(&d_edge_matrix, matrix_dim.x * matrix_dim.y * sizeof(unsigned char));
  cudaMalloc(&d_macro_cutout_matrix, macro_matrix_dim.x * macro_matrix_dim.y * sizeof(char));
  cudaMalloc(&d_micro_cutout_matrix, matrix_dim.x * matrix_dim.y * sizeof(char));
  cudaMalloc(&d_done, sizeof(int));
  cudaMalloc(&d_top_left_tracking, sizeof(int2));
  cudaMalloc(&d_bottom_right_tracking, sizeof(int2));
  cudaMalloc(&d_cutout_looking_pixels, sizeof(char) * 2);
  cudaMalloc(&d_tracking_looking_pixels, sizeof(char) * 3);
  cudaMalloc(&d_tracking_top_left, sizeof(int2));
  cudaMalloc(&d_tracking_bottom_right, sizeof(int2));
  
  cudaMemcpy(d_edge_matrix, h_edge_matrix, matrix_dim.x * matrix_dim.y * sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_cutout_looking_pixels, h_cutout_looking_pixels, 2 * sizeof(char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_tracking_top_left, &h_tracking_top_left, sizeof(int2), cudaMemcpyHostToDevice);
  cudaMemcpy(d_tracking_bottom_right, &h_tracking_bottom_right, sizeof(int2), cudaMemcpyHostToDevice);
  
  draw_edges_on_cutout_matrix_kernel<<<grid_dim, block_dim>>>(
    d_edge_matrix,
    d_micro_cutout_matrix,
    matrix_dim,
    cutout_start_pixel,
    *tracking_start_pixel,
    threshold,
    d_macro_cutout_matrix);

  // Macro
  while (h_done == 0) {
    h_done = 1; // Let's assume that the process is done
    cudaMemcpy(d_done, &h_done, sizeof(int), cudaMemcpyHostToDevice);
    cutout_algorithm_kernel<<<grid_dim, block_dim>>>(d_macro_cutout_matrix, macro_matrix_dim, d_done, d_cutout_looking_pixels);
    
    Cutout::transfer_edges_between_blocks_kernel<<<grid_dim, block_dim>>>(d_macro_cutout_matrix, macro_matrix_dim, d_done, d_cutout_looking_pixels);
    cudaMemcpy(&h_done, d_done, sizeof(int), cudaMemcpyDeviceToHost);
  }
  
  apply_macro_to_micro_cutout_matrix_kernel<<<grid_dim, block_dim>>>(d_macro_cutout_matrix, d_micro_cutout_matrix, macro_matrix_dim, matrix_dim);

  // Micro
  h_done = 0;
  while (h_done == 0) {
    h_done = 1; // Let's assume that the process is done
    cudaMemcpy(d_done, &h_done, sizeof(int), cudaMemcpyHostToDevice);
    cutout_algorithm_kernel<<<grid_dim, block_dim>>>(d_micro_cutout_matrix, matrix_dim, d_done, d_cutout_looking_pixels);
    
    Cutout::transfer_edges_between_blocks_kernel<<<grid_dim, block_dim>>>(d_micro_cutout_matrix, matrix_dim, d_done, d_cutout_looking_pixels);
    cudaMemcpy(&h_done, d_done, sizeof(int), cudaMemcpyDeviceToHost);
  }
  
  // Object tracking
  h_done = 0;
  cudaMemcpy(d_tracking_looking_pixels, h_tracking_looking_pixels, 3 * sizeof(char), cudaMemcpyHostToDevice);
  while (h_done == 0) {
    h_done = 1; // Let's assume that the process is done
    cudaMemcpy(d_done, &h_done, sizeof(int), cudaMemcpyHostToDevice);
    cutout_algorithm_kernel<<<grid_dim, block_dim>>>(d_micro_cutout_matrix, matrix_dim,
                                                     d_done, d_tracking_looking_pixels,
                                                     'T', d_tracking_top_left, d_tracking_bottom_right);
    
    Cutout::transfer_edges_between_blocks_kernel<<<grid_dim, block_dim>>>(d_micro_cutout_matrix,
                                                                          matrix_dim, d_done, d_tracking_looking_pixels, 'T');
    cudaMemcpy(&h_done, d_done, sizeof(int), cudaMemcpyDeviceToHost);
  }
  cudaMemcpy(&h_tracking_top_left, d_tracking_top_left, sizeof(int2), cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_tracking_bottom_right, d_tracking_bottom_right, sizeof(int2), cudaMemcpyDeviceToHost);
  if (abs(tracking_start_pixel->x - (((h_tracking_bottom_right.x - h_tracking_top_left.x) / 2) + h_tracking_top_left.x)) < 99 &&
      abs(tracking_start_pixel->y - (((h_tracking_bottom_right.y - h_tracking_top_left.y) / 2) + h_tracking_top_left.y)) < 99) {
    // Check that the center of gravity did not move too much
    tracking_start_pixel->x = ((h_tracking_bottom_right.x - h_tracking_top_left.x) / 2) + h_tracking_top_left.x;
    tracking_start_pixel->y = ((h_tracking_bottom_right.y - h_tracking_top_left.y) / 2) + h_tracking_top_left.y;
  }
  std::cout <<  tracking_start_pixel->x << "-" << tracking_start_pixel->y << std::endl;
   
  cudaMemcpy(d_rgb_image, h_rgb_image, 3 * matrix_dim.x * matrix_dim.y * sizeof(unsigned char), cudaMemcpyHostToDevice);
  apply_cutout_kernel<<<grid_dim, block_dim>>>(d_micro_cutout_matrix, d_rgb_image, matrix_dim, cutout_start_pixel);
  cudaMemcpy(h_rgb_image, d_rgb_image, 3 * matrix_dim.x * matrix_dim.y * sizeof(unsigned char), cudaMemcpyDeviceToHost);

  cudaFree(d_rgb_image);
  cudaFree(d_edge_matrix);
  cudaFree(d_macro_cutout_matrix);
  cudaFree(d_micro_cutout_matrix);
  cudaFree(d_done);
  cudaFree(d_top_left_tracking);
  cudaFree(d_bottom_right_tracking);
  cudaFree(d_cutout_looking_pixels);
  cudaFree(d_tracking_looking_pixels);
  cudaFree(d_tracking_top_left);
  cudaFree(d_tracking_bottom_right);
}

void ProcessingUnitHost::cutout(unsigned char *rgb_image, unsigned char *edge_matrix,
                                dim3 matrix_dim, int2 cutout_start_pixel,
                                int2 *tracking_start_pixel, int threshold
) {
  int done = 0;
  char cutout_matrix[matrix_dim.y * matrix_dim.x];
  char cutout_looking_pixels[2] = {'B', '\0'};
  char tracking_looking_pixels[3] = {'B', 'D', '\0'};
  
  for (int i = 0; i < matrix_dim.y; i++) {
    for (int j = 0; j < matrix_dim.x; j++) {
      cutout_matrix[i*matrix_dim.x + j] = 'D';
    }
  }
  
  int2 index;
  for (index.y = 0; index.y < matrix_dim.y; index.y++) {
    for (index.x = 0; index.x < matrix_dim.x; index.x++) {
      cutout_matrix[index.y*matrix_dim.x + index.x] =
        draw_edges_on_cutout_matrix_core(index, edge_matrix, matrix_dim, cutout_start_pixel, *tracking_start_pixel, threshold);
    }
  }

  while (done == 0) {
    done = 1;
    for (index.y = 0; index.y < matrix_dim.y; index.y++) {
      for (index.x = 0; index.x < matrix_dim.x; index.x++) {
        cutout_algorithm_core(index, cutout_matrix, matrix_dim, make_int2(matrix_dim.x, matrix_dim.y), &done, cutout_looking_pixels);
      }
    }
  }
  
  done = 0;
  
  // Tracking
  while (done == 0) {
    done = 1;
    for (index.y = 0; index.y < matrix_dim.y; index.y++) {
      for (index.x = 0; index.x < matrix_dim.x; index.x++) {
        cutout_algorithm_core(index, cutout_matrix, matrix_dim, make_int2(matrix_dim.x, matrix_dim.y), &done, tracking_looking_pixels, 'T');
      }
    }
  }
  
  for (index.y = 0; index.y < matrix_dim.y; index.y++) {
    for (index.x = 0; index.x < matrix_dim.x; index.x++) {
      apply_cutout_core(index, cutout_matrix, rgb_image, matrix_dim, cutout_start_pixel);
    }
  }
}
