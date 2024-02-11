#include <math.h>

#include "canny_launcher.hpp"

#include "../../kernel/edge_detection/canny_kernel.hpp"
#include "../../core/edge_detection/canny_core.hpp"
#include "../../main.hpp"

void ProcessingUnitDevice::canny(unsigned char *h_gradient_matrix, float *h_angle_matrix, dim3 matrix_dim, int canny_min, int canny_max) {
  dim3 block_dim(MATRIX_SIZE_PER_BLOCK, MATRIX_SIZE_PER_BLOCK);
  dim3 grid_dim(ceil((float) matrix_dim.x/MATRIX_SIZE_PER_BLOCK), ceil((float) matrix_dim.y/MATRIX_SIZE_PER_BLOCK));
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

  // Normally part of the Canny edge detector process.
  // Can allow the cutout process to enter an object when small holes are present in the edges of this object.
  // This kernel will be kept as a reference on the work done on this feature, but it should
  // probably not be used here with regards to the objectives of this application.
  //non_maximum_suppression_kernel<<<grid_dim, block_dim>>>(d_gradient_matrix, d_angle_matrix, matrix_dim);

  histeresis_thresholding_init_kernel<<<grid_dim, block_dim>>>(d_gradient_matrix, d_ht_matrix, matrix_dim, canny_min, canny_max);
  while (h_done == 0) {
    h_done = 1;
    cudaMemcpy(d_done, &h_done, sizeof(int), cudaMemcpyHostToDevice);
    histeresis_thresholding_loop_kernel<<<grid_dim, block_dim>>>(d_ht_matrix, matrix_dim, d_done);
    
    Canny::transfer_edges_between_blocks_kernel<<<grid_dim, block_dim>>>(d_ht_matrix, matrix_dim, d_done);
    cudaMemcpy(&h_done, d_done, sizeof(int), cudaMemcpyDeviceToHost);
  }
  histeresis_thresholding_end_kernel<<<grid_dim, block_dim>>>(d_gradient_matrix, d_ht_matrix, matrix_dim);

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
  
  // First step: non maximum suppression
  for (index.y = 0; index.y < matrix_dim.y; index.y++) {
    for (index.x = 0; index.x < matrix_dim.x; index.x++) {
      gradient_matrix_buffer[index.y*matrix_dim.x + index.x] = gradient_matrix[index.y*matrix_dim.x + index.x];
    }
  }
  // Normally part of the Canny edge detector process.
  // Can allow the cutout process to enter an object when small holes are present in the edges of this object.
  // This function call will be kept as a reference on the work done on this feature, but it should
  // probably not be used here with regards to the objectives of this application.
  /*for (index.y = 0; index.y < matrix_dim.y; index.y++) {
    for (index.x = 0; index.x < matrix_dim.x; index.x++) {
      gradient_matrix_buffer[index.y*matrix_dim.x + index.x] = non_maximum_suppression_core(index, gradient_matrix, angle_matrix, matrix_dim);
    }
  }*/
  for (index.y = 0; index.y < matrix_dim.y; index.y++) {
    for (index.x = 0; index.x < matrix_dim.x; index.x++) {
      gradient_matrix[index.y*matrix_dim.x + index.x] = gradient_matrix_buffer[index.y*matrix_dim.x + index.x];
    }
  }
 
  // Second step: histeresis thresholding init
  for (index.y = 0; index.y < matrix_dim.y; index.y++) {
    for (index.x = 0; index.x < matrix_dim.x; index.x++) {
      ht_matrix[index.y*matrix_dim.x + index.x] = histeresis_thresholding_init_core(index, gradient_matrix, matrix_dim, canny_min, canny_max);
    }
  }

  // Third step: histeresis thresholding loop
  while (done == 0) {
    done = 1;

    for (index.y = 0; index.y < matrix_dim.y; index.y++) {
      for (index.x = 0; index.x < matrix_dim.x; index.x++) {
        histeresis_thresholding_loop_core(index, ht_matrix, matrix_dim, make_int2(matrix_dim.x, matrix_dim.y), &done);
      }
    }
  }

  // Fourth step: histeresis thresholding end
  for (index.y = 0; index.y < matrix_dim.y; index.y++) {
    for (index.x = 0; index.x < matrix_dim.x; index.x++) {
      gradient_matrix[index.y*matrix_dim.x + index.x] = histeresis_thresholding_end_core(index, ht_matrix, matrix_dim);
    }
  }
}
