#include "sobel_feldman_kernel.hpp"

#include "../../core/edge_detection/sobel_feldman_core.hpp"



/**
 * Computes the global gradient of an image after being processed by the Sobel-Feldman operator.
 **/
__global__ void global_gradient_kernel(unsigned char *output_matrix, int *horizontal_edges, int *vertical_edges, dim3 matrix_dim) {
  int2 global_index = make_int2(threadIdx.x + (blockIdx.x * blockDim.x), threadIdx.y + (blockIdx.y * blockDim.y));

  if (global_index.x < matrix_dim.x && global_index.y < matrix_dim.y) {
    output_matrix[global_index.y*matrix_dim.x + global_index.x] = global_gradient_core(global_index, horizontal_edges, vertical_edges, matrix_dim);
  }
}

__global__ void angle_kernel(float *angle_matrix, int *horizontal_gradient, int *vertical_gradient, dim3 matrix_dim) {
  int2 global_index = make_int2(threadIdx.x + (blockIdx.x * blockDim.x), threadIdx.y + (blockIdx.y * blockDim.y));

  if (global_index.x < matrix_dim.x && global_index.y < matrix_dim.y) {
    angle_matrix[global_index.y*matrix_dim.x + global_index.x] = angle_core(global_index, horizontal_gradient, vertical_gradient, matrix_dim); 
  }
}

/**
 * Give a color to edges depending on their direction.
 **/
__global__ void edge_color_kernel(float *angle_matrix, unsigned char *output_image, dim3 image_dim) { 
  int2 global_index = make_int2(threadIdx.x + (blockIdx.x * blockDim.x), threadIdx.y + (blockIdx.y * blockDim.y));
  
  if (global_index.x < image_dim.x && global_index.y < image_dim.y) {
    edge_color_core(global_index, angle_matrix, output_image, image_dim);
  }
}
