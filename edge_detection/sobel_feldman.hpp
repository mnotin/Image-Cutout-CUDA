#ifndef SOBEL_FELDMAN_HPP
#define SOBEL_FELDMAN_HPP

#include "../main.hpp"

__global__ void global_gradient_kernel(unsigned char *output_matrix, int *horizontal_edges, int *vertical_edges, dim3 matrix_dim);
__device__ __host__ unsigned char global_gradient_core(int2 index, int *horizontal_edges, int *vertical_edges, dim3 matrix_dim);
__global__ void angle_kernel(float *angle_matrix, int *horizontal_gradient, int *vertical_gradient, dim3 matrix_dim);
__device__ __host__ float angle_core(int2 index, int *horizontal_gradient, int *vertical_gradient, dim3 matrix_dim);

__global__ void edge_color_kernel(unsigned char *gradient_matrix, float *angle_matrix, unsigned char *output_image, dim3 image_dim);
__device__ __host__ void edge_color_core(int2 index, unsigned char *gradient_matrix, float *angle_matrix, unsigned char *output_image, dim3 image_dim);
__device__ __host__ char get_color_sobel(float angle);


namespace ProcessingUnitDevice {
  void sobel_feldman(unsigned char *h_input_matrix, unsigned char *h_gradient_matrix, float *h_angle_matrix, dim3 matrix_dim);
  void generate_edge_color(unsigned char *h_gradient_matrix, float *h_angle_matrix, unsigned char *h_output_image, dim3 matrix_dim);
}


namespace ProcessingUnitHost {
  void sobel_feldman(unsigned char *input_matrix, unsigned char *gradient_matrix, float *angle_matrix, dim3 matrix_dim);
  void generate_edge_color(unsigned char *gradient_matrix, float *angle_matrix, unsigned char *output_image, dim3 matrix_dim);
}

#endif // SOBEL_FELDMAN_HPP
