#ifndef CUTOUT_H
#define CUTOUT_H

#include "main.hpp"

__global__ void draw_edges_on_cutout_matrix_kernel(unsigned char *edge_matrix, char *micro_cutout_matrix, dim3 matrix_dim, int2 start_pixel, int threshold, char *macro_cutout_matrix);
__device__ __host__ char draw_edges_on_cutout_matrix_core(int2 index, unsigned char *edge_matrix, dim3 matrix_dim, int2 start_pixel, int threshold);

__global__ void cutout_algorithm_kernel(char *cutout_matrix, dim3 matrix_dim, int *done);
__device__ __host__ void cutout_algorithm_core(int2 index, char *cutout_matrix, dim3 matrix_dim, int *done);


__global__ void apply_macro_to_micro_cutout_matrix_kernel(char *macro_cutout_matrix, char *micro_cutout_matrix, dim3 macro_matrix_dim, dim3 micro_matrix_dim);

__global__ void apply_cutout_kernel(char *micro_cutout_matrix, unsigned char *output_image, dim3 image_dim, int2 start_pixel);
__device__ __host__ void apply_cutout_core(int2 index, char *micro_cutout_matrix, unsigned char *output_image, dim3 image_dim, int2 start_pixel);


namespace ProcessingUnitDevice {
  void cutout(unsigned char *h_rgb_image, unsigned char *h_edge_matrix, dim3 matrix_dim, int2 start_pixel, int threshold);

  namespace Cutout {
    __global__ void transfer_edges_between_blocks_kernel(char *cutout_matrix, dim3 matrix_dim, int *done);
  }
}


namespace ProcessingUnitHost {
  void cutout(unsigned char *rgb_image, unsigned char *edge_matrix, dim3 matrix_dim, int2 start_pixel, int threshold);
}



#endif // CUTOUT_H
