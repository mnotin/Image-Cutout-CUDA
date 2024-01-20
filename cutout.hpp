#ifndef CUTOUT_H
#define CUTOUT_H

#include "main.hpp"

__global__ void draw_edges_on_cutout_matrix_kernel(unsigned char *edge_matrix, unsigned char *cutout_matrix, Dim matrix_dim, Vec2 start_pixel, int threshold);
__device__ __host__ unsigned char draw_edges_on_cutout_matrix_core(Vec2 index, unsigned char *edge_matrix, Dim matrix_dim, Vec2 start_pixel, int threshold);

__global__ void cutout_algorithm_kernel(unsigned char *cutout_matrix, Dim matrix_dim, int *done);
__device__ __host__ void cutout_algorithm_core(Vec2 index, unsigned char *cutout_matrix, Dim matrix_dim, int *done);

__global__ void apply_cutout_kernel(unsigned char *cutout_matrix, unsigned char *output_image, Dim image_dim, Vec2 start_pixel);
__device__ __host__ void apply_cutout_core(Vec2 index, unsigned char *cutout_matrix, unsigned char *output_image, Dim image_dim, Vec2 start_pixel);


namespace ProcessingUnitDevice {
  void cutout(unsigned char *h_rgb_image, unsigned char *h_edge_matrix, Dim matrix_dim, Vec2 start_pixel, int threshold);
}


namespace ProcessingUnitHost {
  void cutout(unsigned char *rgb_image, unsigned char *edge_matrix, Dim matrix_dim, Vec2 start_pixel, int threshold);
}

#endif // CUTOUT_H
