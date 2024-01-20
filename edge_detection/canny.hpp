#ifndef CANNY_HPP
#define CANNY_HPP

#include "../main.hpp"

__global__ void non_maximum_suppression_kernel(unsigned char *gradient_matrix, float *angle_matrix, Dim matrix_dim);
__device__ __host__ unsigned char non_maximum_suppression_core(Vec2 index, unsigned char *gradient_matrix, float *angle_matrix, Dim matrix_dim);

__global__ void histeresis_thresholding_init_kernel(unsigned char *gradient_matrix, unsigned char *ht_matrix, Dim matrix_dim, int canny_min, int canny_max);
__device__ __host__ unsigned char histeresis_thresholding_init_core(Vec2 index, unsigned char *gradient_matrix, Dim matrix_dim, int canny_min, int canny_max);

__global__ void histeresis_thresholding_loop_kernel(unsigned char *ht_matrix, Dim matrix_dim, int *done);
__device__ __host__ void histeresis_thresholding_loop_core(Vec2 index, unsigned char *ht_matrix, Dim matrix_dim, int *done);

__global__ void histeresis_thresholding_end_kernel(unsigned char *gradient_matrix, unsigned char *ht_matrix, Dim matrix_dim);
__device__ __host__ unsigned char histeresis_thresholding_end_core(Vec2 index, unsigned char *ht_matrix, Dim matrix_dim);
__device__ __host__ char get_color_canny(float angle);


namespace ProcessingUnitDevice {
  void canny(unsigned char *h_gradient_matrix, float *h_angle_matrix, Dim matrix_dim, int canny_min, int canny_max);
}


namespace ProcessingUnitHost {
  void canny(unsigned char *gradient_matrix, float *angle_matrix, Dim matrix_dim, int canny_min, int canny_max);
}

#endif // CANNY_HPP
