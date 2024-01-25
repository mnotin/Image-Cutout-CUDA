#ifndef CANNY_HPP
#define CANNY_HPP

#include "../main.hpp"

__global__ void non_maximum_suppression_kernel(unsigned char *gradient_matrix, float *angle_matrix, dim3 matrix_dim);
__device__ __host__ unsigned char non_maximum_suppression_core(int2 index, unsigned char *gradient_matrix, float *angle_matrix, dim3 matrix_dim);

__global__ void histeresis_thresholding_init_kernel(unsigned char *gradient_matrix, char *ht_matrix, dim3 matrix_dim, int canny_min, int canny_max);
__device__ __host__ char histeresis_thresholding_init_core(int2 index, unsigned char *gradient_matrix, dim3 matrix_dim, int canny_min, int canny_max);

__global__ void histeresis_thresholding_loop_kernel(char *ht_matrix, dim3 matrix_dim, int *done);
__device__ __host__ void histeresis_thresholding_loop_core(int2 index, char *ht_matrix, dim3 matrix_dim, int2 read_limit, int *done);

__global__ void histeresis_thresholding_end_kernel(unsigned char *gradient_matrix, char *ht_matrix, dim3 matrix_dim);
__device__ __host__ unsigned char histeresis_thresholding_end_core(int2 index, char *ht_matrix, dim3 matrix_dim);
__device__ __host__ char get_color_canny(float angle);


namespace ProcessingUnitDevice {
  void canny(unsigned char *h_gradient_matrix, float *h_angle_matrix, dim3 matrix_dim, int canny_min, int canny_max);

  namespace Canny {
    __global__ void transfer_edges_between_blocks_kernel(char *ht_matrix, dim3 matrix_dim, int *done);
  }
}


namespace ProcessingUnitHost {
  void canny(unsigned char *gradient_matrix, float *angle_matrix, dim3 matrix_dim, int canny_min, int canny_max);
}

#endif // CANNY_HPP
