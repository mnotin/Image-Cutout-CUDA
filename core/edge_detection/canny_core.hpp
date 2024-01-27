#ifndef CANNY_CORE_HPP
#define CANNY_CORE_HPP

__device__ __host__ unsigned char non_maximum_suppression_core(int2 index, unsigned char *gradient_matrix, float *angle_matrix, dim3 matrix_dim);

__device__ __host__ char histeresis_thresholding_init_core(int2 index, unsigned char *gradient_matrix, dim3 matrix_dim, int canny_min, int canny_max);

__device__ __host__ void histeresis_thresholding_loop_core(int2 index, char *ht_matrix, dim3 matrix_dim, int2 read_limit, int *done);

__device__ __host__ unsigned char histeresis_thresholding_end_core(int2 index, char *ht_matrix, dim3 matrix_dim);

#endif // CANNY_CORE_HPP
