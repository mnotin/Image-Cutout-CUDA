#ifndef CANNY_HPP
#define CANNY_HPP

#include "../main.hpp"

void canny(unsigned char *h_gradient_matrix, float *h_angle_matrix, Dim matrix_dim, int canny_min, int canny_max);
__global__ void non_maximum_suppression(unsigned char *gradient_matrix, float *angle_matrix, Dim matrix_dim);
__global__ void histeresis_thresholding_init(unsigned char *gradient_matrix, unsigned char *ht_matrix, Dim matrix_dim, int canny_min, int canny_max);
__global__ void histeresis_thresholding_loop(unsigned char *ht_matrix, Dim matrix_dim, int *done);
__global__ void histeresis_thresholding_end(unsigned char *gradient_matrix, unsigned char *ht_matrix, Dim matrix_dim);
__device__ char get_color_canny(float angle);

#endif // CANNY_HPP
