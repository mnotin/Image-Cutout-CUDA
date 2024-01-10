#ifndef CANNY_H
#define CANNY_H

void canny(unsigned char *h_gradient_matrix, float *h_angle_matrix, int matrix_width, int matrix_height, int canny_min, int canny_max);
__global__ void non_maximum_suppression(unsigned char *gradient_matrix, float *angle_matrix, int matrix_width, int matrix_height);
__global__ void histeresis_thresholding_init(unsigned char *gradient_matrix, unsigned char *ht_matrix, int matrix_width, int matrix_height, int canny_min, int canny_max);
__global__ void histeresis_thresholding_loop(unsigned char *ht_matrix, int matrix_width, int matrix_height, int *done);
__global__ void histeresis_thresholding_end(unsigned char *gradient_matrix, unsigned char *ht_matrix, int matrix_width, int matrix_height);
__device__ char get_color_canny(float angle);

#endif // CANNY_H
