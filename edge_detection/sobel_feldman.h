#ifndef SOBEL_FELDMAN_H
#define SOBEL_FELDMAN_H

void sobel_feldman(unsigned char *h_input_matrix, unsigned char *h_gradient_matrix, float *h_angle_matrix, int matrix_width, int matrix_height);
__global__ void global_gradient(unsigned char *matrix, unsigned char *horizontal_edges, unsigned char *vertical_edges, int matrix_width, int matrix_height);
__global__ void angle(unsigned char *horizontal_edges, unsigned char *vertical_edges, float *angle_matrix, int matrix_width, int matrix_height);

#endif // SOBEL_FELDMAN_H
