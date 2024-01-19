#ifndef SOBEL_FELDMAN_HPP
#define SOBEL_FELDMAN_HPP

#include "../main.hpp"

void sobel_feldman(unsigned char *h_input_matrix, unsigned char *h_gradient_matrix, float *h_angle_matrix, Dim matrix_dim);
__global__ void global_gradient(unsigned char *output_matrix, int *horizontal_edges, int *vertical_edges, Dim matrix_dim);
__global__ void angle(int *horizontal_gradient, int *vertical_gradient, float *angle_matrix, Dim matrix_dim);
void generate_edge_color(unsigned char *h_gradient_matrix, float *h_angle_matrix, unsigned char *h_output_image, Dim matrix_dim);
__global__ void edge_color(unsigned char *gradient_matrix, float *angle_matrix, unsigned char *output_image, Dim image_dim);
__device__ char get_color_sobel(float angle);

#endif // SOBEL_FELDMAN_HPP
