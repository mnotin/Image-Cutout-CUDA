#ifndef SOBEL_FELDMAN_H
#define SOBEL_FELDMAN_H

void sobel_feldman(unsigned char *h_input_matrix, unsigned char *h_gradient_matrix, float *h_angle_matrix, int matrix_width, int matrix_height);
__global__ void global_gradient(unsigned char *output_matrix, int *horizontal_edges, int *vertical_edges, int matrix_width, int matrix_height);
__global__ void angle(int *horizontal_gradient, int *vertical_gradient, float *angle_matrix, int matrix_width, int matrix_height);
void generate_edge_color(unsigned char *h_gradient_matrix, float *h_angle_matrix, unsigned char *h_output_image, int matrix_width, int matrix_height);
__global__ void edge_color(unsigned char *gradient_matrix, float *angle_matrix, unsigned char *output_image, int image_width, int image_height);

#endif // SOBEL_FELDMAN_H
