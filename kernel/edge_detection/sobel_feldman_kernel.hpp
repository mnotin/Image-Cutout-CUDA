#ifndef SOBEL_FELDMAN_KERNEL_HPP
#define SOBEL_FELDMAN_KERNEL_HPP

__global__ void global_gradient_kernel(unsigned char *output_matrix, int *horizontal_edges, int *vertical_edges, dim3 matrix_dim);
__global__ void angle_kernel(float *angle_matrix, int *horizontal_gradient, int *vertical_gradient, dim3 matrix_dim);

__global__ void edge_color_kernel(float *angle_matrix, unsigned char *output_image, dim3 image_dim);

#endif // SOBEL_FELDMAN_KERNEL_HPP
