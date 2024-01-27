#ifndef SOBEL_FELDMAN_CORE_HPP
#define SOBEL_FELDMAN_CORE_HPP

__device__ __host__ unsigned char global_gradient_core(int2 index, int *horizontal_edges, int *vertical_edges, dim3 matrix_dim);
__device__ __host__ float angle_core(int2 index, int *horizontal_gradient, int *vertical_gradient, dim3 matrix_dim);

__device__ __host__ void edge_color_core(int2 index, unsigned char *gradient_matrix, float *angle_matrix, unsigned char *output_image, dim3 image_dim);

#endif // SOBEL_FELDMAN_CORE_HPP
