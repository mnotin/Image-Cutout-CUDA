#ifndef CUTOUT_CORE_H
#define CUTOUT_CORE_H


__device__ __host__ char draw_edges_on_cutout_matrix_core(int2 index, unsigned char *edge_matrix,
                                                          dim3 matrix_dim, int2 start_pixel,
                                                          int2 tracking_start_pixel, int threshold);

__device__ __host__ char cutout_algorithm_core(int2 index, char *cutout_matrix, dim3 matrix_dim,
                                               int2 read_limit, int *done, char *looking_pixels,
                                               char spread_pixel = 'M');

__device__ __host__ void apply_cutout_core(int2 index, char *micro_cutout_matrix, unsigned char *output_image, dim3 image_dim, int2 start_pixel);


#endif // CUTOUT_CORE_H
