#ifndef CUTOUT_KERNEL_H
#define CUTOUT_KERNEL_H

__global__ void draw_edges_on_cutout_matrix_kernel(unsigned char *edge_matrix,
                                                   char *micro_cutout_matrix, dim3 matrix_dim,
                                                   int2 cutout_start_pixel, int2 tracking_start_pixel,
                                                   int threshold, char *macro_cutout_matrix);

__global__ void cutout_algorithm_kernel(char *cutout_matrix, dim3 matrix_dim, int *done,
                                        char *looking_pixels, char spread_char = 'M',
                                        int2 *tracking_top_left = nullptr, int2 *tracking_bottom_right = nullptr);


__global__ void apply_macro_to_micro_cutout_matrix_kernel(char *macro_cutout_matrix, char *micro_cutout_matrix, dim3 macro_matrix_dim, dim3 micro_matrix_dim);

__global__ void apply_cutout_kernel(char *micro_cutout_matrix, unsigned char *output_image, dim3 image_dim, int2 start_pixel, bool targeting);


namespace ProcessingUnitDevice {
  namespace Cutout {
    __global__ void transfer_edges_between_blocks_kernel(char *cutout_matrix, dim3 matrix_dim,
                                                         int *done, char *looking_pixels, char spread_char = 'M');
  }
}


#endif // CUTOUT_KERNEL_H
