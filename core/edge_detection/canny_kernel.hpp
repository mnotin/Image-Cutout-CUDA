#ifndef CANNY_KERNEL_HPP
#define CANNY_KERNEL_HPP


__global__ void non_maximum_suppression_kernel(unsigned char *gradient_matrix, float *angle_matrix, dim3 matrix_dim);

__global__ void histeresis_thresholding_init_kernel(unsigned char *gradient_matrix, char *ht_matrix, dim3 matrix_dim, int canny_min, int canny_max);

__global__ void histeresis_thresholding_loop_kernel(char *ht_matrix, dim3 matrix_dim, int *done);

__global__ void histeresis_thresholding_end_kernel(unsigned char *gradient_matrix, char *ht_matrix, dim3 matrix_dim);


namespace ProcessingUnitDevice {
  namespace Canny {
    __global__ void transfer_edges_between_blocks_kernel(char *ht_matrix, dim3 matrix_dim, int *done);
  }
}

#endif // CANNY_KERNEL_HPP
