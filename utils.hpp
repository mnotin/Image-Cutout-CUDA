#ifndef UTILS_HPP
#define UTILS_HPP

#include "main.hpp"
#include "img.h"

__global__ void convolution(unsigned char *input_matrix, int *output_matrix, int matrix_width, int matrix_height, float *kernel, int kernel_size);

__global__ void rgb_to_gray_kernel(unsigned char *rgb_image, unsigned char *gray_image, int image_width, int image_height);
__device__ __host__ void rgb_to_gray_core(Vec2 index, unsigned char *rgb_image, unsigned char *gray_image, int image_width, int image_height);

void gaussian_blur(unsigned char *h_matrix, int matrix_width, int matrix_height);


namespace ProcessingUnitDevice {
  void rgb_to_gray(RGBImage *h_rgb_image, GrayImage *h_gray_image);
}


namespace ProcessingUnitHost {
  void rgb_to_gray(RGBImage *rgb_image, GrayImage *gray_image);
}

#endif // UTILS_HPP
