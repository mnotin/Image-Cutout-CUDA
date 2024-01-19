
#ifndef RGB_TO_GRAY_HPP
#define RGB_TO_GRAY_HPP

#include "../main.hpp"
#include "../img.h"

__global__ void rgb_to_gray_kernel(unsigned char *rgb_image, unsigned char *gray_image, Dim image_dim);
__device__ __host__ unsigned char rgb_to_gray_core(Vec2 index, unsigned char *rgb_image, Dim image_dim);


namespace ProcessingUnitDevice {
  void rgb_to_gray(RGBImage *h_rgb_image, GrayImage *h_gray_image);
}


namespace ProcessingUnitHost {
  void rgb_to_gray(RGBImage *rgb_image, GrayImage *gray_image);
}

#endif // RGB_TO_GRAY_HPP
