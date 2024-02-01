#include "rgb_to_gray_kernel.hpp"

#include "../../core/utils/rgb_to_gray_core.hpp"

__global__ void rgb_to_gray_kernel(unsigned char *rgb_image, unsigned char *gray_image, dim3 image_dim) {
  int2 global_index = make_int2(threadIdx.x + (blockIdx.x * blockDim.x), threadIdx.y + (blockIdx.y * blockDim.y));

  if (image_dim.x <= global_index.x || image_dim.y <= global_index.y) {
    return;
  }

  gray_image[global_index.y*image_dim.x + global_index.x] = rgb_to_gray_core(global_index, rgb_image, image_dim);
}
