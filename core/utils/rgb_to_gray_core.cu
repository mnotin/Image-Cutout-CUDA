#include "rgb_to_gray_core.hpp"

/**
 * Convert an RGB pixel into a gray pixel.
 **/
__device__ __host__ unsigned char rgb_to_gray_core(int2 index, unsigned char *rgb_image, dim3 image_dim) {
  unsigned char r = 0, g = 0, b = 0;

  r = rgb_image[3 * (index.y*image_dim.x + index.x)];
  g = rgb_image[3 * (index.y*image_dim.x + index.x) + 1];
  b = rgb_image[3 * (index.y*image_dim.x + index.x) + 2];

  return (0.21 * r + 0.71 * g + 0.07 * b);
}
