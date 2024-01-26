#include "rgb_to_gray_launcher.hpp"

#include "../../kernel/utils/rgb_to_gray_kernel.hpp"
#include "../../core/utils/rgb_to_gray_core.hpp"
#include "../../main.hpp"

void ProcessingUnitDevice::rgb_to_gray(RGBImage *h_rgb_image, GrayImage *h_gray_image) {
  dim3 image_dim(h_rgb_image->width, h_rgb_image->height);
  dim3 block_dim(MATRIX_SIZE_PER_BLOCK, MATRIX_SIZE_PER_BLOCK);
  dim3 grid_dim(ceil((float) image_dim.x/MATRIX_SIZE_PER_BLOCK), ceil((float) image_dim.y/MATRIX_SIZE_PER_BLOCK));

  unsigned char *d_rgb_image;
  unsigned char *d_gray_image;

  cudaMalloc(&d_rgb_image, sizeof(unsigned char) * (3 * image_dim.x * image_dim.y));
  cudaMalloc(&d_gray_image, sizeof(unsigned char) * (image_dim.x * image_dim.y));

  cudaMemcpy(d_rgb_image, h_rgb_image->data, 3 * image_dim.x * image_dim.y, cudaMemcpyHostToDevice);

  rgb_to_gray_kernel<<<grid_dim, block_dim>>>(d_rgb_image, d_gray_image, image_dim);

  cudaMemcpy(h_gray_image->data, d_gray_image, image_dim.x * image_dim.y, cudaMemcpyDeviceToHost);

  cudaFree(d_rgb_image);
  cudaFree(d_gray_image);
}

void ProcessingUnitHost::rgb_to_gray(RGBImage *rgb_image, GrayImage *gray_image) {
  dim3 image_dim(rgb_image->width, rgb_image->height);
  int2 index;

  for (index.y = 0; index.y < gray_image->height; index.y++) {
    for (index.x = 0; index.x < gray_image->width; index.x++) {
      gray_image->data[index.y*image_dim.x + index.x] = rgb_to_gray_core(index, rgb_image->data, image_dim);
    }
  }
}
