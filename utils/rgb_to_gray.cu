#include <iostream>

#include "rgb_to_gray.hpp"
#include "../main.hpp"

__global__ void rgb_to_gray_kernel(unsigned char *rgb_image, unsigned char *gray_image, dim3 image_dim) {
  int2 global_index = make_int2(threadIdx.x + (blockIdx.x * blockDim.x), threadIdx.y + (blockIdx.y * blockDim.y));

  if (image_dim.x <= global_index.x || image_dim.y <= global_index.y) {
    return;
  }

  gray_image[global_index.y*image_dim.x + global_index.x] = rgb_to_gray_core(global_index, rgb_image, image_dim);
}

__device__ __host__ unsigned char rgb_to_gray_core(int2 index, unsigned char *rgb_image, dim3 image_dim) {
  unsigned char r = 0, g = 0, b = 0;

  r = rgb_image[3 * (index.y*image_dim.x + index.x)];
  g = rgb_image[3 * (index.y*image_dim.x + index.x) + 1];
  b = rgb_image[3 * (index.y*image_dim.x + index.x) + 2];

  return (0.21 * r + 0.71 * g + 0.07 * b);
}

void ProcessingUnitDevice::rgb_to_gray(RGBImage *h_rgb_image, GrayImage *h_gray_image) {
  dim3 image_dim(h_rgb_image->width, h_rgb_image->height);

  unsigned char *d_rgb_image;
  unsigned char *d_gray_image;

  cudaMalloc(&d_rgb_image, sizeof(unsigned char) * (3 * image_dim.x * image_dim.y));
  cudaMalloc(&d_gray_image, sizeof(unsigned char) * (image_dim.x * image_dim.y));

  cudaMemcpy(d_rgb_image, h_rgb_image->data, 3 * image_dim.x * image_dim.y, cudaMemcpyHostToDevice);

  dim3 threads = dim3(MATRIX_SIZE_PER_BLOCK, MATRIX_SIZE_PER_BLOCK);
  dim3 blocks = dim3(ceil((float) image_dim.x/MATRIX_SIZE_PER_BLOCK), ceil((float) image_dim.y/MATRIX_SIZE_PER_BLOCK));

  rgb_to_gray_kernel<<<blocks, threads>>>(d_rgb_image, d_gray_image, image_dim);

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
