
#include <iostream>

#include "rgb_to_gray.hpp"
#include "../main.hpp"

__global__ void rgb_to_gray_kernel(unsigned char *rgb_image, unsigned char *gray_image, int image_width, int image_height) {
  unsigned int localIdxX = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int localIdxY = threadIdx.y + blockIdx.y * blockDim.y;

  Vec2 index;
  index.x = localIdxX;
  index.y = localIdxY;

  rgb_to_gray_core(index, rgb_image, gray_image, image_width, image_height);
}

__device__ __host__ void rgb_to_gray_core(Vec2 index, unsigned char *rgb_image, unsigned char *gray_image, int image_width, int image_height) {
  unsigned char r, g, b;

  if (index.y*image_width+index.x < image_width * image_height) {
    r = rgb_image[3 * (index.y*image_width + index.x)];
    g = rgb_image[3 * (index.y*image_width + index.x) + 1];
    b = rgb_image[3 * (index.y*image_width + index.x) + 2];

    gray_image[index.y*image_width + index.x] = (0.21 * r + 0.71 * g + 0.07 * b);
  }
}

void ProcessingUnitDevice::rgb_to_gray(RGBImage *h_rgb_image, GrayImage *h_gray_image) {
  // Allocating device memory
  unsigned char *d_rgb_image;
  unsigned char *d_gray_image;

  cudaMalloc((void **) &d_rgb_image, sizeof(unsigned char) * (3 * h_rgb_image->width * h_rgb_image->height));
  cudaMalloc((void **) &d_gray_image, sizeof(unsigned char) * (h_gray_image->width * h_gray_image->height)); 

  // Copying host memory to device
  cudaMemcpy(d_rgb_image, h_rgb_image->data, 3 * h_rgb_image->width * h_rgb_image->height, cudaMemcpyHostToDevice);

  // Initialize thread block and kernel grid dimensions
  dim3 threads = dim3(MATRIX_SIZE_PER_BLOCK, MATRIX_SIZE_PER_BLOCK);
  dim3 blocks = dim3(h_rgb_image->width/MATRIX_SIZE_PER_BLOCK, h_rgb_image->height/MATRIX_SIZE_PER_BLOCK);

  // Invoke CUDA kernel
  rgb_to_gray_kernel<<<blocks, threads>>>(d_rgb_image, d_gray_image, h_rgb_image->width, h_rgb_image->height);

  // Copy result from device to host
  cudaMemcpy(h_gray_image->data, d_gray_image, h_gray_image->width * h_gray_image->height, cudaMemcpyDeviceToHost);

  cudaFree(d_rgb_image);
  cudaFree(d_gray_image);
}

void ProcessingUnitHost::rgb_to_gray(RGBImage *rgb_image, GrayImage *gray_image) {
  for (int i = 0; i < gray_image->height; i++) {
    for (int j = 0; j < gray_image->width; j++) {
      Vec2 index;
      index.x = j;
      index.y = i;

      rgb_to_gray_core(index, rgb_image->data, gray_image->data, gray_image->width, gray_image->height);
    }
  }
}
