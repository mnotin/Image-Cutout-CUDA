#include <iostream>

#include "rgb_to_gray.hpp"
#include "../main.hpp"

__global__ void rgb_to_gray_kernel(unsigned char *rgb_image, unsigned char *gray_image, Dim image_dim) {
  Vec2 index;
  index.x = threadIdx.x + blockIdx.x * blockDim.x;
  index.y = threadIdx.y + blockIdx.y * blockDim.y;


  gray_image[index.y*image_dim.width + index.x] = rgb_to_gray_core(index, rgb_image, image_dim);
}

__device__ __host__ unsigned char rgb_to_gray_core(Vec2 index, unsigned char *rgb_image, Dim image_dim) {
  unsigned char r = 0, g = 0, b = 0;

  if (index.y*image_dim.width+index.x < image_dim.width * image_dim.height) {
    r = rgb_image[3 * (index.y*image_dim.width + index.x)];
    g = rgb_image[3 * (index.y*image_dim.width + index.x) + 1];
    b = rgb_image[3 * (index.y*image_dim.width + index.x) + 2];
  }

  return (0.21 * r + 0.71 * g + 0.07 * b);
}

void ProcessingUnitDevice::rgb_to_gray(RGBImage *h_rgb_image, GrayImage *h_gray_image) {
  Dim rgb_image_dim;
  rgb_image_dim.width = h_rgb_image->width;
  rgb_image_dim.height = h_rgb_image->height;

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
  rgb_to_gray_kernel<<<blocks, threads>>>(d_rgb_image, d_gray_image, rgb_image_dim);

  // Copy result from device to host
  cudaMemcpy(h_gray_image->data, d_gray_image, h_gray_image->width * h_gray_image->height, cudaMemcpyDeviceToHost);

  cudaFree(d_rgb_image);
  cudaFree(d_gray_image);
}

void ProcessingUnitHost::rgb_to_gray(RGBImage *rgb_image, GrayImage *gray_image) {
  Dim gray_image_dim;
  gray_image_dim.width = gray_image->width;
  gray_image_dim.height = gray_image->height;

  for (int i = 0; i < gray_image->height; i++) {
    for (int j = 0; j < gray_image->width; j++) {
      Vec2 index;
      index.x = j;
      index.y = i;

      gray_image->data[i*gray_image->width + j] = rgb_to_gray_core(index, rgb_image->data, gray_image_dim);
    }
  }
}
