#include <stdio.h>
#include <cuda_runtime.h>

#include "utils.h"
#include "main.h"

__global__ void rgb_to_gray_kernel(unsigned char *rgb_image, unsigned char *gray_image, int image_width, int image_height) {
  unsigned int localIdxX = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int localIdxY = threadIdx.y + blockIdx.y * blockDim.y;

  unsigned char r, g, b;

  if (localIdxY*image_width+localIdxX < image_width * image_height) {
    r = rgb_image[3 * (localIdxY*image_width + localIdxX)];
    g = rgb_image[3 * (localIdxY*image_width + localIdxX) + 1];
    b = rgb_image[3 * (localIdxY*image_width + localIdxX) + 2];

    gray_image[localIdxY*image_width + localIdxX] = (0.21 * r + 0.71 * g + 0.07 * b);
  }
}

void rgb_to_gray(RGBImage *h_rgb_image, GrayImage *h_gray_image) {
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

/**
 * Applies a gaussian blur over a matrix.
 **/
void gaussian_blur(unsigned char *h_matrix, int matrix_width, int matrix_height) {
  const int KERNEL_WIDTH = 3;
  float gaussian_blur_kernel[KERNEL_WIDTH*KERNEL_WIDTH] = {1/16.0, 2/16.0, 1/16.0, 
                                                         2/16.0, 4/16.0, 2/16.0, 
                                                         1/16.0, 2/16.0, 1/16.0};
  int h_int_matrix[matrix_width*matrix_height];
 
  unsigned char *d_input_matrix;
  int *d_output_matrix;
  float *d_kernel;
  cudaMalloc((void **) &d_input_matrix, matrix_width * matrix_height * sizeof(unsigned char));
  cudaMalloc((void **) &d_output_matrix, matrix_width * matrix_height * sizeof(int));
  cudaMalloc((void **) &d_kernel, KERNEL_WIDTH*KERNEL_WIDTH * sizeof(float));

  cudaMemcpy(d_input_matrix, h_matrix, matrix_width*matrix_height*sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel, gaussian_blur_kernel, KERNEL_WIDTH*KERNEL_WIDTH * sizeof(int), cudaMemcpyHostToDevice);

  dim3 threads = dim3(MATRIX_SIZE_PER_BLOCK, MATRIX_SIZE_PER_BLOCK);
  dim3 blocks = dim3(matrix_width/MATRIX_SIZE_PER_BLOCK, matrix_height/MATRIX_SIZE_PER_BLOCK);
  printf("Nombre de blocs lancés: %d %d\n", blocks.x, blocks.y);
  convolution<<<blocks, threads>>>(d_input_matrix, d_output_matrix, matrix_width, matrix_height, d_kernel, KERNEL_WIDTH);
 
  cudaMemcpy(h_int_matrix, d_output_matrix, matrix_width*matrix_height*sizeof(int), cudaMemcpyDeviceToHost);

  for (int i = 0; i < matrix_height; i++) {
    for (int j = 0; j < matrix_width; j++) {
      h_matrix[i*matrix_width + j] = h_int_matrix[i*matrix_width + j];
    }
  }

  cudaFree(d_input_matrix);
  cudaFree(d_output_matrix);
  cudaFree(d_kernel);
}
