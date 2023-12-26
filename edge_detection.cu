#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#include "main.h"
#include "edge_detection.h"

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

void rgb_to_gray(RGBImage *h_rgb_image, GrayImage *h_gray_image)  {
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
}

/**
 * Applies a gaussian blur over a matrix.
 **/
void gaussian_blur(unsigned char *h_matrix, int matrix_width, int matrix_height) {
  const int KERNEL_WIDTH = 3;
  float gaussian_blur_kernel[KERNEL_WIDTH*KERNEL_WIDTH] = {1/16.0, 2/16.0, 1/16.0, 
                                                         2/16.0, 4/16.0, 2/16.0, 
                                                         1/16.0, 2/16.0, 1/16.0};
 
  unsigned char *d_input_matrix;
  unsigned char *d_output_matrix;
  float *d_kernel;
  cudaMalloc((void **) &d_input_matrix, matrix_width * matrix_height * sizeof(unsigned char));
  cudaMalloc((void **) &d_output_matrix, matrix_width * matrix_height * sizeof(unsigned char));
  cudaMalloc((void **) &d_kernel, KERNEL_WIDTH*KERNEL_WIDTH * sizeof(float));

  cudaMemcpy(d_input_matrix, h_matrix, matrix_width*matrix_height*sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel, gaussian_blur_kernel, KERNEL_WIDTH*KERNEL_WIDTH * sizeof(int), cudaMemcpyHostToDevice);

  dim3 threads = dim3(MATRIX_SIZE_PER_BLOCK, MATRIX_SIZE_PER_BLOCK);
  dim3 blocks = dim3(matrix_width/MATRIX_SIZE_PER_BLOCK, matrix_height/MATRIX_SIZE_PER_BLOCK);
  printf("Nombre de blocs lancés: %d %d\n", blocks.x, blocks.y);
  convolution<<<blocks, threads>>>(d_input_matrix, d_output_matrix, matrix_width, matrix_height, d_kernel, KERNEL_WIDTH);
 
  cudaMemcpy(h_matrix, d_output_matrix, matrix_width*matrix_height*sizeof(unsigned char), cudaMemcpyDeviceToHost);

  cudaFree(d_input_matrix);
  cudaFree(d_output_matrix);
  cudaFree(d_kernel);
}

/**
 * Applies the Sobel-Feldman operator over a matrix.
 * The picture should have been smoothed and converted to grayscale prior to being passed over the Sobel-Feldman operator. 
 **/
void sobel_feldman(unsigned char *h_matrix, int matrix_width, int matrix_height) {
  const int KERNEL_SIZE = 3;
  float sobel_kernel_horizontal_kernel[KERNEL_SIZE*KERNEL_SIZE] = {1, 0,  -1, 
                                                                   2, 0,  -2, 
                                                                   1, 0, -1};
  float sobel_kernel_vertical_kernel[KERNEL_SIZE*KERNEL_SIZE] = { 1,  2,  1,
                                                                  0,  0,  0,
                                                                 -1, -2, -1};
 
  unsigned char *d_input_matrix;
  unsigned char *d_output_matrix;
  unsigned char *d_horizontal_edges;
  unsigned char *d_vertical_edges;
  float *d_kernel;
  cudaMalloc((void **) &d_input_matrix, matrix_width * matrix_height * sizeof(unsigned char));
  cudaMalloc((void **) &d_output_matrix, matrix_width * matrix_height * sizeof(unsigned char));
  cudaMalloc((void **) &d_horizontal_edges, matrix_width * matrix_height * sizeof(unsigned char));
  cudaMalloc((void **) &d_vertical_edges, matrix_width * matrix_height * sizeof(unsigned char));
  cudaMalloc((void **) &d_kernel, KERNEL_SIZE*KERNEL_SIZE * sizeof(float));

  cudaMemcpy(d_input_matrix, h_matrix, matrix_width*matrix_height*sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_horizontal_edges, h_matrix, matrix_width*matrix_height*sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vertical_edges, h_matrix, matrix_width*matrix_height*sizeof(unsigned char), cudaMemcpyHostToDevice);

  dim3 threads = dim3(MATRIX_SIZE_PER_BLOCK, MATRIX_SIZE_PER_BLOCK);
  dim3 blocks = dim3(matrix_width/MATRIX_SIZE_PER_BLOCK, matrix_height/MATRIX_SIZE_PER_BLOCK);
  cudaMemcpy(d_kernel, sobel_kernel_horizontal_kernel, KERNEL_SIZE*KERNEL_SIZE * sizeof(int), cudaMemcpyHostToDevice);
  printf("Nombre de blocs lancés: %d %d\n", blocks.x, blocks.y);
  convolution<<<blocks, threads>>>(d_input_matrix, d_horizontal_edges, matrix_width, matrix_height, d_kernel, 3);
  cudaMemcpy(d_kernel, sobel_kernel_vertical_kernel, KERNEL_SIZE*KERNEL_SIZE * sizeof(int), cudaMemcpyHostToDevice);
  convolution<<<blocks, threads>>>(d_input_matrix, d_vertical_edges, matrix_width, matrix_height, d_kernel, 3);
  cudaDeviceSynchronize();
  global_gradient<<<blocks, threads>>>(d_output_matrix, d_horizontal_edges, d_vertical_edges, matrix_width, matrix_height);
 
  cudaMemcpy(h_matrix, d_output_matrix, matrix_width*matrix_height*sizeof(unsigned char), cudaMemcpyDeviceToHost);

  cudaFree(d_input_matrix);
  cudaFree(d_output_matrix);
  cudaFree(d_horizontal_edges);
  cudaFree(d_vertical_edges);
  cudaFree(d_kernel);
}

/**
 * Computes the global gradient of an image after being processed by the Sobel-Feldman operator.
 **/
__global__ void global_gradient(unsigned char *matrix, unsigned char *horizontal_edges, unsigned char *vertical_edges, int matrix_width, int matrix_height) {
  int globalIdxX = threadIdx.x + (blockIdx.x * blockDim.x);
  int globalIdxY = threadIdx.y + (blockIdx.y * blockDim.y);

  unsigned char g_x = horizontal_edges[globalIdxY*matrix_width + globalIdxX];
  unsigned char g_y = vertical_edges[globalIdxY*matrix_width + globalIdxX];

  matrix[globalIdxY*matrix_width + globalIdxX] = sqrt((double) g_x * g_x + g_y * g_y);
}

void cutout(unsigned char *h_matrix, int matrix_width, int matrix_height, int start_pixel_x, int start_pixel_y) {
  unsigned char **h_cutout_matrix;
  h_cutout_matrix = (unsigned char **) malloc(matrix_height * sizeof(unsigned char*));
  for (int i = 0; i < matrix_height; i++) {
    h_cutout_matrix[i] = (unsigned char *) malloc(matrix_width * sizeof(unsigned char));
  }
  
  for (int i = 0; i < matrix_height; i++) {
    for (int j = 0; j < matrix_height; j++) {
      h_cutout_matrix[i][j] = 'D';
    }
  }

  unsigned char *d_input_matrix;
  unsigned char *d_cutout_matrix;

  cudaMalloc((void **) &d_input_matrix, matrix_width * matrix_height * sizeof(unsigned char));
  cudaMalloc((void **) &d_cutout_matrix, matrix_width * matrix_height * sizeof(unsigned char));

  cudaMemcpy(d_input_matrix, h_matrix, matrix_width*matrix_height*sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_cutout_matrix, h_cutout_matrix, matrix_width*matrix_height*sizeof(unsigned char), cudaMemcpyHostToDevice);

  cudaFree(d_input_matrix);
  cudaFree(d_cutout_matrix);
}

/**
 *  
 **/
__global__ void cutout_algorithm(int start_x, int start_y, int done, unsigned char *input_matrix, unsigned char *cutout_matrix, int matrix_width, int matrix_height) {
  int globalIdxX = threadIdx.x + (blockIdx.x * blockDim.x);
  int globalIdxY = threadIdx.y + (blockIdx.y * blockDim.y);

  __shared__ int shared_done;

  if (globalIdxX == 0 && globalIdxY == 0) {
    shared_done = 1;
  }

  // Process
  // if () {
  //   shared_done = 0;
  // }

  if (globalIdxX == 0 && globalIdxY == 0 && shared_done == 0 && done == 1) {
    done = 0; 
  }
}
