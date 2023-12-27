#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#include "sobel_feldman.h"
#include "../main.h"
#include "../utils.h"

/**
 * Applies the Sobel-Feldman operator over a matrix.
 * The picture should have been smoothed and converted to grayscale prior to being passed over the Sobel-Feldman operator. 
 **/
void sobel_feldman(unsigned char *h_input_matrix, unsigned char *h_gradient_matrix, float *h_angle_matrix, int matrix_width, int matrix_height) {
  const int KERNEL_SIZE = 3;
  float sobel_kernel_horizontal_kernel[KERNEL_SIZE*KERNEL_SIZE] = {1, 0,  -1, 
                                                                   2, 0,  -2, 
                                                                   1, 0, -1};
  float sobel_kernel_vertical_kernel[KERNEL_SIZE*KERNEL_SIZE] = { 1,  2,  1,
                                                                  0,  0,  0,
                                                                 -1, -2, -1}; 
  unsigned char *d_input_matrix;
  unsigned char *d_gradient_matrix;
  unsigned char *d_horizontal_gradient;
  unsigned char *d_vertical_gradient;
  float *d_angle_matrix;
  float *d_kernel;
  cudaMalloc((void **) &d_input_matrix, matrix_width * matrix_height * sizeof(unsigned char));
  cudaMalloc((void **) &d_gradient_matrix, matrix_width * matrix_height * sizeof(unsigned char));
  cudaMalloc((void **) &d_horizontal_gradient, matrix_width * matrix_height * sizeof(unsigned char));
  cudaMalloc((void **) &d_vertical_gradient, matrix_width * matrix_height * sizeof(unsigned char));
  cudaMalloc((void **) &d_angle_matrix, matrix_width * matrix_height * sizeof(float));
  cudaMalloc((void **) &d_kernel, KERNEL_SIZE*KERNEL_SIZE * sizeof(float));

  cudaMemcpy(d_input_matrix, h_input_matrix, matrix_width * matrix_height * sizeof(unsigned char), cudaMemcpyHostToDevice);
  dim3 threads = dim3(MATRIX_SIZE_PER_BLOCK, MATRIX_SIZE_PER_BLOCK);
  dim3 blocks = dim3(matrix_width/MATRIX_SIZE_PER_BLOCK, matrix_height/MATRIX_SIZE_PER_BLOCK);
  cudaMemcpy(d_kernel, sobel_kernel_horizontal_kernel, KERNEL_SIZE*KERNEL_SIZE * sizeof(int), cudaMemcpyHostToDevice);
  printf("Nombre de blocs lancés: %d %d\n", blocks.x, blocks.y);
  convolution<<<blocks, threads>>>(d_input_matrix, d_horizontal_gradient, matrix_width, matrix_height, d_kernel, 3);
  cudaMemcpy(d_kernel, sobel_kernel_vertical_kernel, KERNEL_SIZE*KERNEL_SIZE * sizeof(int), cudaMemcpyHostToDevice);
  convolution<<<blocks, threads>>>(d_input_matrix, d_vertical_gradient, matrix_width, matrix_height, d_kernel, 3);
  cudaDeviceSynchronize();
  global_gradient<<<blocks, threads>>>(d_gradient_matrix, d_horizontal_gradient, d_vertical_gradient, matrix_width, matrix_height); 
  cudaMemcpy(h_gradient_matrix, d_gradient_matrix, matrix_width * matrix_height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

  angle<<<blocks, threads>>>(d_horizontal_gradient, d_vertical_gradient, d_angle_matrix, matrix_width, matrix_height);
  cudaMemcpy(h_angle_matrix, d_angle_matrix, matrix_width * matrix_height * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_input_matrix);
  cudaFree(d_gradient_matrix);
  cudaFree(d_horizontal_gradient);
  cudaFree(d_vertical_gradient);
  cudaFree(d_angle_matrix);
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

__global__ void angle(unsigned char *horizontal_edges, unsigned char *vertical_edges, float *angle_matrix, int matrix_width, int matrix_height) {
  int globalIdxX = threadIdx.x + (blockIdx.x * blockDim.x);
  int globalIdxY = threadIdx.y + (blockIdx.y * blockDim.y);

  int g_x = horizontal_edges[globalIdxY*matrix_width + globalIdxX] - '0';
  int g_y = vertical_edges[globalIdxY*matrix_width + globalIdxX] - '0';

  angle_matrix[globalIdxY*matrix_width + globalIdxX] = atan((float) g_y / g_x);
}
