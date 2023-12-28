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
  dim3 threads = dim3(MATRIX_SIZE_PER_BLOCK, MATRIX_SIZE_PER_BLOCK);
  dim3 blocks = dim3(matrix_width/MATRIX_SIZE_PER_BLOCK, matrix_height/MATRIX_SIZE_PER_BLOCK);

  unsigned char *d_input_matrix;
  unsigned char *d_gradient_matrix;
  int *d_horizontal_gradient;
  int *d_vertical_gradient;
  float *d_angle_matrix;
  float *d_kernel;

  cudaMalloc((void **) &d_input_matrix, matrix_width * matrix_height * sizeof(unsigned char));
  cudaMalloc((void **) &d_gradient_matrix, matrix_width * matrix_height * sizeof(unsigned char));
  cudaMalloc((void **) &d_horizontal_gradient, matrix_width * matrix_height * sizeof(int));
  cudaMalloc((void **) &d_vertical_gradient, matrix_width * matrix_height * sizeof(int));
  cudaMalloc((void **) &d_angle_matrix, matrix_width * matrix_height * sizeof(float));
  cudaMalloc((void **) &d_kernel, KERNEL_SIZE*KERNEL_SIZE * sizeof(float));

  cudaMemcpy(d_input_matrix, h_input_matrix, matrix_width * matrix_height * sizeof(unsigned char), cudaMemcpyHostToDevice);

  // Horizontal gradient
  cudaMemcpy(d_kernel, sobel_kernel_horizontal_kernel, KERNEL_SIZE*KERNEL_SIZE * sizeof(int), cudaMemcpyHostToDevice);
  printf("Nombre de blocs lancés: %d %d\n", blocks.x, blocks.y);
  convolution<<<blocks, threads>>>(d_input_matrix, d_horizontal_gradient, matrix_width, matrix_height, d_kernel, 3);
  cudaDeviceSynchronize();

  // Vertical gradient
  cudaMemcpy(d_kernel, sobel_kernel_vertical_kernel, KERNEL_SIZE*KERNEL_SIZE * sizeof(int), cudaMemcpyHostToDevice);
  convolution<<<blocks, threads>>>(d_input_matrix, d_vertical_gradient, matrix_width, matrix_height, d_kernel, 3);
  cudaDeviceSynchronize();
  
  // Global gradient
  global_gradient<<<blocks, threads>>>(d_gradient_matrix, d_horizontal_gradient, d_vertical_gradient, matrix_width, matrix_height); 
  cudaMemcpy(h_gradient_matrix, d_gradient_matrix, matrix_width * matrix_height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
 
  // Angle of the gradient
  angle<<<blocks, threads>>>(d_horizontal_gradient, d_vertical_gradient, d_angle_matrix, matrix_width, matrix_height);
  cudaMemcpy(h_angle_matrix, d_angle_matrix, matrix_width * matrix_height * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

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
__global__ void global_gradient(unsigned char *output_matrix, int *horizontal_edges, int *vertical_edges, int matrix_width, int matrix_height) {
  int globalIdxX = threadIdx.x + (blockIdx.x * blockDim.x);
  int globalIdxY = threadIdx.y + (blockIdx.y * blockDim.y);
  const int GLOBAL_IDX = globalIdxY * matrix_width + globalIdxX;

  int g_x = horizontal_edges[GLOBAL_IDX];
  int g_y = vertical_edges[GLOBAL_IDX];
  float global_gradient = sqrt((double) g_x * g_x + g_y * g_y);

  output_matrix[GLOBAL_IDX] = global_gradient <= 255.0 ? (unsigned char) global_gradient : 255;
}

__global__ void angle(int *horizontal_gradient, int *vertical_gradient, float *angle_matrix, int matrix_width, int matrix_height) {
  int globalIdxX = threadIdx.x + (blockIdx.x * blockDim.x);
  int globalIdxY = threadIdx.y + (blockIdx.y * blockDim.y);
  const int GLOBAL_IDX = globalIdxY * matrix_width + globalIdxX;

  int g_x = horizontal_gradient[GLOBAL_IDX];
  int g_y = vertical_gradient[GLOBAL_IDX];
  float angle = atan((float) g_y / g_x);

  angle_matrix[GLOBAL_IDX] = angle; 
}

void generate_edge_color(unsigned char *h_gradient_matrix, float *h_angle_matrix, unsigned char *h_output_image, int matrix_width, int matrix_height) {
  unsigned char *d_gradient_matrix;
  float *d_angle_matrix;
  unsigned char *d_output_image;

  cudaMalloc((void **) &d_gradient_matrix, matrix_width * matrix_height * sizeof(unsigned char));
  cudaMalloc((void **) &d_angle_matrix, matrix_width * matrix_height * sizeof(float));
  cudaMalloc((void **) &d_output_image, 3 * matrix_width * matrix_height * sizeof(unsigned char));

  cudaMemcpy(d_gradient_matrix, h_gradient_matrix, matrix_width * matrix_height * sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_angle_matrix, h_angle_matrix, matrix_width * matrix_height * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_output_image, h_output_image, 3 * matrix_width * matrix_height * sizeof(unsigned char), cudaMemcpyHostToDevice);

  dim3 threads = dim3(MATRIX_SIZE_PER_BLOCK, MATRIX_SIZE_PER_BLOCK);
  dim3 blocks = dim3(matrix_width/MATRIX_SIZE_PER_BLOCK, matrix_height/MATRIX_SIZE_PER_BLOCK);
  printf("Nombre de blocs lancés: %d %d\n", blocks.x, blocks.y);
  edge_color<<<blocks, threads>>>(d_gradient_matrix, d_angle_matrix, d_output_image, matrix_width, matrix_height);

  cudaMemcpy(h_output_image, d_output_image, 3 * matrix_width * matrix_height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

  cudaFree(d_gradient_matrix);
  cudaFree(d_angle_matrix);
  cudaFree(d_output_image);
}

/**
 * Give a color to edges depending on their direction.
 **/
__global__ void edge_color(unsigned char *gradient_matrix, float *angle_matrix, unsigned char *output_image, int image_width, int image_height) { 
  int globalIdxX = threadIdx.x + (blockIdx.x * blockDim.x);
  int globalIdxY = threadIdx.y + (blockIdx.y * blockDim.y);
  const int GLOBAL_IDX = globalIdxY * image_width + globalIdxX;

  const float ANGLE = angle_matrix[GLOBAL_IDX] + M_PI_2;
  
  if (50 < gradient_matrix[GLOBAL_IDX]) {
    if (ANGLE < M_PI / 8.0 || (M_PI / 8.0) * 7 < ANGLE) {
      // Horizontal gradient direction : Yellow
      output_image[3 * (GLOBAL_IDX)] = 255;
      output_image[3 * (GLOBAL_IDX) + 1] = 255; 
      output_image[3 * (GLOBAL_IDX) + 2] = 0; 
    } else if (M_PI / 8.0 < ANGLE && ANGLE < (M_PI / 8.0) * 3) {
      // Top right gradient direction : Green
      output_image[3 * (GLOBAL_IDX)] = 0; 
      output_image[3 * (GLOBAL_IDX) + 1] = 255; 
      output_image[3 * (GLOBAL_IDX) + 2] = 0; 
    } else if ((M_PI / 8.0) * 5 < ANGLE && ANGLE < (M_PI / 8.0) * 7) {
      // Top left gradient direction : Red
      output_image[3 * (GLOBAL_IDX)] = 255; 
      output_image[3 * (GLOBAL_IDX) + 1] = 0; 
      output_image[3 * (GLOBAL_IDX) + 2] = 0; 
    } else {
      // Vertical gradient direction : Blue
      output_image[3 * (GLOBAL_IDX)] = 0; 
      output_image[3 * (GLOBAL_IDX) + 1] = 0; 
      output_image[3 * (GLOBAL_IDX) + 2] = 255; 
    }
  } else {
    output_image[3 * (GLOBAL_IDX)] = 0; 
    output_image[3 * (GLOBAL_IDX) + 1] = 0; 
    output_image[3 * (GLOBAL_IDX) + 2] = 0; 
  }
}
