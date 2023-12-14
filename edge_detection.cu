#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#include "main.h"
#include "edge_detection.h"

/**
 * Applies the Sobel-Feldman operator over a matrix.
 * The picture should have been smoothed and converted to grayscale prior to being passed over the Sobel-Feldman operator. 
 **/
void sobel_feldman(unsigned char **h_matrix, int matrix_width, int matrix_height) {
  int sobel_kernel_horizontal[9] = {1, 0,  1, 
                                    2, 0  -2, 
                                    1, 0, -1};
  int sobel_kernel_vertical[9] = { 1,  2,  1,
                                   0,  0,  0,
                                  -1, -2, -1};

  unsigned char h_horizontal_edges[matrix_width][matrix_height];
  unsigned char h_vertical_edges[matrix_width][matrix_height];
  
  unsigned char *d_matrix;
  unsigned char *d_horizontal_edges;
  unsigned char *d_vertical_edges;
  cudaMalloc((void **) &d_horizontal_edges, matrix_width * matrix_height * sizeof(unsigned char));
  cudaMalloc((void **) &d_vertical_edges, matrix_width * matrix_height * sizeof(unsigned char));

  for (int i = 0; i < matrix_height; i++) {
    cudaMemcpy(d_horizontal_edges+(i*matrix_width), h_matrix[i], matrix_width*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vertical_edges+(i*matrix_width), h_matrix[i], matrix_width*sizeof(unsigned char), cudaMemcpyHostToDevice);
  }

  dim3 threads = dim3(MATRIX_SIZE_PER_BLOCK, MATRIX_SIZE_PER_BLOCK);
  dim3 blocks = dim3(matrix_width/MATRIX_SIZE_PER_BLOCK, matrix_height/MATRIX_SIZE_PER_BLOCK);
  convolution<<<blocks, threads>>>(d_horizontal_edges, matrix_width, matrix_height, sobel_kernel_horizontal, 3);
  convolution<<<blocks, threads>>>(d_vertical_edges, matrix_width, matrix_height, sobel_kernel_vertical, 3);
  // cudaDeviceSynchronize();
  global_gradient<<<blocks, threads>>>(d_matrix, d_horizontal_edges, d_vertical_edges, matrix_width, matrix_height);

  for (int i = 0; i < matrix_height; i++) {
    cudaMemcpy(h_horizontal_edges[i], d_horizontal_edges+(i*matrix_width), matrix_width*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vertical_edges[i], d_vertical_edges+(i*matrix_width), matrix_width*sizeof(unsigned char), cudaMemcpyDeviceToHost);
  }

  printf("===\n");
  for (int i = 0; i < matrix_height; i++) {
    for (int j = 0; j < matrix_width; j++) {
      printf("%c ", h_horizontal_edges[i][j] + '0');
    }
    printf("\n");
  }
 
  for (int i = 0; i < matrix_height; i++) {
    cudaMemcpy(h_matrix[i], d_matrix+(i*matrix_width), matrix_width*sizeof(unsigned char), cudaMemcpyDeviceToHost);
  }

  cudaFree(d_horizontal_edges);
  cudaFree(d_vertical_edges);
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
