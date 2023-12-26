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

void cutout(unsigned char *h_rgb_image, unsigned char *h_edge_matrix, int matrix_width, int matrix_height, int start_pixel_x, int start_pixel_y) {
  int *h_done = (int *) malloc(sizeof(int));
  unsigned char **h_cutout_matrix;

  *h_done = 0;
  h_cutout_matrix = (unsigned char **) malloc(matrix_height * sizeof(unsigned char*));
  for (int i = 0; i < matrix_height; i++) {
    h_cutout_matrix[i] = (unsigned char *) malloc(matrix_width * sizeof(unsigned char));
  }
  
  for (int i = 0; i < matrix_height; i++) {
    for (int j = 0; j < matrix_height; j++) {
      h_cutout_matrix[i][j] = 'D';
    }
  }

  unsigned char *d_rgb_image;
  unsigned char *d_edge_matrix;
  unsigned char *d_cutout_matrix;
  int *d_done;

  cudaMalloc((void **) &d_rgb_image, 3 * matrix_width * matrix_height * sizeof(unsigned char));
  cudaMalloc((void **) &d_edge_matrix, matrix_width * matrix_height * sizeof(unsigned char));
  cudaMalloc((void **) &d_cutout_matrix, matrix_width * matrix_height * sizeof(unsigned char));
  cudaMalloc((void **) &d_done, sizeof(int));

  cudaMemcpy(d_rgb_image, h_rgb_image, 3 * matrix_width * matrix_height * sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_edge_matrix, h_edge_matrix, matrix_width * matrix_height * sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_cutout_matrix, h_cutout_matrix, matrix_width * matrix_height * sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_done, h_done, sizeof(int), cudaMemcpyHostToDevice);
  
  dim3 threads = dim3(MATRIX_SIZE_PER_BLOCK, MATRIX_SIZE_PER_BLOCK);
  dim3 blocks = dim3(matrix_width/MATRIX_SIZE_PER_BLOCK, matrix_height/MATRIX_SIZE_PER_BLOCK);
  draw_edges_on_cutout_matrix<<<blocks, threads>>>(d_edge_matrix, d_cutout_matrix, matrix_width, matrix_height, start_pixel_x, start_pixel_y);
  while (*h_done == 0) {
    *h_done = 1;
    cudaMemcpy(d_done, h_done, sizeof(int), cudaMemcpyHostToDevice);
    cutout_algorithm<<<blocks, threads>>>(d_edge_matrix, d_cutout_matrix, matrix_width, matrix_height, d_done);
    cudaMemcpy(h_done, d_done, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
  }
  apply_cutout<<<blocks, threads>>>(d_cutout_matrix, d_rgb_image, matrix_width, matrix_height);

  cudaMemcpy(h_rgb_image, d_rgb_image, 3 * matrix_width * matrix_height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

  cudaFree(d_rgb_image);
  cudaFree(d_edge_matrix);
  cudaFree(d_cutout_matrix);
  cudaFree(d_done);
}

__global__ void draw_edges_on_cutout_matrix(unsigned char *edge_matrix, unsigned char *cutout_matrix, int matrix_width, int matrix_height, int start_pixel_x, int start_pixel_y) {
  int globalIdxX = threadIdx.x + (blockIdx.x * blockDim.x);
  int globalIdxY = threadIdx.y + (blockIdx.y * blockDim.y);

  int threshold = 250;

  if (globalIdxX < matrix_width && globalIdxY < matrix_height && threshold < edge_matrix[globalIdxY*matrix_width + matrix_height]) {
    cutout_matrix[globalIdxY*matrix_width + matrix_height] = 'B'; 
  }
  
  __syncthreads();

  cutout_matrix[start_pixel_y*matrix_width + start_pixel_x] = 'A';
}

/**
 * The input_matrix is the output of an edge detection operator (Sobel-Feldman or Canny)  
 **/
__global__ void cutout_algorithm(unsigned char *edge_matrix, unsigned char *cutout_matrix, int matrix_width, int matrix_height, int *done) {
  int globalIdxX = threadIdx.x + (blockIdx.x * blockDim.x);
  int globalIdxY = threadIdx.y + (blockIdx.y * blockDim.y);

  __shared__ int shared_done;

  if (globalIdxX == 0 && globalIdxY == 0) {
    shared_done = 1; // Initialize the variable of the block
  }

  __syncthreads();

  // Process
  if (cutout_matrix[globalIdxY*matrix_width + globalIdxX] == 'A') {
    // Active pixel
    if (cutout_matrix[globalIdxY*matrix_width + globalIdxX-1] == 'D') {
      cutout_matrix[globalIdxY*matrix_width + globalIdxX-1] = 'A';
      shared_done = 0;
    }
    
    if (cutout_matrix[globalIdxY*matrix_width + globalIdxX+1] == 'D') {
      cutout_matrix[globalIdxY*matrix_width + globalIdxX+1] = 'A';
      shared_done = 0;
    }
    
    if (cutout_matrix[(globalIdxY-1)*matrix_width + globalIdxX] == 'D') {
      cutout_matrix[(globalIdxY-1)*matrix_width + globalIdxX] = 'A';
      shared_done = 0;
    }
    
    if (cutout_matrix[(globalIdxY+1)*matrix_width + globalIdxX] == 'D') {
      cutout_matrix[(globalIdxY+1)*matrix_width + globalIdxX-1] = 'A';
      shared_done = 0;
    }
  }

  __syncthreads();
  
  cutout_matrix[globalIdxY*matrix_width + globalIdxX] = 'M';
  if (globalIdxX == 0 && globalIdxY == 0 && shared_done == 0) {
    *done = 0; 
  }
}

__global__ void apply_cutout(unsigned char *cutout_matrix, unsigned char *output_image, int image_width, int image_height) { 
  int globalIdxX = threadIdx.x + (blockIdx.x * blockDim.x);
  int globalIdxY = threadIdx.y + (blockIdx.y * blockDim.y);

  if (cutout_matrix[globalIdxY*image_width + globalIdxY] == 'D') {
    output_image[3 * (globalIdxY*image_width + globalIdxX)] = 0; 
    output_image[3 * (globalIdxY*image_width + globalIdxX) + 1] = 0; 
    output_image[3 * (globalIdxY*image_width + globalIdxX) + 2] = 0; 
  }
}
