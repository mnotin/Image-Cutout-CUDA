#include <iostream>

#include "utils.hpp"
#include "../main.hpp"

/**
 * Applies discrete convolution over a matrix using a given kernel.
 * This kernel should be called using appropriate number of grids, blocks and threads to match the resolution of the image.
 **/
__global__ void convolution(unsigned char *input_matrix, int *output_matrix, int matrix_width, int matrix_height, float *kernel, int kernel_size) {
  int globalIdxX = threadIdx.x + (blockIdx.x * blockDim.x);
  int globalIdxY = threadIdx.y + (blockIdx.y * blockDim.y);
  int localIdxX = threadIdx.x;
  int localIdxY = threadIdx.y;
  
  int current_matrix_index = globalIdxY*matrix_width + globalIdxX;
  int current_shared_matrix_index = MATRIX_SIZE_PER_BLOCK+2+1+ localIdxY*(MATRIX_SIZE_PER_BLOCK+2) + localIdxX;

  __shared__ unsigned char shared_matrix[(MATRIX_SIZE_PER_BLOCK+2)*(MATRIX_SIZE_PER_BLOCK+2)];

  /*
   * x x x x x x MATRIX_SIZE_PER_BLOCK + 2
   * x o o o o x
   * x o o o o x
   * x o o o o x
   * x o o o o x
   * x x x x x x
   */
  shared_matrix[current_shared_matrix_index] = input_matrix[current_matrix_index];

  // Handle the borders of each block
  if (localIdxX == 0 && localIdxY == 0) {
    // Fill the edges
    for (int i = 0; i < MATRIX_SIZE_PER_BLOCK+2; i++) {
      // First line
      int first_line_offset = -1;
      if (0 == globalIdxY) {
        first_line_offset = 0;
      }
      shared_matrix[i] = input_matrix[(globalIdxY+first_line_offset)*matrix_width + globalIdxX + i - 1];
      
      // Last line
      int last_line_offset = 0;
      if (globalIdxY+MATRIX_SIZE_PER_BLOCK == matrix_height) {
        last_line_offset = -1;
      }
      shared_matrix[(MATRIX_SIZE_PER_BLOCK+2)*(MATRIX_SIZE_PER_BLOCK+1)+i] =
        input_matrix[(globalIdxY+MATRIX_SIZE_PER_BLOCK+last_line_offset)*matrix_width + globalIdxX + i - 1];
    }

    for (int i = 0; i < MATRIX_SIZE_PER_BLOCK; i++) {
      // Left side
      int left_side_offset = -1;
      if (0 == globalIdxX) {
        left_side_offset = 0;
      }
      shared_matrix[MATRIX_SIZE_PER_BLOCK+2 + i*(MATRIX_SIZE_PER_BLOCK+2)] = 
        input_matrix[(globalIdxY+i)*matrix_width + globalIdxX + left_side_offset];

      // Right side
      int right_side_offset = 0;
      if (globalIdxX+MATRIX_SIZE_PER_BLOCK == matrix_width) {
        right_side_offset = -1;
      }
      shared_matrix[MATRIX_SIZE_PER_BLOCK+2 + (i+1)*(MATRIX_SIZE_PER_BLOCK+2) - 1] =
        input_matrix[(globalIdxY+i)*matrix_width + globalIdxX+MATRIX_SIZE_PER_BLOCK + right_side_offset];
    }
  }
  __syncthreads();

  int convolution_result = 0;

  for (int i = 0; i < kernel_size; i++) {
    for (int j = 0; j < kernel_size; j++) {
      int vertical_offset = ((localIdxY + i) - (int)floor(kernel_size/2.0));
      int horizontal_offset = (localIdxX + j) - (int)floor(kernel_size/2.0);
      int tmp_index = vertical_offset*(MATRIX_SIZE_PER_BLOCK+2) + horizontal_offset;
      
      convolution_result += shared_matrix[MATRIX_SIZE_PER_BLOCK+2+1 + tmp_index] * kernel[i*kernel_size + j];
    }
  }

  if (255 < abs(convolution_result)) {
    convolution_result = convolution_result < 0 ? -255 : 255;
  }
  
  output_matrix[current_matrix_index] = convolution_result;
}

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
  std::cout << "Nombre de blocs lancés: " << blocks.x << " " << blocks.y << std::endl;
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
