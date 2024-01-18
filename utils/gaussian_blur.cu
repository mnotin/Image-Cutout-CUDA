
#include <iostream>

#include "gaussian_blur.hpp"
#include "convolution.hpp"
#include "../main.hpp"

/**
 * Applies a gaussian blur over a matrix.
 **/
void ProcessingUnitDevice::gaussian_blur(unsigned char *h_matrix, int matrix_width, int matrix_height) {
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
  convolution_kernel<<<blocks, threads>>>(d_input_matrix, d_output_matrix, matrix_width, matrix_height, d_kernel, KERNEL_WIDTH);
 
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

/**
 * Applies a gaussian blur over a matrix.
 **/
void ProcessingUnitHost::gaussian_blur(unsigned char *matrix, int matrix_width, int matrix_height) {
  const int KERNEL_WIDTH = 3;
  float gaussian_blur_kernel[KERNEL_WIDTH*KERNEL_WIDTH] = {1/16.0, 2/16.0, 1/16.0, 
                                                         2/16.0, 4/16.0, 2/16.0, 
                                                         1/16.0, 2/16.0, 1/16.0};
  int int_matrix[matrix_width*matrix_height];
  int *output_matrix = new int[matrix_width * matrix_height];

  for (int i = 0; i < matrix_height; i++) {
    for (int j = 0; j < matrix_width; j++) {
      Vec2 index;
      index.x = j;
      index.y = i;

      output_matrix[i*matrix_width + j] = convolution_core(index, 
        matrix,
        output_matrix,
        matrix_width,
        matrix_height,
        gaussian_blur_kernel,
        KERNEL_WIDTH);
    }
  }
 
  memcpy(int_matrix, output_matrix, matrix_width*matrix_height*sizeof(int));

  for (int i = 0; i < matrix_height; i++) {
    for (int j = 0; j < matrix_width; j++) {
      matrix[i*matrix_width + j] = int_matrix[i*matrix_width + j];
    }
  }

  delete [] output_matrix;
}
