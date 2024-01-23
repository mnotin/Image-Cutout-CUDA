
#include <iostream>

#include "gaussian_blur.hpp"
#include "convolution.hpp"
#include "../main.hpp"

/**
 * Applies a gaussian blur over a matrix.
 **/
void ProcessingUnitDevice::gaussian_blur(unsigned char *h_matrix, Dim matrix_dim) {
  const int KERNEL_WIDTH = 3;
  float gaussian_blur_kernel[KERNEL_WIDTH*KERNEL_WIDTH] = {1/16.0, 2/16.0, 1/16.0, 
                                                           2/16.0, 4/16.0, 2/16.0, 
                                                           1/16.0, 2/16.0, 1/16.0};
  int h_int_matrix[matrix_dim.width*matrix_dim.height];
 
  unsigned char *d_input_matrix;
  int *d_output_matrix;
  float *d_kernel;
  cudaMalloc(&d_input_matrix, matrix_dim.width * matrix_dim.height * sizeof(unsigned char));
  cudaMalloc(&d_output_matrix, matrix_dim.width * matrix_dim.height * sizeof(int));
  cudaMalloc(&d_kernel, KERNEL_WIDTH*KERNEL_WIDTH * sizeof(float));

  cudaMemcpy(d_input_matrix, h_matrix, matrix_dim.width*matrix_dim.height*sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel, gaussian_blur_kernel, KERNEL_WIDTH*KERNEL_WIDTH * sizeof(int), cudaMemcpyHostToDevice);

  dim3 threads = dim3(MATRIX_SIZE_PER_BLOCK, MATRIX_SIZE_PER_BLOCK);
  dim3 blocks = dim3(matrix_dim.width/MATRIX_SIZE_PER_BLOCK, matrix_dim.height/MATRIX_SIZE_PER_BLOCK);
  std::cout << "Nombre de blocs lancés: " << blocks.x << " " << blocks.y << std::endl;
  convolution_kernel<<<blocks, threads>>>(d_input_matrix, d_output_matrix, matrix_dim, d_kernel, KERNEL_WIDTH);
 
  cudaMemcpy(h_int_matrix, d_output_matrix, matrix_dim.width*matrix_dim.height*sizeof(int), cudaMemcpyDeviceToHost);

  for (int i = 0; i < matrix_dim.height; i++) {
    for (int j = 0; j < matrix_dim.width; j++) {
      h_matrix[i*matrix_dim.width + j] = h_int_matrix[i*matrix_dim.width + j];
    }
  }

  cudaFree(d_input_matrix);
  cudaFree(d_output_matrix);
  cudaFree(d_kernel);
}

/**
 * Applies a gaussian blur over a matrix.
 **/
void ProcessingUnitHost::gaussian_blur(unsigned char *matrix, Dim matrix_dim) {
  const int KERNEL_WIDTH = 3;
  float gaussian_blur_kernel[KERNEL_WIDTH*KERNEL_WIDTH] = {1/16.0, 2/16.0, 1/16.0, 
                                                         2/16.0, 4/16.0, 2/16.0, 
                                                         1/16.0, 2/16.0, 1/16.0};
  int int_matrix[matrix_dim.width*matrix_dim.height];
  int *output_matrix = new int[matrix_dim.width * matrix_dim.height];

  for (int i = 0; i < matrix_dim.height; i++) {
    for (int j = 0; j < matrix_dim.width; j++) {
      int2 index = make_int2(j, i);

      output_matrix[i*matrix_dim.width + j] = convolution_core(index, 
        matrix,
        matrix_dim,
        gaussian_blur_kernel,
        KERNEL_WIDTH);
    }
  }
 
  memcpy(int_matrix, output_matrix, matrix_dim.width*matrix_dim.height*sizeof(int));

  for (int i = 0; i < matrix_dim.height; i++) {
    for (int j = 0; j < matrix_dim.width; j++) {
      matrix[i*matrix_dim.width + j] = int_matrix[i*matrix_dim.width + j];
    }
  }

  delete [] output_matrix;
}
