#include <math.h>

#include "sobel_feldman_launcher.hpp"

#include "../../kernel/edge_detection/sobel_feldman_kernel.hpp"
#include "../../core/edge_detection/sobel_feldman_core.hpp"
#include "../../main.hpp"
#include "../../kernel/utils/convolution_kernel.hpp"
#include "../../core/utils/convolution_core.hpp"

const int KERNEL_SIZE = 3;
const float SOBEL_HORIZONTAL_KERNEL[KERNEL_SIZE*KERNEL_SIZE] = { 1, 0,  -1, 
                                                                 2, 0,  -2, 
                                                                 1, 0, -1};
const float SOBEL_VERTICAL_KERNEL[KERNEL_SIZE*KERNEL_SIZE] = { 1,  2,  1,
                                                               0,  0,  0,
                                                               -1, -2, -1}; 
/**
 * Applies the Sobel-Feldman operator over a matrix.
 * The picture should have been smoothed and converted to grayscale prior to being passed over the Sobel-Feldman operator. 
 **/
void ProcessingUnitDevice::sobel_feldman(unsigned char *h_input_matrix, unsigned char *h_gradient_matrix, float *h_angle_matrix, dim3 matrix_dim) {
  dim3 block_dim(MATRIX_SIZE_PER_BLOCK, MATRIX_SIZE_PER_BLOCK);
  dim3 grid_dim(ceil((float) matrix_dim.x/MATRIX_SIZE_PER_BLOCK), ceil((float) matrix_dim.y/MATRIX_SIZE_PER_BLOCK));

  cudaStream_t cuda_streams[2];
  for (int i = 0; i < 2; ++i) {
    cudaStreamCreate(&cuda_streams[i]);
  }

  unsigned char *d_input_matrix;
  unsigned char *d_gradient_matrix;
  int *d_horizontal_gradient;
  int *d_vertical_gradient;
  float *d_angle_matrix;
  float *d_horizontal_kernel;
  float *d_vertical_kernel;

  cudaMalloc(&d_input_matrix, matrix_dim.x * matrix_dim.y * sizeof(unsigned char));
  cudaMalloc(&d_gradient_matrix, matrix_dim.x * matrix_dim.y * sizeof(unsigned char));
  cudaMalloc(&d_horizontal_gradient, matrix_dim.x * matrix_dim.y * sizeof(int));
  cudaMalloc(&d_vertical_gradient, matrix_dim.x * matrix_dim.y * sizeof(int));
  cudaMalloc(&d_angle_matrix, matrix_dim.x * matrix_dim.y * sizeof(float));
  cudaMalloc(&d_horizontal_kernel, KERNEL_SIZE*KERNEL_SIZE * sizeof(float));
  cudaMalloc(&d_vertical_kernel, KERNEL_SIZE*KERNEL_SIZE * sizeof(float));

  cudaMemcpy(d_input_matrix, h_input_matrix, matrix_dim.x * matrix_dim.y * sizeof(unsigned char), cudaMemcpyHostToDevice);

  // Horizontal gradient
  cudaMemcpyAsync(d_horizontal_kernel, SOBEL_HORIZONTAL_KERNEL, KERNEL_SIZE*KERNEL_SIZE * sizeof(int), cudaMemcpyHostToDevice, cuda_streams[0]);
  convolution_kernel<<<grid_dim, block_dim, 0, cuda_streams[0]>>>(d_input_matrix, d_horizontal_gradient, matrix_dim, d_horizontal_kernel, 3);

  // Vertical gradient
  cudaMemcpyAsync(d_vertical_kernel, SOBEL_VERTICAL_KERNEL, KERNEL_SIZE*KERNEL_SIZE * sizeof(int), cudaMemcpyHostToDevice, cuda_streams[1]);
  convolution_kernel<<<grid_dim, block_dim, 0, cuda_streams[1]>>>(d_input_matrix, d_vertical_gradient, matrix_dim, d_vertical_kernel, KERNEL_SIZE);

  cudaDeviceSynchronize();
  
  // Global gradient
  global_gradient_kernel<<<grid_dim, block_dim, 0, cuda_streams[0]>>>(d_gradient_matrix, d_horizontal_gradient, d_vertical_gradient, matrix_dim); 
  cudaMemcpyAsync(h_gradient_matrix, d_gradient_matrix, matrix_dim.x * matrix_dim.y * sizeof(unsigned char), cudaMemcpyDeviceToHost, cuda_streams[0]);
 
  // Angle of the gradient
  angle_kernel<<<grid_dim, block_dim, 0, cuda_streams[1]>>>(d_angle_matrix, d_horizontal_gradient, d_vertical_gradient, matrix_dim);
  cudaMemcpyAsync(h_angle_matrix, d_angle_matrix, matrix_dim.x * matrix_dim.y * sizeof(float), cudaMemcpyDeviceToHost, cuda_streams[1]);

  cudaDeviceSynchronize();

  for (int i = 0; i < 2; ++i) {
    cudaStreamDestroy(cuda_streams[i]);
  }
  cudaFree(d_input_matrix);
  cudaFree(d_gradient_matrix);
  cudaFree(d_horizontal_gradient);
  cudaFree(d_vertical_gradient);
  cudaFree(d_angle_matrix);
  cudaFree(d_horizontal_kernel);
  cudaFree(d_vertical_kernel);
}

void ProcessingUnitHost::sobel_feldman(unsigned char *input_matrix, unsigned char *gradient_matrix, float *angle_matrix, dim3 matrix_dim) {
  int *horizontal_gradient = new int[matrix_dim.x * matrix_dim.y];
  int *vertical_gradient = new int[matrix_dim.x * matrix_dim.y];

  int2 index;

  // Horizontal gradient
  for (index.y = 0; index.y < matrix_dim.y; index.y++) {
    for (index.x = 0; index.x < matrix_dim.x; index.x++) {
      horizontal_gradient[index.y*matrix_dim.x + index.x] = convolution_core(index, input_matrix, matrix_dim, SOBEL_HORIZONTAL_KERNEL, KERNEL_SIZE);
    }
  }

  // Vertical gradient
  for (index.y = 0; index.y < matrix_dim.y; index.y++) {
    for (index.x = 0; index.x < matrix_dim.x; index.x++) {
      vertical_gradient[index.y*matrix_dim.x + index.x] = convolution_core(index, input_matrix, matrix_dim, SOBEL_VERTICAL_KERNEL, KERNEL_SIZE);
    }
  }
  
  // Global gradient
  for (index.y = 0; index.y < matrix_dim.y; index.y++) {
    for (index.x = 0; index.x < matrix_dim.x; index.x++) {
      gradient_matrix[index.y*matrix_dim.x + index.x] = global_gradient_core(index, horizontal_gradient, vertical_gradient, matrix_dim); 
    }
  }
  
  for (index.y = 0; index.y < matrix_dim.y; index.y++) {
    for (index.x = 0; index.x < matrix_dim.x; index.x++) {
      angle_matrix[index.y*matrix_dim.x + index.x] = angle_core(index, horizontal_gradient, vertical_gradient, matrix_dim);
    }
  }
  
  delete [] horizontal_gradient;
  delete [] vertical_gradient;
}



void ProcessingUnitDevice::generate_edge_color(unsigned char *h_gradient_matrix, float *h_angle_matrix, unsigned char *h_output_image, dim3 matrix_dim) {
  dim3 block_dim(MATRIX_SIZE_PER_BLOCK, MATRIX_SIZE_PER_BLOCK);
  dim3 grid_dim(ceil((float) matrix_dim.x/MATRIX_SIZE_PER_BLOCK), ceil((float) matrix_dim.y/MATRIX_SIZE_PER_BLOCK));

  unsigned char *d_gradient_matrix;
  float *d_angle_matrix;
  unsigned char *d_output_image;

  cudaMalloc(&d_gradient_matrix, matrix_dim.x * matrix_dim.y * sizeof(unsigned char));
  cudaMalloc(&d_angle_matrix, matrix_dim.x * matrix_dim.y * sizeof(float));
  cudaMalloc(&d_output_image, 3 * matrix_dim.x * matrix_dim.y * sizeof(unsigned char));

  cudaMemcpy(d_gradient_matrix, h_gradient_matrix, matrix_dim.x * matrix_dim.y * sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_angle_matrix, h_angle_matrix, matrix_dim.x * matrix_dim.y * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_output_image, h_output_image, 3 * matrix_dim.x * matrix_dim.y * sizeof(unsigned char), cudaMemcpyHostToDevice);

  edge_color_kernel<<<grid_dim, block_dim>>>(d_gradient_matrix, d_angle_matrix, d_output_image, matrix_dim);

  cudaMemcpy(h_output_image, d_output_image, 3 * matrix_dim.x * matrix_dim.y * sizeof(unsigned char), cudaMemcpyDeviceToHost);

  cudaFree(d_gradient_matrix);
  cudaFree(d_angle_matrix);
  cudaFree(d_output_image);
}

void ProcessingUnitHost::generate_edge_color(unsigned char *gradient_matrix, float *angle_matrix, unsigned char *output_image, dim3 matrix_dim) {
  int2 index;

  for (index.y = 0; index.y < matrix_dim.y; index.y++) {
    for (index.x = 0; index.x < matrix_dim.x; index.x++) {
      edge_color_core(index, gradient_matrix, angle_matrix, output_image, matrix_dim);
    }
  }
}
