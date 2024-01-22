#include <iostream>
#include <math.h>

#include "sobel_feldman.hpp"
#include "../main.hpp"
#include "../utils/convolution.hpp"

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
void ProcessingUnitDevice::sobel_feldman(unsigned char *h_input_matrix, unsigned char *h_gradient_matrix, float *h_angle_matrix, Dim matrix_dim) {
  dim3 threads = dim3(MATRIX_SIZE_PER_BLOCK, MATRIX_SIZE_PER_BLOCK);
  dim3 blocks = dim3(matrix_dim.width/MATRIX_SIZE_PER_BLOCK, matrix_dim.height/MATRIX_SIZE_PER_BLOCK);

  unsigned char *d_input_matrix;
  unsigned char *d_gradient_matrix;
  int *d_horizontal_gradient;
  int *d_vertical_gradient;
  float *d_angle_matrix;
  float *d_kernel;

  cudaMalloc(&d_input_matrix, matrix_dim.width * matrix_dim.height * sizeof(unsigned char));
  cudaMalloc(&d_gradient_matrix, matrix_dim.width * matrix_dim.height * sizeof(unsigned char));
  cudaMalloc(&d_horizontal_gradient, matrix_dim.width * matrix_dim.height * sizeof(int));
  cudaMalloc(&d_vertical_gradient, matrix_dim.width * matrix_dim.height * sizeof(int));
  cudaMalloc(&d_angle_matrix, matrix_dim.width * matrix_dim.height * sizeof(float));
  cudaMalloc(&d_kernel, KERNEL_SIZE*KERNEL_SIZE * sizeof(float));

  cudaMemcpy(d_input_matrix, h_input_matrix, matrix_dim.width * matrix_dim.height * sizeof(unsigned char), cudaMemcpyHostToDevice);

  // Horizontal gradient
  cudaMemcpy(d_kernel, SOBEL_HORIZONTAL_KERNEL, KERNEL_SIZE*KERNEL_SIZE * sizeof(int), cudaMemcpyHostToDevice);
  std::cout << "Nombre de blocs lancés: " << blocks.x << " " << blocks.y << std::endl;
  convolution_kernel<<<blocks, threads>>>(d_input_matrix, d_horizontal_gradient, matrix_dim, d_kernel, 3);
  cudaDeviceSynchronize();

  // Vertical gradient
  cudaMemcpy(d_kernel, SOBEL_VERTICAL_KERNEL, KERNEL_SIZE*KERNEL_SIZE * sizeof(int), cudaMemcpyHostToDevice);
  convolution_kernel<<<blocks, threads>>>(d_input_matrix, d_vertical_gradient, matrix_dim, d_kernel, KERNEL_SIZE);
  cudaDeviceSynchronize();
  
  // Global gradient
  global_gradient_kernel<<<blocks, threads>>>(d_gradient_matrix, d_horizontal_gradient, d_vertical_gradient, matrix_dim); 
  cudaMemcpy(h_gradient_matrix, d_gradient_matrix, matrix_dim.width * matrix_dim.height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
 
  // Angle of the gradient
  angle_kernel<<<blocks, threads>>>(d_horizontal_gradient, d_vertical_gradient, d_angle_matrix, matrix_dim);
  cudaMemcpy(h_angle_matrix, d_angle_matrix, matrix_dim.width * matrix_dim.height * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  cudaFree(d_input_matrix);
  cudaFree(d_gradient_matrix);
  cudaFree(d_horizontal_gradient);
  cudaFree(d_vertical_gradient);
  cudaFree(d_angle_matrix);
  cudaFree(d_kernel);
}

void ProcessingUnitHost::sobel_feldman(unsigned char *input_matrix, unsigned char *gradient_matrix, float *angle_matrix, Dim matrix_dim) {
  int *horizontal_gradient = new int[matrix_dim.width * matrix_dim.height];
  int *vertical_gradient = new int[matrix_dim.width * matrix_dim.height];

  Vec2 index;

  // Horizontal gradient
  for (index.y = 0; index.y < matrix_dim.height; index.y++) {
    for (index.x = 0; index.x < matrix_dim.width; index.x++) {
      horizontal_gradient[index.y*matrix_dim.width + index.x] = convolution_core(index, input_matrix, matrix_dim, SOBEL_HORIZONTAL_KERNEL, KERNEL_SIZE);
    }
  }

  // Vertical gradient
  for (index.y = 0; index.y < matrix_dim.height; index.y++) {
    for (index.x = 0; index.x < matrix_dim.width; index.x++) {
      vertical_gradient[index.y*matrix_dim.width + index.x] = convolution_core(index, input_matrix, matrix_dim, SOBEL_VERTICAL_KERNEL, KERNEL_SIZE);
    }
  }
  
  // Global gradient
  for (index.y = 0; index.y < matrix_dim.height; index.y++) {
    for (index.x = 0; index.x < matrix_dim.width; index.x++) {
      gradient_matrix[index.y*matrix_dim.width + index.x] = global_gradient_core(index, horizontal_gradient, vertical_gradient, matrix_dim); 
    }
  }
  
  for (index.y = 0; index.y < matrix_dim.height; index.y++) {
    for (index.x = 0; index.x < matrix_dim.width; index.x++) {
      angle_matrix[index.y*matrix_dim.width + index.x] = angle_core(index, horizontal_gradient, vertical_gradient, matrix_dim);
    }
  }
  
  delete [] horizontal_gradient;
  delete [] vertical_gradient;
}


/**
 * Computes the global gradient of an image after being processed by the Sobel-Feldman operator.
 **/
__global__ void global_gradient_kernel(unsigned char *output_matrix, int *horizontal_edges, int *vertical_edges, Dim matrix_dim) {
  Vec2 index;
  index.x = threadIdx.x + (blockIdx.x * blockDim.x);
  index.y = threadIdx.y + (blockIdx.y * blockDim.y);

  output_matrix[index.y * matrix_dim.width + index.x] = global_gradient_core(index, horizontal_edges, vertical_edges, matrix_dim);
}

__device__ __host__ unsigned char global_gradient_core(Vec2 index, int *horizontal_edges, int *vertical_edges, Dim matrix_dim) {
  int g_x = horizontal_edges[index.y * matrix_dim.width + index.x];
  int g_y = vertical_edges[index.y * matrix_dim.width + index.x];
  float global_gradient = sqrt((double) g_x * g_x + g_y * g_y);

  return global_gradient <= 255.0 ? (unsigned char) global_gradient : 255;
}


__global__ void angle_kernel(int *horizontal_gradient, int *vertical_gradient, float *angle_matrix, Dim matrix_dim) {
  Vec2 index;
  index.x = threadIdx.x + (blockIdx.x * blockDim.x);
  index.y = threadIdx.y + (blockIdx.y * blockDim.y);

  angle_matrix[index.y * matrix_dim.width + index.x] = angle_core(index, horizontal_gradient, vertical_gradient, matrix_dim); 
}

__device__ __host__ float angle_core(Vec2 index, int *horizontal_gradient, int *vertical_gradient, Dim matrix_dim) {
  int g_x = horizontal_gradient[index.y * matrix_dim.width + index.x];
  int g_y = vertical_gradient[index.y * matrix_dim.width + index.x];
  float angle = atan((float) g_y / g_x);

  return angle; 
}


void ProcessingUnitDevice::generate_edge_color(unsigned char *h_gradient_matrix, float *h_angle_matrix, unsigned char *h_output_image, Dim matrix_dim) {
  unsigned char *d_gradient_matrix;
  float *d_angle_matrix;
  unsigned char *d_output_image;

  cudaMalloc(&d_gradient_matrix, matrix_dim.width * matrix_dim.height * sizeof(unsigned char));
  cudaMalloc(&d_angle_matrix, matrix_dim.width * matrix_dim.height * sizeof(float));
  cudaMalloc(&d_output_image, 3 * matrix_dim.width * matrix_dim.height * sizeof(unsigned char));

  cudaMemcpy(d_gradient_matrix, h_gradient_matrix, matrix_dim.width * matrix_dim.height * sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_angle_matrix, h_angle_matrix, matrix_dim.width * matrix_dim.height * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_output_image, h_output_image, 3 * matrix_dim.width * matrix_dim.height * sizeof(unsigned char), cudaMemcpyHostToDevice);

  dim3 threads = dim3(MATRIX_SIZE_PER_BLOCK, MATRIX_SIZE_PER_BLOCK);
  dim3 blocks = dim3(matrix_dim.width/MATRIX_SIZE_PER_BLOCK, matrix_dim.height/MATRIX_SIZE_PER_BLOCK);
  std::cout << "Nombre de blocs lancés: " << blocks.x << " " << blocks.y << std::endl;
  edge_color_kernel<<<blocks, threads>>>(d_gradient_matrix, d_angle_matrix, d_output_image, matrix_dim);

  cudaMemcpy(h_output_image, d_output_image, 3 * matrix_dim.width * matrix_dim.height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

  cudaFree(d_gradient_matrix);
  cudaFree(d_angle_matrix);
  cudaFree(d_output_image);
}

void ProcessingUnitHost::generate_edge_color(unsigned char *gradient_matrix, float *angle_matrix, unsigned char *output_image, Dim matrix_dim) {
  Vec2 index;
  for (index.y = 0; index.y < matrix_dim.height; index.y++) {
    for (index.x = 0; index.x < matrix_dim.width; index.x++) {
      edge_color_core(index, gradient_matrix, angle_matrix, output_image, matrix_dim);
    }
  }
}

/**
 * Give a color to edges depending on their direction.
 **/
__global__ void edge_color_kernel(unsigned char *gradient_matrix, float *angle_matrix, unsigned char *output_image, Dim image_dim) { 
  Vec2 index;
  index.x = threadIdx.x + (blockIdx.x * blockDim.x);
  index.y = threadIdx.y + (blockIdx.y * blockDim.y);
  
  edge_color_core(index, gradient_matrix, angle_matrix, output_image, image_dim);
}

__device__ __host__ void edge_color_core(Vec2 index, unsigned char *gradient_matrix, float *angle_matrix, unsigned char *output_image, Dim image_dim) { 
  const float ANGLE = angle_matrix[index.y*image_dim.width + index.x] + M_PI_2;
  const int INT_INDEX = index.y*image_dim.width + index.x;
  
  if (50 < gradient_matrix[INT_INDEX]) {
    if (get_color_sobel(ANGLE) == 'Y') {
      // Horizontal gradient direction : Yellow
      output_image[3 * (INT_INDEX)] = 255;
      output_image[3 * (INT_INDEX) + 1] = 255; 
      output_image[3 * (INT_INDEX) + 2] = 0; 
    } else if (get_color_sobel(ANGLE) == 'G') {
      // Top right gradient direction : Green
      output_image[3 * (INT_INDEX)] = 0; 
      output_image[3 * (INT_INDEX) + 1] = 255; 
      output_image[3 * (INT_INDEX) + 2] = 0; 
    } else if (get_color_sobel(ANGLE) == 'R')  {
      // Top left gradient direction : Red
      output_image[3 * (INT_INDEX)] = 255; 
      output_image[3 * (INT_INDEX) + 1] = 0; 
      output_image[3 * (INT_INDEX) + 2] = 0; 
    } else {
      // Vertical gradient direction : Blue
      output_image[3 * (INT_INDEX)] = 0; 
      output_image[3 * (INT_INDEX) + 1] = 0; 
      output_image[3 * (INT_INDEX) + 2] = 255; 
    }
  } else {
    output_image[3 * (INT_INDEX)] = 0; 
    output_image[3 * (INT_INDEX) + 1] = 0; 
    output_image[3 * (INT_INDEX) + 2] = 0; 
  }
}

__device__ __host__ char get_color_sobel(float angle) {
  char color = ' ';

  if (angle < M_PI / 8.0 || (M_PI / 8.0) * 7 < angle) {
    // Horizontal gradient direction : Yellow
    color = 'Y';
  } else if (M_PI / 8.0 < angle && angle < (M_PI / 8.0) * 3) {
    // Top right gradient direction : Green
    color = 'G';
  } else if ((M_PI / 8.0) * 5 < angle && angle < (M_PI / 8.0) * 7) {
    // Top left gradient direction : Red
    color = 'R';
  } else {
    // Vertical gradient direction : Blue
    color = 'B';
  }

  return color;
}
