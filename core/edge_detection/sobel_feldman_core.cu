#include <math.h>

#include "sobel_feldman_core.hpp"
#include "edge_detection_core.hpp"

__device__ __host__ unsigned char global_gradient_core(int2 index, int *horizontal_edges, int *vertical_edges, dim3 matrix_dim) {
  int g_x = horizontal_edges[index.y * matrix_dim.x + index.x];
  int g_y = vertical_edges[index.y * matrix_dim.x + index.x];
  float global_gradient = sqrt((double) g_x * g_x + g_y * g_y);

  return global_gradient <= 255.0 ? (unsigned char) global_gradient : 255;
}

__device__ __host__ float angle_core(int2 index, int *horizontal_gradient, int *vertical_gradient, dim3 matrix_dim) {
  int g_x = horizontal_gradient[index.y * matrix_dim.x + index.x];
  int g_y = vertical_gradient[index.y * matrix_dim.x + index.x];
  float angle = atan((float) g_y / g_x);

  return angle; 
}

__device__ __host__ void edge_color_core(int2 index, unsigned char *gradient_matrix, float *angle_matrix, unsigned char *output_image, dim3 image_dim) { 
  const float ANGLE = angle_matrix[index.y*image_dim.x + index.x] + M_PI_2;
  const int INT_INDEX = index.y*image_dim.x + index.x;
  
  if (50 < gradient_matrix[INT_INDEX]) {
    if (get_color(ANGLE) == 'Y') {
      // Horizontal gradient direction : Yellow
      output_image[3 * (INT_INDEX)] = 255;
      output_image[3 * (INT_INDEX) + 1] = 255; 
      output_image[3 * (INT_INDEX) + 2] = 0; 
    } else if (get_color(ANGLE) == 'G') {
      // Top right gradient direction : Green
      output_image[3 * (INT_INDEX)] = 0; 
      output_image[3 * (INT_INDEX) + 1] = 255; 
      output_image[3 * (INT_INDEX) + 2] = 0; 
    } else if (get_color(ANGLE) == 'R')  {
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
