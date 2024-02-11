#include <math.h>
#include <stdio.h>

#include "sobel_feldman_core.hpp"
#include "edge_detection_core.hpp"

/**
 * Returns the global gradient of a pixel.
 **/
__device__ __host__ unsigned char global_gradient_core(int2 index, int *horizontal_edges, int *vertical_edges, dim3 matrix_dim) {
  int g_x = horizontal_edges[index.y * matrix_dim.x + index.x];
  int g_y = vertical_edges[index.y * matrix_dim.x + index.x];
  float global_gradient = sqrt((double) g_x * g_x + g_y * g_y);

  return global_gradient <= 255.0 ? (unsigned char) global_gradient : 255;
}


/**
 * Returns the angle of a pixel.
 **/
__device__ __host__ float angle_core(int2 index, int *horizontal_gradient, int *vertical_gradient, dim3 matrix_dim) {
  int g_x = horizontal_gradient[index.y * matrix_dim.x + index.x];
  int g_y = vertical_gradient[index.y * matrix_dim.x + index.x];
  float angle;
  
  if (g_x != 0 && g_y != 0) {
    angle = atan((float) g_y / g_x);
  } else {
    angle = -M_PI;
  }

  return angle; 
}

/**
 * Color a pixel depending on the value of the direction of its gradient.
 **/
__device__ __host__ void edge_color_core(int2 index, float *angle_matrix, unsigned char *output_image, dim3 image_dim) { 
  const float ANGLE = angle_matrix[index.y*image_dim.x + index.x] + M_PI_2;
  const int INT_INDEX = index.y*image_dim.x + index.x;
  
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
  } else if (get_color(ANGLE) == 'B')  {
    // Vertical gradient direction : Blue
    output_image[3 * (INT_INDEX)] = 0; 
    output_image[3 * (INT_INDEX) + 1] = 0; 
    output_image[3 * (INT_INDEX) + 2] = 255; 
  } else if (get_color(ANGLE) == ' ')  {
    output_image[3 * (INT_INDEX)] = 0; 
    output_image[3 * (INT_INDEX) + 1] = 0; 
    output_image[3 * (INT_INDEX) + 2] = 0; 
  }
}
