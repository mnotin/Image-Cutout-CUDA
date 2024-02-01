#include <math.h>

#include "cutout_core.hpp"

/**
 * First step of the cutout process.
 * Each gradient pixel with a value above the threshold is considered a border.
 **/
__device__ __host__ char draw_edges_on_cutout_matrix_core(int2 index, unsigned char *edge_matrix,
                                                          dim3 matrix_dim, int2 start_pixel,
                                                          int2 tracking_start_pixel, int threshold
) {
  char result = 'D'; // Discard

  if (start_pixel.x == index.x && start_pixel.y == index.y) {
    result = 'M'; // Marked
  } else if (tracking_start_pixel.x == index.x && tracking_start_pixel.y == index.y) {
    result = 'T'; // Tracking
  } else if (threshold < edge_matrix[index.y*matrix_dim.x + index.x]) {
    result = 'B'; // Border
  }

  return result;
}

/**
 * Main part of the cutout process.
 * Adds a pixel in the final target process if it is surrounded by at least one pixel already targeted. 
 **/
__device__ __host__ char cutout_algorithm_core(int2 index, char *cutout_matrix, dim3 matrix_dim,
                                               int2 read_limit, int *done, char *looking_pixels,
                                               char spread_pixel) {
  const int INT_INDEX = index.y*matrix_dim.x + index.x;
  char result_char = cutout_matrix[INT_INDEX];
  int i = 0;

  while (looking_pixels[i] != '\0') {
    if (cutout_matrix[INT_INDEX] == looking_pixels[i]) {
      if (0 < index.x && cutout_matrix[INT_INDEX-1] == spread_pixel ||
          index.x < read_limit.x-1 && cutout_matrix[INT_INDEX+1] == spread_pixel ||
          0 < index.y && cutout_matrix[INT_INDEX - matrix_dim.x] == spread_pixel ||
          index.y < read_limit.y-1 && cutout_matrix[INT_INDEX + matrix_dim.x] == spread_pixel) {
        cutout_matrix[INT_INDEX] = spread_pixel;
        result_char = spread_pixel;
        *done = 0;
      }
    }

    i++;
  }

  return result_char;
}

/**
 * Set the color of a pixel that is not targeted by the cutout process to black. 
 **/
__device__ __host__ void apply_cutout_core(int2 index, char *cutout_matrix, unsigned char *output_image, dim3 image_dim, int2 start_pixel) {
  const int INT_INDEX = index.y*image_dim.x + index.x;

  if (index.x == start_pixel.x && index.y == start_pixel.y) {
    output_image[3 * (INT_INDEX)] = 255;
    output_image[3 * (INT_INDEX) + 1] = 0;
    output_image[3 * (INT_INDEX) + 2] = 0;
  } else if (cutout_matrix[INT_INDEX] == 'T') {
    output_image[3 * (INT_INDEX)] = 0;
    output_image[3 * (INT_INDEX) + 1] = 0;
    output_image[3 * (INT_INDEX) + 2] = 255;
  }
}
