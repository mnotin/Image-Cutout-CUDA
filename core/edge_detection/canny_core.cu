#include <math.h>

#include "canny_core.hpp"
#include "edge_detection_core.hpp"

__device__ __host__ unsigned char non_maximum_suppression_core(int2 index, unsigned char *gradient_matrix, float *angle_matrix, dim3 matrix_dim) {
  const int INT_INDEX = index.y*matrix_dim.x + index.x;
  const float ANGLE = angle_matrix[INT_INDEX] + M_PI_2;
  unsigned char final_value = gradient_matrix[INT_INDEX];

  if (get_color(ANGLE) == 'Y') {
    // Vertical gradient direction : Yellow
    if (0 < index.y && gradient_matrix[INT_INDEX] < gradient_matrix[INT_INDEX - matrix_dim.x] &&
          get_color(angle_matrix[INT_INDEX - matrix_dim.x] + M_PI_2) == 'Y') {
      final_value = 0;
    } else if (index.y < matrix_dim.y - 1 && gradient_matrix[INT_INDEX] < gradient_matrix[INT_INDEX + matrix_dim.x] &&
        get_color(angle_matrix[INT_INDEX + matrix_dim.x] + M_PI_2) == 'Y') {
      final_value = 0;
    }
  } else if (get_color(ANGLE) == 'G') {
    // Top right gradient direction : Green
    if (index.x < matrix_dim.x-1 && 0 < index.y && gradient_matrix[INT_INDEX] < gradient_matrix[INT_INDEX - matrix_dim.x + 1] &&
      get_color(angle_matrix[INT_INDEX - matrix_dim.x + 1] + M_PI_2) == 'G') {
      final_value = 0;
    } else if (0 < index.x && index.y < matrix_dim.y-1 && gradient_matrix[INT_INDEX] < gradient_matrix[INT_INDEX + matrix_dim.x - 1] &&
      get_color(angle_matrix[INT_INDEX + matrix_dim.x - 1] + M_PI_2) == 'G') {
      final_value = 0;
    }
  } else if (get_color(ANGLE) == 'R') {
    // Top left gradient direction : Red
    if (0 < index.x && 0 < index.y && gradient_matrix[INT_INDEX] < gradient_matrix[INT_INDEX - matrix_dim.x - 1] &&
      get_color(angle_matrix[INT_INDEX - matrix_dim.x - 1] + M_PI_2) == 'R') {
      final_value = 0;
    } else if (index.x < matrix_dim.x-1 && index.y < matrix_dim.y-1 &&
      gradient_matrix[INT_INDEX] < gradient_matrix[INT_INDEX + matrix_dim.x + 1] &&
      get_color(angle_matrix[INT_INDEX + matrix_dim.x + 1] + M_PI_2) == 'R') {
      final_value = 0;
    }
  } else if (get_color(ANGLE) == 'B')  {
    // Horizontal gradient direction : Blue
    if (0 < index.x && gradient_matrix[INT_INDEX] < gradient_matrix[INT_INDEX - 1] &&
      get_color(angle_matrix[INT_INDEX - 1] + M_PI_2) == 'B') {
      final_value = 0;
    } else if (index.x < matrix_dim.x - 1 && gradient_matrix[INT_INDEX] < gradient_matrix[INT_INDEX + 1] &&
      get_color(angle_matrix[INT_INDEX + 1] + M_PI_2) == 'B') {
      final_value = 0;
    }
  }

  return final_value;
}

__device__ __host__ char histeresis_thresholding_init_core(int2 index, unsigned char *gradient_matrix, dim3 matrix_dim, int canny_min, int canny_max) {
  const int INT_INDEX = index.y*matrix_dim.x + index.x;
  char result;

  if (gradient_matrix[INT_INDEX] < canny_min) {
    result = 'D'; // Discarded
  } else if (canny_max < gradient_matrix[INT_INDEX]) {
    result = 'M'; // Marked
  } else {
    result = 'P'; // Pending
  }

  return result;
}

/**
 * Mark pending pixels connected to a marked pixel.
 */
__device__ __host__ void histeresis_thresholding_loop_core(int2 index, char *ht_matrix, dim3 matrix_dim, int2 read_limit, int *done) {
  const int INT_INDEX = index.y*matrix_dim.x + index.x;

  if (ht_matrix[INT_INDEX] == 'P') {
    // Pending pixel

    if (0 < index.x && 0 < index.y && ht_matrix[(index.y-1)*matrix_dim.x + index.x-1] == 'M') {
      // Top Left
      ht_matrix[INT_INDEX] = 'M';
      *done = 0;
    } else if (0 < index.y && ht_matrix[(index.y-1)*matrix_dim.x + index.x] == 'M') {
      // Top
      ht_matrix[INT_INDEX] = 'M';
      *done = 0;
    } else if (index.x < read_limit.x-1 && 0 < index.y && ht_matrix[(index.y-1)*matrix_dim.x + index.x+1] == 'M') {
      // Top Right
      ht_matrix[INT_INDEX] = 'M';
      *done = 0;
    } else if (0 < index.x && ht_matrix[index.y*matrix_dim.x + index.x-1] == 'M') {
      // Left
      ht_matrix[INT_INDEX] = 'M';
      *done = 0;
    } else if (index.x < read_limit.x-1 && ht_matrix[index.y*matrix_dim.x+ index.x+1] == 'M') {
      // Right
      ht_matrix[INT_INDEX] = 'M';
      *done = 0;
    } else if (0 < index.x && index.y < read_limit.y-1 && ht_matrix[(index.y+1)*matrix_dim.x + index.x-1] == 'M') {
      // Bottom Left
      ht_matrix[INT_INDEX] = 'M';
      *done = 0;
    } else if (index.y < read_limit.y-1 && ht_matrix[(index.y+1)*matrix_dim.x + index.x] == 'M') {
      // Bottom
      ht_matrix[INT_INDEX] = 'M';
      *done = 0;
    } else if (index.x < read_limit.x-1 && index.y < read_limit.y-1 && ht_matrix[(index.y+1)*matrix_dim.x + index.x+1] == 'M') {
      // Bottom Right
      ht_matrix[INT_INDEX] = 'M';
      *done = 0;
    }
  }
}

__device__ __host__ unsigned char histeresis_thresholding_end_core(int2 index, char *ht_matrix, dim3 matrix_dim) {
  const int INT_INDEX = index.y*matrix_dim.x + index.x;
  unsigned char result;

  if (ht_matrix[INT_INDEX] == 'P') {
    // All the still pending pixels are discarded
    ht_matrix[INT_INDEX] = 'D';
  }

  if (ht_matrix[INT_INDEX] == 'D') {
    // Final step, we set every discarded pixel to 0 in the gradient matrix
    result = 0;
  } else {
    result = 255;
  }

  return result;
}
