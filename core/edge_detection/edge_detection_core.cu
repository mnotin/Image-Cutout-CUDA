#include <math.h>

#include "edge_detection_core.hpp"

/**
 * Returns the first letter of the color associated to a specific angle value. 
 **/
__device__ __host__ char get_color(float angle) {
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
