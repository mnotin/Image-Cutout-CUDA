#ifndef GAUSSIAN_BLUR_HPP
#define GAUSSIAN_BLUR_HPP

#include "../main.hpp"

namespace ProcessingUnitDevice {
  void gaussian_blur(unsigned char *h_matrix, int matrix_width, int matrix_height);
}


namespace ProcessingUnitHost {
  void gaussian_blur(unsigned char *matrix, int matrix_width, int matrix_height);
}

#endif // GAUSSIAN_BLUR_HPP
