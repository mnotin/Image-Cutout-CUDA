#ifndef GAUSSIAN_BLUR_HPP
#define GAUSSIAN_BLUR_HPP

#include "../main.hpp"

namespace ProcessingUnitDevice {
  void gaussian_blur(unsigned char *h_matrix, dim3 matrix_dim);
}


namespace ProcessingUnitHost {
  void gaussian_blur(unsigned char *matrix, dim3 matrix_dim);
}

#endif // GAUSSIAN_BLUR_HPP
