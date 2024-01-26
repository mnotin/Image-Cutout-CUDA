#ifndef GAUSSIAN_BLUR_LAUNCHER_HPP
#define GAUSSIAN_BLUR_LAUNCHER_HPP


namespace ProcessingUnitDevice {
  void gaussian_blur(unsigned char *h_matrix, dim3 matrix_dim);
}


namespace ProcessingUnitHost {
  void gaussian_blur(unsigned char *matrix, dim3 matrix_dim);
}

#endif // GAUSSIAN_BLUR_LAUNCHER_HPP
