#ifndef CANNY_LAUNCHER_HPP
#define CANNY_LAUNCHER_HPP


namespace ProcessingUnitDevice {
  void canny(unsigned char *h_gradient_matrix, float *h_angle_matrix, dim3 matrix_dim, int canny_min, int canny_max);
}


namespace ProcessingUnitHost {
  void canny(unsigned char *gradient_matrix, float *angle_matrix, dim3 matrix_dim, int canny_min, int canny_max);
}


#endif // CANNY_LAUNCHER_HPP
