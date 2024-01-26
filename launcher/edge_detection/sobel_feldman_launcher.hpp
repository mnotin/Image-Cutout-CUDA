#ifndef SOBEL_FELDMAN_LAUNCHER_HPP
#define SOBEL_FELDMAN_LAUNCHER_HPP


namespace ProcessingUnitDevice {
  void sobel_feldman(unsigned char *h_input_matrix, unsigned char *h_gradient_matrix, float *h_angle_matrix, dim3 matrix_dim);
  void generate_edge_color(unsigned char *h_gradient_matrix, float *h_angle_matrix, unsigned char *h_output_image, dim3 matrix_dim);
}


namespace ProcessingUnitHost {
  void sobel_feldman(unsigned char *input_matrix, unsigned char *gradient_matrix, float *angle_matrix, dim3 matrix_dim);
  void generate_edge_color(unsigned char *gradient_matrix, float *angle_matrix, unsigned char *output_image, dim3 matrix_dim);
}


#endif // SOBEL_FELDMAN_LAUNCHER_HPP
