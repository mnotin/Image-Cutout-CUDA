#ifndef CUTOUT_LAUNCHER_H
#define CUTOUT_LAUNCHER_H


namespace ProcessingUnitDevice {
  void cutout(unsigned char *h_rgb_image, unsigned char *h_edge_matrix, dim3 matrix_dim, int2 start_pixel, int threshold);
}


namespace ProcessingUnitHost {
  void cutout(unsigned char *rgb_image, unsigned char *edge_matrix, dim3 matrix_dim, int2 start_pixel, int threshold);
}


#endif // CUTOUT_LAUNCHER_H
