#ifndef UTILS_H
#define UTILS_H

#include "img.h"

__global__ void rgb_to_gray_kernel(unsigned char *rgb_image, unsigned char *gray_image, int image_width, int image_height);
void rgb_to_gray(RGBImage *h_rgb_image, GrayImage *h_gray_image);
void gaussian_blur(unsigned char *h_matrix, int matrix_width, int matrix_height);

#endif // UTILS_H
