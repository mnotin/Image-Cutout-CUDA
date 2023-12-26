#ifndef EDGE_DETECTION_H
#define EDGE_DETECTION_H

#include "img.h"

__global__ void rgb_to_gray_kernel(unsigned char *rgbImage, unsigned char *grayImage, int image_width, int image_height);
void rgb_to_gray(RGBImage *h_rgbImage, GrayImage *h_grayImage);
void gaussian_blur(unsigned char *h_matrix, int matrix_width, int matrix_height);
void sobel_feldman(unsigned char *matrix, int matrix_width, int matrix_height);
__global__ void global_gradient(unsigned char *matrix, unsigned char *horizontal_edges, unsigned char *vertical_edges, int matrix_width, int matrix_height);
void cutout(unsigned char *h_matrix, int matrix_width, int matrix_height, int start_pixel_x, int start_pixel_y);
__global__ void cutout_algorithm(int start_x, int start_y, int done, unsigned char *input_matrix, unsigned char *cutout_matrix, int matrix_width, int matrix_height);

#endif // EDGE_DETECTION_H
