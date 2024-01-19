#ifndef CUTOUT_H
#define CUTOUT_H

#include "main.hpp"

void cutout(unsigned char *h_rgb_image, unsigned char *h_edge_matrix, int matrix_width, int matrix_height, Vec2 start_pixel, int threshold);
__global__ void draw_edges_on_cutout_matrix(unsigned char *edge_matrix, unsigned char *cutout_matrix, int matrix_width, int matrix_height,Vec2 start_pixel, int threshold);
__global__ void cutout_algorithm(unsigned char *cutout_matrix, int matrix_width, int matrix_height, int *done);
__global__ void apply_cutout(unsigned char *cutout_matrix, unsigned char *output_image, int image_width, int image_height, Vec2 start_pixel);

#endif // CUTOUT_H
