#ifndef EDGE_DETECTION_H
#define EDGE_DETECTION_H

void sobel_feldman(unsigned char *matrix, int matrix_width, int matrix_height);
__global__ void global_gradient(unsigned char *matrix, unsigned char *horizontal_edges, unsigned char *vertical_edges, int matrix_width, int matrix_height);

#endif // EDGE_DETECTION_H
