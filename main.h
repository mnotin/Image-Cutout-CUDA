#ifndef MAIN_H
#define MAIN_H

#define MATRIX_SIZE_PER_BLOCK 32

__global__ void convolution(unsigned char* input_matrix, unsigned char *output_matrix, int matrix_width, int matrix_height, float *kernel, int kernel_size);

#endif // MAIN_H
