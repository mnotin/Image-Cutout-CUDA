#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "main.h"
#include "edge_detection.h"

/**
 * Applies discrete convolution over a matrix using a given kernel.
 * This kernel should be called using appropriate number of grids, blocks and threads to match the resolution of the image.
 **/
__global__ void convolution(unsigned char *matrix, int matrix_width, int matrix_height, int *kernel, int kernel_size) {
  int globalIdxX = threadIdx.x + (blockIdx.x * blockDim.x);
  int globalIdxY = threadIdx.y + (blockIdx.y * blockDim.y);
  int localIdxX = globalIdxX % MATRIX_SIZE_PER_BLOCK;
  int localIdxY = globalIdxY % MATRIX_SIZE_PER_BLOCK;

  int current_matrix_index = globalIdxY*matrix_width + globalIdxX;
  int current_shared_matrix_index = localIdxY*MATRIX_SIZE_PER_BLOCK + localIdxX;

  __shared__ unsigned char shared_matrix[MATRIX_SIZE_PER_BLOCK*MATRIX_SIZE_PER_BLOCK];

  unsigned char convolution_result = 0;

  // Todo ...

  matrix[current_matrix_index] = convolution_result;
}

int main(int argc, char **argv) {

  return 0;
}
