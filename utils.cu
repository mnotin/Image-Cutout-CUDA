#include <cuda_runtime.h>

#include "utils.h"

__global__ copy_matrix(unsigned char *destination_matrix, unsigned char *source_matrix, int matrix_width) {
  int indexX = threadIdx.x + (blockIdx.x * blockDim.x);
  int indexY = threadIdx.y + (blockIdx.y * blockDim.y);

  destination_matrix[indexY * matrix_width + indexX] = source_matrix[indexY * matrix_width + indexX];
}
