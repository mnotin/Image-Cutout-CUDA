#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "main.h"
#include "edge_detection.h"

#include "tests.h"

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
  shared_matrix[current_shared_matrix_index] = matrix[current_matrix_index];
  __syncthreads();

  int convolution_result = 0;

  if (0 < localIdxX && localIdxX < MATRIX_SIZE_PER_BLOCK-1 && 0 < localIdxY && localIdxY < MATRIX_SIZE_PER_BLOCK-1) {
    for (int i = 0; i < kernel_size; i++) {
      for (int j = 0; j < kernel_size; j++) {
        int vertical_offset = ((localIdxY + i) - (int)floor(kernel_size/2.0));
        int horizontal_offset = (localIdxX + j) - (int)floor(kernel_size/2.0);
        int tmp_index = vertical_offset*MATRIX_SIZE_PER_BLOCK + horizontal_offset;

        convolution_result += shared_matrix[tmp_index] * kernel[i*kernel_size + j];
      }
    }  
  }
  
  matrix[current_matrix_index] = convolution_result;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Please provide the name of the file that has to be processed.\n");
    printf("Usage: ./binary filename.pgm\n");
    exit(EXIT_FAILURE);
  }

  char *filename = argv[1];

  test_sobel_feldman(filename);

  printf(" === \n");
  cudaDeviceSynchronize();
  cudaError_t error = cudaPeekAtLastError();
  printf("Error: %s\n", cudaGetErrorString(error));

  return 0;
}

