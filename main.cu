#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "main.h"
#include "utils.h"
#include "tests.h"

/**
 * Applies discrete convolution over a matrix using a given kernel.
 * This kernel should be called using appropriate number of grids, blocks and threads to match the resolution of the image.
 **/
__global__ void convolution(unsigned char *input_matrix, unsigned char *output_matrix, int matrix_width, int matrix_height, float *kernel, int kernel_size) {
  int globalIdxX = threadIdx.x + (blockIdx.x * blockDim.x);
  int globalIdxY = threadIdx.y + (blockIdx.y * blockDim.y);
  int localIdxX = threadIdx.x;
  int localIdxY = threadIdx.y;
  
  int current_matrix_index = globalIdxY*matrix_width + globalIdxX;
  int current_shared_matrix_index = MATRIX_SIZE_PER_BLOCK+2+1+ localIdxY*(MATRIX_SIZE_PER_BLOCK+2) + localIdxX;

  __shared__ unsigned char shared_matrix[(MATRIX_SIZE_PER_BLOCK+2)*(MATRIX_SIZE_PER_BLOCK+2)];

  /*
   * x x x x x x MATRIX_SIZE_PER_BLOCK + 2
   * x o o o o x
   * x o o o o x
   * x o o o o x
   * x o o o o x
   * x x x x x x
   */
  shared_matrix[current_shared_matrix_index] = input_matrix[current_matrix_index];
  if (localIdxX == 0 && localIdxY == 0) {
    // Fill the edges
    for (int i = 0; i < MATRIX_SIZE_PER_BLOCK+2; i++) {
      shared_matrix[i] = globalIdxY == 0 || (globalIdxX == 0 && i == 0) ? 0 : input_matrix[(globalIdxY-1)*matrix_width + globalIdxX + i - 1]; // First line
      shared_matrix[(MATRIX_SIZE_PER_BLOCK+2)*(MATRIX_SIZE_PER_BLOCK+1)+i] = globalIdxY+MATRIX_SIZE_PER_BLOCK == matrix_height ? 0 :
        input_matrix[(globalIdxY+MATRIX_SIZE_PER_BLOCK)*matrix_width + globalIdxX + i - 1]; // Last line
    }

    for (int i = 0; i < MATRIX_SIZE_PER_BLOCK; i++) {
      shared_matrix[MATRIX_SIZE_PER_BLOCK+2 + i*(MATRIX_SIZE_PER_BLOCK+2)] = globalIdxX == 0 ? 0 : input_matrix[(globalIdxY+i)*matrix_width + globalIdxX - 1]; // Left side
      shared_matrix[MATRIX_SIZE_PER_BLOCK+2 + (i+1)*(MATRIX_SIZE_PER_BLOCK+2) - 1] = globalIdxX == matrix_width-1 ? 0 :
        input_matrix[(globalIdxY+i)*matrix_width + globalIdxX+MATRIX_SIZE_PER_BLOCK]; // Right side
    }
  }
  __syncthreads();

  float convolution_result = 0;

  for (int i = 0; i < kernel_size; i++) {
    for (int j = 0; j < kernel_size; j++) {
      int vertical_offset = ((localIdxY + i) - (int)floor(kernel_size/2.0));
      int horizontal_offset = (localIdxX + j) - (int)floor(kernel_size/2.0);
      int tmp_index = vertical_offset*(MATRIX_SIZE_PER_BLOCK+2) + horizontal_offset;
       
      convolution_result += shared_matrix[MATRIX_SIZE_PER_BLOCK+2+1 + tmp_index] * kernel[i*kernel_size + j];
    }
  }  
  
  output_matrix[current_matrix_index] = abs(convolution_result);
}

int main(int argc, char **argv) {
  if (argc != 4) {
    printf("Please provide the name of the file that has to be processed.\n");
    printf("Usage: ./binary filename.ppm start_pixel_x start_pixel_y\n");
    exit(EXIT_FAILURE);
  }

  char *filename = argv[1];
  int start_pixel_x = atoi(argv[2]);
  int start_pixel_y = atoi(argv[3]);

  test_sobel_feldman(filename, start_pixel_x, start_pixel_y);

  printf(" === \n");
  cudaDeviceSynchronize();
  cudaError_t error = cudaPeekAtLastError();
  printf("Error: %s\n", cudaGetErrorString(error));

  return 0;
}

