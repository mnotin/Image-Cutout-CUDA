#include <math.h>
#include <iostream>

#include "cutout_kernel.hpp"

#include "../core/cutout_core.hpp"
#include "../main.hpp"

/**
 * First step of the cutout process.
 * Each gradient pixel with a value above the threshold is considered a border.
 **/
__global__ void draw_edges_on_cutout_matrix_kernel(unsigned char *edge_matrix, 
                                                   char *micro_cutout_matrix, dim3 matrix_dim,
                                                   int2 cutout_start_pixel, int2 tracking_start_pixel,
                                                   int threshold, char *macro_cutout_matrix
) {
  int2 global_index = make_int2(threadIdx.x + (blockIdx.x * blockDim.x), threadIdx.y + (blockIdx.y * blockDim.y));
  int2 local_index = make_int2(threadIdx.x, threadIdx.y);
  __shared__ bool block_contains_edge;
  __shared__ bool block_contains_start_pixel;
  
  if (local_index.x == 0 && local_index.y == 0) {
    block_contains_edge = false;
    block_contains_start_pixel = false;
  }
  __syncthreads();
  
  if (global_index.x < matrix_dim.x && global_index.y < matrix_dim.y) {
    char result = draw_edges_on_cutout_matrix_core(global_index, edge_matrix, matrix_dim, cutout_start_pixel, tracking_start_pixel, threshold);
    micro_cutout_matrix[global_index.y*matrix_dim.x + global_index.x] = result;

    if (result == 'B') {
      // This block contains at least one border
      block_contains_edge = true;
    } else if (result == 'M') {
      block_contains_start_pixel = true;
    }
  }

  __syncthreads();

  if (local_index.x == 0 && local_index.y == 0) {
    if (block_contains_edge) {
      macro_cutout_matrix[blockIdx.y*gridDim.x + blockIdx.x] = 'B';
    } else if (block_contains_start_pixel) {
      macro_cutout_matrix[blockIdx.y*gridDim.x + blockIdx.x] = 'M';
    } else {
      macro_cutout_matrix[blockIdx.y*gridDim.x + blockIdx.x] = 'D';
    }
  }
}

/**
 * Main part of the cutout process.
 * Loops over a cutout matrix from the start pixel to fill the shape it is in.
 **/
__global__ void cutout_algorithm_kernel(char *cutout_matrix, dim3 matrix_dim,
                                        int *done, char *looking_char, char spread_char,
                                        int2 *tracking_top_left, int2 *tracking_bottom_right
) {
  int2 global_index = make_int2(threadIdx.x + (blockIdx.x * blockDim.x), threadIdx.y + (blockIdx.y * blockDim.y));
  int2 local_index = make_int2(threadIdx.x, threadIdx.y);
  char result_char = '\0';

  __shared__ int2 shared_tracking_top_left;
  __shared__ int2 shared_tracking_bottom_right;
  __shared__ bool right_block;
  __shared__ bool bottom_block;
  if (local_index.x == 0 && local_index.y == 0) {
    right_block = false;
    bottom_block = false;

    if (tracking_top_left != nullptr && tracking_bottom_right != nullptr) {
      shared_tracking_top_left = *tracking_top_left;
      shared_tracking_bottom_right = *tracking_bottom_right;
    }
  }
  __syncthreads();
  if (matrix_dim.x <= global_index.x) {
    right_block = true;
  }
  if (matrix_dim.y <= global_index.y) {
    bottom_block = true;
  }
  __syncthreads();
  
  dim3 shared_matrix_dim(MATRIX_SIZE_PER_BLOCK, MATRIX_SIZE_PER_BLOCK);
  int2 read_limit = make_int2(shared_matrix_dim.x, shared_matrix_dim.y);
  if (right_block) {
    read_limit.x = matrix_dim.x % MATRIX_SIZE_PER_BLOCK;
  }
  if (bottom_block) {
    read_limit.y = matrix_dim.y % MATRIX_SIZE_PER_BLOCK;
  }

  __shared__ int shared_done;
  __shared__ char shared_cutout_matrix[MATRIX_SIZE_PER_BLOCK*MATRIX_SIZE_PER_BLOCK];
  
  if (global_index.x < matrix_dim.x && global_index.y < matrix_dim.y) { 
    shared_cutout_matrix[local_index.y*MATRIX_SIZE_PER_BLOCK + local_index.x] = cutout_matrix[global_index.y*matrix_dim.x + global_index.x];
  }

  if (local_index.x == 0 && local_index.y == 0) {
    shared_done = 0;
  }

  __syncthreads();
  
  while (shared_done == 0) {
    __syncthreads();

    if (local_index.x == 0 && local_index.y == 0) {
      shared_done = 1; // Let's assume the process is finished
    }
    
    __syncthreads();
    
    if (global_index.x < matrix_dim.x && global_index.y < matrix_dim.y) {
      result_char = cutout_algorithm_core(local_index, shared_cutout_matrix, shared_matrix_dim, read_limit, &shared_done, looking_char, spread_char);
    }
  
    if (result_char == 'T') {
      if (global_index.x < shared_tracking_top_left.x) {
        shared_tracking_top_left.x = global_index.x;
      } else if (shared_tracking_bottom_right.x < global_index.x) {
        shared_tracking_bottom_right.x = global_index.x;
      } else if (global_index.y < shared_tracking_top_left.y) {
        shared_tracking_top_left.y = global_index.y;
      } else if (shared_tracking_bottom_right.y < global_index.y) {
        shared_tracking_bottom_right.y = global_index.y;
      }
    }

    __syncthreads();

    if (local_index.x == 0 && local_index.y == 0 && shared_done == 0) {
      // At least one block had to update a pixel so we will need
      // to rerun them all at least once
      *done = 0;
    }
  }

  __syncthreads();
 
  // Write the result back to global memory
  if (global_index.x < matrix_dim.x && global_index.y < matrix_dim.y) {
    cutout_matrix[global_index.y*matrix_dim.x + global_index.x] =
      shared_cutout_matrix[local_index.y*MATRIX_SIZE_PER_BLOCK + local_index.x];
  }
  
  if (local_index.x == 0 && local_index.y == 0) {
    if (tracking_top_left != nullptr && tracking_bottom_right != nullptr) {
      if (shared_tracking_top_left.x < tracking_top_left->x) {
        tracking_top_left->x = shared_tracking_top_left.x;
      } else if (tracking_bottom_right->x < shared_tracking_bottom_right.x) {
        tracking_bottom_right->x = shared_tracking_bottom_right.x;
      } else if (shared_tracking_top_left.y < tracking_top_left->y) {
        tracking_top_left->y = shared_tracking_top_left.y;
      } else if (tracking_bottom_right->y < shared_tracking_bottom_right.y) {
        tracking_bottom_right->y = shared_tracking_bottom_right.y;
      }
    }
  }
}

__global__ void ProcessingUnitDevice::Cutout::transfer_edges_between_blocks_kernel(char *cutout_matrix,
                                                                                   dim3 matrix_dim, int *done,
                                                                                   char *looking_pixels, char spread_pixel
) {
  int2 global_index = make_int2(threadIdx.x + (blockIdx.x * blockDim.x), threadIdx.y + (blockIdx.y * blockDim.y));
  int2 local_index = make_int2(threadIdx.x, threadIdx.y);
  int2 read_limit = make_int2(matrix_dim.x, matrix_dim.y);
  
  __shared__ bool right_block;
  __shared__ bool bottom_block;
  if (local_index.x == 0 && local_index.y == 0) {
    right_block = false;
    bottom_block = false;
  }
  __syncthreads();
  if (matrix_dim.x <= global_index.x) {
    right_block = true;
  }
  if (matrix_dim.y <= global_index.y) {
    bottom_block = true;
  }
  __syncthreads();
  
  if (right_block) {
    read_limit.x = matrix_dim.x % MATRIX_SIZE_PER_BLOCK;
  }
  if (bottom_block) {
    read_limit.y = matrix_dim.y % MATRIX_SIZE_PER_BLOCK;
  }

  if (matrix_dim.x <= global_index.x || matrix_dim.y <= global_index.y) {
    return;
  }
  
  if (local_index.y == 0 && 0 < global_index.y ||
      local_index.y == MATRIX_SIZE_PER_BLOCK-1 && global_index.y < matrix_dim.y-1 ||
      local_index.x == 0 && 0 < global_index.x ||
      local_index.x == MATRIX_SIZE_PER_BLOCK-1 && global_index.x < matrix_dim.x-1) {
      cutout_algorithm_core(global_index, cutout_matrix, matrix_dim, read_limit, done, looking_pixels, spread_pixel);
  }
}

__global__ void apply_macro_to_micro_cutout_matrix_kernel(char *macro_cutout_matrix, char *micro_cutout_matrix, dim3 macro_matrix_dim, dim3 micro_matrix_dim) {
  int2 global_index = make_int2(threadIdx.x + (blockIdx.x * blockDim.x), threadIdx.y + (blockIdx.y * blockDim.y));
  int2 local_index = make_int2(threadIdx.x, threadIdx.y);

  __shared__ char macro_matrix_content;

  if (local_index.x == 0 && local_index.y == 0) {
    macro_matrix_content = macro_cutout_matrix[blockIdx.y*macro_matrix_dim.x + blockIdx.x];
  }

  __syncthreads();

  if (global_index.x < micro_matrix_dim.x && global_index.y < micro_matrix_dim.y) {
    if (macro_matrix_content == 'M') {
      micro_cutout_matrix[global_index.y*micro_matrix_dim.x + global_index.x] = 'M';
    }
  }
}

__global__ void apply_cutout_kernel(char *cutout_matrix, unsigned char *output_image, dim3 image_dim, int2 start_pixel) { 
  int2 global_index = make_int2(threadIdx.x + (blockIdx.x * blockDim.x), threadIdx.y + (blockIdx.y * blockDim.y));
 
  if (global_index.x < image_dim.x && global_index.y < image_dim.y) {
    apply_cutout_core(global_index, cutout_matrix, output_image, image_dim, start_pixel);
  }
}
