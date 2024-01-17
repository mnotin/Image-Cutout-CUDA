#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "main.h"
#include "utils.h"
#include "tests.h"


int main(int argc, char **argv) {
  char *filename;
  int start_pixel_x = 0;
  int start_pixel_y = 0;
  char edge_detection = 'c';
  char processing_unit = 'd';
  int canny_min_val = 50;
  int canny_max_val = 100;
  int canny_sample_offset = 0; // Zero: no sample; non-zero value: sample

  if (argc == 2 && strcmp(argv[1], "--help") == 0) {
    print_help();

    return 0;
  } else {
    int i;
    int bad_usage = 0;
    int filename_found = 0;

    for (i = 1; i < argc && !bad_usage; i++) {
      if (strcmp(argv[i], "--start-pixel") == 0) {
        start_pixel_x = atoi(argv[i+1]);
        start_pixel_y = atoi(argv[i+2]);
        i += 2;
      } else if (strcmp(argv[i], "--edge-detection") == 0) {
        if (strcmp(argv[i+1], "sobel") == 0) {
          edge_detection = 's';
          i += 1;
        } else if (strcmp(argv[i+1], "canny") == 0) {
          edge_detection = 'c';
          i += 1;
        } else {
          bad_usage = 1;
        }
      } else if (strcmp(argv[i], "--canny-thresholds") == 0) {
        canny_min_val = atoi(argv[i+1]);
        canny_max_val = atoi(argv[i+2]);
        i += 2;

        if (canny_min_val < 0 || 255 < canny_max_val || canny_max_val < canny_min_val) {
          bad_usage = 1;
        }
      } else if (strcmp(argv[i], "--processing-unit") == 0) {
        if (strcmp(argv[i+1], "host") == 0) {
          processing_unit = 'h';
          i += 1;
        } else if (strcmp(argv[i+1], "device") == 0) {
          processing_unit = 'd';
          i += 1;
        } else {
          bad_usage = 1;
        }
      } else if (strcmp(argv[i], "--canny-sampling-offset") == 0) {
        canny_sample_offset = atoi(argv[i+1]);
        i += 1;
      } else {
        // This option did not match any possible one
        if (i != argc-1) {
          // Not the filename
          bad_usage = 1;
        } else {
          filename_found = 1;
        }
      }
    }
  
    if (argc == 1 || i == argc && filename_found == 0 || bad_usage) {
      // Filename is missing or bad usage
      print_bad_usage();
      exit(EXIT_FAILURE);
    } else {
      filename = argv[argc-1];
    }
  }

  test_canny(filename, start_pixel_x, start_pixel_y, canny_min_val, canny_max_val, canny_sample_offset);

  printf(" === \n");
  cudaDeviceSynchronize();
  cudaError_t error = cudaPeekAtLastError();
  printf("Error: %s\n", cudaGetErrorString(error));

  return 0;
}

void print_help() {
  printf("Usage: ./main [OPTION] file\n");
  printf("\t--start-pixel <x> <y>\t\t\tPixel coordinates where the cutout algorithm should start. (default: 0 0)\n");

  printf("\t--edge-detection <method>\t\tSpecify the method to use to process edge detection. (default: canny)\n");
  printf("\t\t\t\t\t\tPermissible methods are 'sobel' and 'canny'.\n");
  printf("\t--canny-thresholds <min> <max>\t\tSpecify the thresholds that have to be used by the Canny edge detector (default: 50 100)\n");
  printf("\t\t\t\t\t\tPermissible values are integer between 0 and 255.\n");
  printf("\t--processing-unit <processing-unit>\tSpecify where the cutout process has to be executed. (default: device)\n");
  printf("\t\t\t\t\t\tPermissible processing units are 'host' (CPU) and 'device' (GPU).\n");
  printf("\t--canny-sampling-offset <offset>\tSpecify that canny should produce multiple outputs, " \
  "starting from the minimum value threshold up to the maximum value\n");
  printf("\t\t\t\t\t\twith an offset of 'offset' between each sample.\n");

  printf("\t--help\t\t\t\t\tDisplay this help and exit.\n");
}

void print_bad_usage() {
  printf("Usage: ./main [OPTION] file\n");
  printf("Try './main --help' for more information.\n");
}
