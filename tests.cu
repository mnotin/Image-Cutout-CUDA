#include <stdio.h>

#include "tests.h"
#include "main.h"
#include "edge_detection.h"
#include "img.h"

void test_sobel_feldman(char *filename) {
  RGBImage *rgb_image = readPPM(filename);
  GrayImage *gray_image = createPGM(rgb_image->width, rgb_image->height);

  if (rgb_image == NULL) {
    printf("Error reading the image\n");
    exit(EXIT_FAILURE);
  }
  
  // 1. First step, convert the picture into grayscale
  rgb_to_gray(rgb_image, gray_image);

  // 2. Second step, smooth the image using a Gaussian blur
  // to remove possible noise in the picture
  for (int i = 0; i < 5; i++) {
    gaussian_blur(gray_image->data, gray_image->width, gray_image->height);
    cudaDeviceSynchronize();
  }

  // 3. Third step, apply the Sobel-Feldman operator to detect edges of shapes
  sobel_feldman(gray_image->data, gray_image->width, gray_image->height);

  // 4. Last step, cutout the object selected by the user
  cutout(rgb_image->data, gray_image->data, gray_image->width, gray_image->height, 0, 0);
  
  writePPM("sobel_feldman_output.ppm", rgb_image);

  destroyPPM(rgb_image);
  destroyPGM(gray_image);  
}
