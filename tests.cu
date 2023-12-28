#include <stdio.h>

#include "tests.h"
#include "main.h"
#include "utils.h"
#include "img.h"
#include "cutout.h"

#include "edge_detection/sobel_feldman.h"
#include "edge_detection/canny.h"

void test_sobel_feldman(char *filename, int start_pixel_x, int start_pixel_y) {
  RGBImage *rgb_image = readPPM(filename);
  GrayImage *gray_image = createPGM(rgb_image->width, rgb_image->height);
  GrayImage *gradient_image = createPGM(rgb_image->width, rgb_image->height);
  float *angle_image = (float *) malloc(rgb_image->width * rgb_image->height * sizeof(float));
  RGBImage *edge_color_image = readPPM(filename);

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
  sobel_feldman(gray_image->data, gradient_image->data, angle_image, gray_image->width, gray_image->height);
  writePGM("sf_gradient_output.pgm", gradient_image);

  generate_edge_color(gradient_image->data, angle_image, edge_color_image->data, edge_color_image->width, edge_color_image->height);
  writePPM("edge_color_output.ppm", edge_color_image);

  canny(gradient_image->data, angle_image, gray_image->width, gray_image->height); 
  writePGM("canny_output.pgm", gradient_image);

  // 4. Last step, cutout the object selected by the user
  cutout(rgb_image->data, gradient_image->data, gray_image->width, gray_image->height, start_pixel_x, start_pixel_y);
  
  writePPM("cutout_output.ppm", rgb_image);

  destroyPPM(rgb_image);
  destroyPGM(gray_image);  
  destroyPGM(gradient_image);  
  destroyPPM(edge_color_image);  
  free(angle_image);  
}
