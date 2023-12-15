#include <stdio.h>

#include "tests.h"
#include "main.h"
#include "edge_detection.h"
#include "img.h"

void test_sobel_feldman() {
  GrayImage* grayImage = readPGM("circle_sample.pgm");

  if (grayImage == NULL) {
    printf("Error reading the image\n");
    exit(EXIT_FAILURE);
  } 
  
  sobel_feldman(grayImage->data, grayImage->width, grayImage->height);
  
  writePGM("sobel_feldman_output.pgm", grayImage);
  destroyPGM(grayImage);  
}
