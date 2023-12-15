#include <stdio.h>

#include "tests.h"
#include "main.h"
#include "edge_detection.h"

void test_sobel_feldman() {
  int matrix_width = 64;
  int matrix_height = 32;
  unsigned char **buffer;

  buffer = (unsigned char **) malloc(sizeof(unsigned char *) * matrix_height);
  
  for (int i = 0; i < matrix_height; i++) {
    buffer[i] = (unsigned char *) malloc(sizeof(unsigned char) * matrix_width);
  }

  for (int i = 0; i < matrix_height; i++) {
    for (int j = 0; j < matrix_width; j++) {
      buffer[i][j] = 5;
    }
  }
  
  // Debug
  for (int i = 0; i < matrix_height; i++) {
    for (int j = 0; j < matrix_width; j++) {
      printf("%c ", buffer[i][j] + '0');
    }
    printf("\n");
  }

  sobel_feldman(buffer, matrix_width, matrix_height);
  
  /*printf("===\n");
  for (int i = 0; i < matrix_height; i++) {
    for (int j = 0; j < matrix_width; j++) {
      printf("%c ", buffer[i][j] + '0');
    }
    printf("\n");
  }*/
  
  for (int i = 0; i < matrix_height; i++) {
    free(buffer[i]);
  }
  free(buffer);
}
