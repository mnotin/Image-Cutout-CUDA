#ifndef MAIN_HPP
#define MAIN_HPP

#define MATRIX_SIZE_PER_BLOCK 32

enum EdgeDetection { SobelFeldman, Canny };
enum ProcessingUnit { Host, Device };

typedef struct Vec2 {
  int x = 0;
  int y = 0;
} Vec2;

typedef struct Dim {
  int width = 0;
  int height = 0;
} Dim;

void print_help();
void print_bad_usage();

#endif // MAIN_HPP
