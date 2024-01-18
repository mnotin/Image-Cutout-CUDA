#ifndef TESTS_HPP
#define TESTS_HPP

#include "main.hpp"

void test_sobel_feldman(char *filename, int start_pixel_x, int start_pixel_y, ProcessingUnit processing_unit);
void test_canny(char *filename, int start_pixel_x, int start_pixel_y, int canny_min,
  int canny_max, int canny_sample_offset, ProcessingUnit processing_unit);

#endif // TESTS_HPP
