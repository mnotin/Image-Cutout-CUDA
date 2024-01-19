#ifndef TESTS_HPP
#define TESTS_HPP

#include "main.hpp"

void test_sobel_feldman(char *filename, Vec2 start_pixel, ProcessingUnit processing_unit);
void test_canny(char *filename, Vec2 start_pixel, int canny_min, int canny_max, int canny_sample_offset, ProcessingUnit processing_unit);

#endif // TESTS_HPP
