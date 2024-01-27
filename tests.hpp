#ifndef TESTS_HPP
#define TESTS_HPP

#include "main.hpp"

void test_canny(std::string filename, int2 cutout_start_pixel, int2 *tracking_start_pixel, int canny_min, int canny_max,
                int canny_sample_offset, ProcessingUnit processing_unit, int file_index = 1);

#endif // TESTS_HPP
