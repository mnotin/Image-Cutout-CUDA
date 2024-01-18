#ifndef TESTS_HPP
#define TESTS_HPP

void test_sobel_feldman(char *filename, int start_pixel_x, int start_pixel_y);
void test_canny(char *filename, int start_pixel_x, int start_pixel_y, int canny_min, int canny_max, int canny_sample_offset);

#endif // TESTS_HPP
