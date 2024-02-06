#ifndef EDGE_DETECTOR_HPP
#define EDGE_DETECTOR__HPP

#include "main.hpp"

void canny(std::string filename, int2 cutout_start_pixel, int2 *tracking_start_pixel, int nb_noise_reduction, int canny_min, int canny_max,
                int canny_sample_offset, ProcessingUnit processing_unit, int file_index = 1);

#endif // EDGE_DETECTOR_HPP
