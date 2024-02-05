#include <iostream>
#include <iomanip>

#include "edge_detector.hpp"
#include "launcher/utils/rgb_to_gray_launcher.hpp"
#include "launcher/utils/gaussian_blur_launcher.hpp"
#include "launcher/cutout_launcher.hpp"

#include "img.h"

#include "launcher/edge_detection/sobel_feldman_launcher.hpp"
#include "launcher/edge_detection/canny_launcher.hpp"

void canny(std::string filename, int2 cutout_start_pixel, int2 *tracking_start_pixel, int canny_min,
  int canny_max, int canny_sample_offset, ProcessingUnit processing_unit, int file_index
) {
  RGBImage *rgb_image = readPPM(filename.c_str());
  GrayImage *gray_image = createPGM(rgb_image->width, rgb_image->height);
  GrayImage *gradient_image = createPGM(rgb_image->width, rgb_image->height);
  float *angle_image = new float[rgb_image->width * rgb_image->height];
  RGBImage *edge_color_image = readPPM(filename.c_str());

  dim3 rgb_image_dim(rgb_image->width, rgb_image->height);
  dim3 gray_image_dim(gray_image->width, gray_image->height);
 
  if (rgb_image == nullptr) {
    std::cout << "Error reading the image" << std::endl;
    exit(EXIT_FAILURE);
  }
 
  // 1. First step, convert the picture into grayscale
  if (processing_unit == ProcessingUnit::Device) {
    ProcessingUnitDevice::rgb_to_gray(rgb_image, gray_image);
  } else if (processing_unit == ProcessingUnit::Host) {
    ProcessingUnitHost::rgb_to_gray(rgb_image, gray_image);
  }
    
  // 2. Second step, smooth the image using a Gaussian blur
  // to remove possible noise in the picture
  for (int i = 0; i < 5; i++) {
    if (processing_unit == ProcessingUnit::Device) {
      ProcessingUnitDevice::gaussian_blur(gray_image->data, gray_image_dim);
      cudaDeviceSynchronize();
    } else if (processing_unit == ProcessingUnit::Host) {
      ProcessingUnitHost::gaussian_blur(gray_image->data, gray_image_dim);
    }
  }
  writePGM("output/blurred_image_output.pgm", gray_image);
    
  // 3. Third step, apply the Sobel-Feldman operator to detect edges of shapes
  if (processing_unit == ProcessingUnit::Device) {
    ProcessingUnitDevice::sobel_feldman(gray_image->data, gradient_image->data, angle_image, gray_image_dim);
    ProcessingUnitDevice::generate_edge_color(gradient_image->data, angle_image, edge_color_image->data, rgb_image_dim);
  } else if (processing_unit == ProcessingUnit::Host) {
    ProcessingUnitHost::sobel_feldman(gray_image->data, gradient_image->data, angle_image, gray_image_dim);
    ProcessingUnitHost::generate_edge_color(gradient_image->data, angle_image, edge_color_image->data, rgb_image_dim);
  }
  writePGM("output/sf_gradient_output.pgm", gradient_image);
  writePPM("output/edge_color_output.ppm", edge_color_image);

  GrayImage *buffer_gray = createPGM(gradient_image->width, gradient_image->height);
  RGBImage *buffer_rgb = createPPM(gradient_image->width, gradient_image->height);

  if (canny_sample_offset == 0) {
    canny_sample_offset = 255;
  }

  for (int i = canny_min; i <= canny_max; i += canny_sample_offset) {
    memcpy(buffer_gray->data, gradient_image->data, sizeof(unsigned char) * gradient_image->width * gradient_image->height);
    
    if (processing_unit == ProcessingUnit::Device) {
      ProcessingUnitDevice::canny(buffer_gray->data, angle_image, gray_image_dim, i, canny_max);
    } else if (processing_unit == ProcessingUnit::Host) {
      ProcessingUnitHost::canny(buffer_gray->data, angle_image, gray_image_dim, i, canny_max);
    }

    // Create the name of the output file
    const char *prefix_gray = "output/canny_output";
    char number_gray[4] = "000";
    sprintf(number_gray, "%d", file_index);
    char filename_gray[strlen(prefix_gray) + 3 + 4 + 1]; // prefix + number + .ppm + \0
    bzero(filename_gray, strlen(prefix_gray) + 3 + 4 + 1);
    strcpy(filename_gray, prefix_gray);
    strcpy(filename_gray + strlen(prefix_gray), number_gray);
    strcpy(filename_gray + strlen(filename_gray), ".ppm");
    writePGM(filename_gray, buffer_gray);
    
    // 4. Last step, cutout the object selected by the user
    memcpy(buffer_rgb->data, rgb_image->data, sizeof(unsigned char) * gradient_image->width * gradient_image->height * 3);
    if (processing_unit == ProcessingUnit::Device) {
      ProcessingUnitDevice::cutout(buffer_rgb->data, buffer_gray->data, gray_image_dim, cutout_start_pixel, tracking_start_pixel, 0);
    } else if (processing_unit == ProcessingUnit::Host) {
      ProcessingUnitHost::cutout(buffer_rgb->data, buffer_gray->data, gray_image_dim, cutout_start_pixel, tracking_start_pixel, 0);
    }

    std::string prefix_rgb("output/frame");
    std::stringstream string_stream;
    std::string number;
    string_stream << std::setw(5) << std::setfill('0') << file_index;
    string_stream >> number;

    std::string filename_rgb; // prefix + number + .ppm + \0
    filename_rgb.append(prefix_rgb);
    filename_rgb.append(number);
    filename_rgb.append(".ppm");
    writePPM(filename_rgb.c_str(), buffer_rgb);
  
    file_index += 1;
  }

  destroyPGM(buffer_gray);
  destroyPPM(buffer_rgb);

  destroyPPM(rgb_image);
  destroyPGM(gray_image);  
  destroyPGM(gradient_image);  
  destroyPPM(edge_color_image);  
  delete [] angle_image; 
}
