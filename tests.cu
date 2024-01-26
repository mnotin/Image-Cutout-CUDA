#include <iostream>

#include "tests.hpp"
#include "launcher/utils/rgb_to_gray_launcher.hpp"
#include "launcher/utils/gaussian_blur_launcher.hpp"
#include "launcher/cutout_launcher.hpp"

#include "img.h"

#include "launcher/edge_detection/sobel_feldman_launcher.hpp"
#include "launcher/edge_detection/canny_launcher.hpp"

void test_sobel_feldman(std::string filename, int2 start_pixel, ProcessingUnit processing_unit) {
  RGBImage *rgb_image = readPPM(filename.c_str());
  GrayImage *gray_image = createPGM(rgb_image->width, rgb_image->height);
  GrayImage *gradient_image = createPGM(rgb_image->width, rgb_image->height);
  float *angle_image = new float[rgb_image->width * rgb_image->height];
  RGBImage *edge_color_image = readPPM(filename.c_str());

  dim3 rgb_image_dim(rgb_image->width, rgb_image->height);
  dim3 gray_image_dim(gray_image->width, gray_image->height);

  if (rgb_image == NULL) {
    std::cout << "Error reading the image" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (processing_unit == ProcessingUnit::Device) {
    // GPU
    // 1. First step, convert the picture into grayscale
    ProcessingUnitDevice::rgb_to_gray(rgb_image, gray_image);

    // 2. Second step, smooth the image using a Gaussian blur
    // to remove possible noise in the picture
    for (int i = 0; i < 5; i++) {
      ProcessingUnitDevice::gaussian_blur(gray_image->data, gray_image_dim);
      cudaDeviceSynchronize();
    }

    // 3. Third step, apply the Sobel-Feldman operator to detect edges of shapes
    ProcessingUnitDevice::sobel_feldman(gray_image->data, gradient_image->data, angle_image, gray_image_dim);
    writePGM("output/sf_gradient_output.pgm", gradient_image);

    ProcessingUnitDevice::generate_edge_color(gradient_image->data, angle_image, edge_color_image->data, rgb_image_dim);
    writePPM("output/edge_color_output.ppm", edge_color_image);

    // 4. Last step, cutout the object selected by the user
    ProcessingUnitDevice::cutout(rgb_image->data, gradient_image->data, gray_image_dim, start_pixel, 0);
  } else if (processing_unit == ProcessingUnit::Host) {
    // CPU
    // 1. First step, convert the picture into grayscale
    ProcessingUnitHost::rgb_to_gray(rgb_image, gray_image);
    
    // 2. Second step, smooth the image using a Gaussian blur
    // to remove possible noise in the picture
    for (int i = 0; i < 5; i++) {
      ProcessingUnitHost::gaussian_blur(gray_image->data, gray_image_dim);
    }
    
    // 3. Third step, apply the Sobel-Feldman operator to detect edges of shapes
    ProcessingUnitHost::sobel_feldman(gray_image->data, gradient_image->data, angle_image, gray_image_dim);
    writePGM("output/sf_gradient_output.pgm", gradient_image);
    
    ProcessingUnitHost::generate_edge_color(gradient_image->data, angle_image, edge_color_image->data, rgb_image_dim);
    writePPM("output/edge_color_output.ppm", edge_color_image);
    
    // 4. Last step, cutout the object selected by the user
    ProcessingUnitHost::cutout(rgb_image->data, gradient_image->data, gray_image_dim, start_pixel, 0);
  }
  
  writePPM("output/cutout_output.ppm", rgb_image);

  destroyPPM(rgb_image);
  destroyPGM(gray_image);  
  destroyPGM(gradient_image);  
  destroyPPM(edge_color_image);  
  delete [] angle_image;
}

void test_canny(std::string filename, int2 start_pixel, int canny_min,
  int canny_max, int canny_sample_offset, ProcessingUnit processing_unit, int file_index
) {
  RGBImage *rgb_image = readPPM(filename.c_str());
  GrayImage *gray_image = createPGM(rgb_image->width, rgb_image->height);
  GrayImage *gradient_image = createPGM(rgb_image->width, rgb_image->height);
  float *angle_image = new float[rgb_image->width * rgb_image->height];
  RGBImage *edge_color_image = readPPM(filename.c_str());

  dim3 rgb_image_dim(rgb_image->width, rgb_image->height);
  dim3 gray_image_dim(gray_image->width, gray_image->height);
  
  if (rgb_image == NULL) {
    std::cout << "Error reading the image" << std::endl;
    exit(EXIT_FAILURE);
  }
 
  if (processing_unit == ProcessingUnit::Device) {
    // GPU
    // 1. First step, convert the picture into grayscale
    ProcessingUnitDevice::rgb_to_gray(rgb_image, gray_image);
    
    // 2. Second step, smooth the image using a Gaussian blur
    // to remove possible noise in the picture
    for (int i = 0; i < 5; i++) {
      ProcessingUnitDevice::gaussian_blur(gray_image->data, gray_image_dim);
      cudaDeviceSynchronize();
    }
    writePGM("output/blurred_image_output.pgm", gray_image);
    
    // 3. Third step, apply the Sobel-Feldman operator to detect edges of shapes
    ProcessingUnitDevice::sobel_feldman(gray_image->data, gradient_image->data, angle_image, gray_image_dim);
    writePGM("output/sf_gradient_output.pgm", gradient_image);

    ProcessingUnitDevice::generate_edge_color(gradient_image->data, angle_image, edge_color_image->data, rgb_image_dim);
    writePPM("output/edge_color_output.ppm", edge_color_image);

    GrayImage *buffer_gray = createPGM(gradient_image->width, gradient_image->height);
    RGBImage *buffer_rgb = createPPM(gradient_image->width, gradient_image->height);

    if (canny_sample_offset == 0) {
      canny_sample_offset = 255;
    }

    for (int i = canny_min; i <= canny_max; i += canny_sample_offset) {
      memcpy(buffer_gray->data, gradient_image->data, sizeof(unsigned char) * gradient_image->width * gradient_image->height);
      ProcessingUnitDevice::canny(buffer_gray->data, angle_image, gray_image_dim, i, canny_max);

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
      ProcessingUnitDevice::cutout(buffer_rgb->data, buffer_gray->data, gray_image_dim, start_pixel, 0);

      const char *prefix_rgb = "output/cutout_output";
      char number_rgb[4] = "000";
      sprintf(number_rgb, "%d", file_index);
      char filename_rgb[strlen(prefix_rgb) + 3 + 4 + 1]; // prefix + number + .ppm + \0
      bzero(filename_rgb, strlen(prefix_rgb) + 3 + 4 + 1);
      strcpy(filename_rgb, prefix_rgb);
      strcpy(filename_rgb + strlen(prefix_rgb), number_rgb);
      strcpy(filename_rgb + strlen(filename_rgb), ".ppm");
      printf("%s\n", filename_rgb);
      writePPM(filename_rgb, buffer_rgb);
  
      file_index += 1;
    }
    destroyPGM(buffer_gray);
    destroyPPM(buffer_rgb);
  } else if (processing_unit == ProcessingUnit::Host) {
    // CPU
    // 1. First step, convert the picture into grayscale
    ProcessingUnitHost::rgb_to_gray(rgb_image, gray_image);
    writePGM("output/gray_image.pgm", gray_image);

    // 2. Second step, smooth the image using a Gaussian blur
    // to remove possible noise in the picture
    for (int i = 0; i < 5; i++) {
      ProcessingUnitHost::gaussian_blur(gray_image->data, gray_image_dim);
    }
    writePGM("output/blurred_image.pgm", gray_image);
    
    // 3. Third step, apply the Sobel-Feldman operator to detect edges of shapes
    ProcessingUnitHost::sobel_feldman(gray_image->data, gradient_image->data, angle_image, gray_image_dim);
    writePGM("output/sf_gradient_output.pgm", gradient_image);
    
    ProcessingUnitHost::generate_edge_color(gradient_image->data, angle_image, edge_color_image->data, rgb_image_dim);
    writePPM("output/edge_color_output.ppm", edge_color_image);
    
    GrayImage *buffer_gray = createPGM(gradient_image->width, gradient_image->height);
    RGBImage *buffer_rgb = createPPM(gradient_image->width, gradient_image->height);
    int file_index = 0;

    if (canny_sample_offset == 0) {
      canny_sample_offset = 255;
    }

    for (int i = canny_min; i <= canny_max; i += canny_sample_offset) {
      memcpy(buffer_gray->data, gradient_image->data, sizeof(unsigned char) * gradient_image->width * gradient_image->height);
      ProcessingUnitHost::canny(buffer_gray->data, angle_image, gray_image_dim, i, canny_max);

      // Create the name of the output file
      const char *prefix_gray = "output/canny_output";
      char number_gray[4] = "000";
      sprintf(number_gray, "%d", file_index);
      char filename_gray[strlen(prefix_gray) + 3 + 4 + 1]; // prefix + number + .ppm + \0
      bzero(filename_gray, strlen(prefix_gray) + 3 + 4 + 1);
      strcpy(filename_gray, prefix_gray);
      strcpy(filename_gray + strlen(prefix_gray), number_gray);
      strcpy(filename_gray + strlen(filename_gray), ".ppm");

      printf("%s\n", filename_gray);
      writePGM(filename_gray, buffer_gray);
    
      // 4. Last step, cutout the object selected by the user
      memcpy(buffer_rgb->data, rgb_image->data, sizeof(unsigned char) * gradient_image->width * gradient_image->height * 3);
      ProcessingUnitHost::cutout(buffer_rgb->data, buffer_gray->data, gray_image_dim, start_pixel, 0);

      const char *prefix_rgb = "output/cutout_output";
      char number_rgb[4] = "000";
      sprintf(number_rgb, "%d", file_index);
      char filename_rgb[strlen(prefix_rgb) + 3 + 4 + 1]; // prefix + number + .ppm + \0
      bzero(filename_rgb, strlen(prefix_rgb) + 3 + 4 + 1);
      strcpy(filename_rgb, prefix_rgb);
      strcpy(filename_rgb + strlen(prefix_rgb), number_rgb);
      strcpy(filename_rgb + strlen(filename_rgb), ".ppm");
      printf("%s\n", filename_rgb);
      writePPM(filename_rgb, buffer_rgb);
  
      file_index += 1;
    }
    destroyPGM(buffer_gray);
    destroyPPM(buffer_rgb);
  }

  destroyPPM(rgb_image);
  destroyPGM(gray_image);  
  destroyPGM(gradient_image);  
  destroyPPM(edge_color_image);  
  delete [] angle_image; 
}
