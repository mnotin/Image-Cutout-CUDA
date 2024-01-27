#include <iostream>
#include <chrono>
#include <fstream>
#include <iomanip>

#include "main.hpp"
#include "tests.hpp"

int main(int argc, char **argv) {
  int2 cutout_start_pixel = make_int2(0, 0);
  int2 tracking_start_pixel = make_int2(-1, -1);
  ProcessingUnit processing_unit = ProcessingUnit::Device;
  int canny_min_val = 50;
  int canny_max_val = 100;
  int canny_sample_offset = 0; // Zero: no sample; non-zero value: sample

  if (argc == 2 && strcmp(argv[1], "--help") == 0) {
    print_help();

    return 0;
  } else {
    int i;
    bool bad_usage = false;

    for (i = 1; i < argc && !bad_usage; i++) {
      if (strcmp(argv[i], "--cutout-start-pixel") == 0) {
        cutout_start_pixel.x = atoi(argv[i+1]);
        cutout_start_pixel.y = atoi(argv[i+2]);
        i += 2;
      } else if (strcmp(argv[i], "--tracking-start-pixel") == 0) {
        tracking_start_pixel.x = atoi(argv[i+1]);
        tracking_start_pixel.y = atoi(argv[i+2]);
        i += 2;
      } else if (strcmp(argv[i], "--canny-thresholds") == 0) {
        canny_min_val = atoi(argv[i+1]);
        canny_max_val = atoi(argv[i+2]);
        i += 2;

        if (canny_min_val < 0 || 255 < canny_max_val || canny_max_val < canny_min_val) {
          bad_usage = true;
        }
      } else if (strcmp(argv[i], "--processing-unit") == 0) {
        if (strcmp(argv[i+1], "host") == 0) {
          processing_unit = ProcessingUnit::Host;
          i += 1;
        } else if (strcmp(argv[i+1], "device") == 0) {
          processing_unit = ProcessingUnit::Device;
          i += 1;
        } else {
          bad_usage = true;
        }
      } else if (strcmp(argv[i], "--canny-sampling-offset") == 0) {
        canny_sample_offset = atoi(argv[i+1]);
        i += 1;

        if (canny_sample_offset < 0 || 255 < canny_sample_offset) {
          bad_usage = true;
        }
      } else {
        // This option did not match any possible one
        bad_usage = true;
      }
    }
  
    if (bad_usage) {
      // Filename is missing or bad usage
      print_bad_usage();
      exit(EXIT_FAILURE);
    }
  }
  
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  
  int file_index = 1;
  std::string prefix("input/frame");

  std::stringstream string_stream;
  std::string number;
  string_stream << std::setw(5) << std::setfill('0') << file_index;
  string_stream >> number;

  std::string filename; // prefix + number + .ppm + \0
  filename.append(prefix);
  filename.append(number);
  filename.append(".ppm");
  std::ifstream file(filename);
  
  while (file.good()) {
    file.close();

    test_canny(filename, cutout_start_pixel, &tracking_start_pixel, canny_min_val, canny_max_val, canny_sample_offset, processing_unit, file_index);
    if (canny_sample_offset != 0)
      break; // We sample only the first image
    file_index += 1;
    
    std::stringstream string_stream2;
    string_stream2 << std::setw(5) << std::setfill('0') << file_index;
    number.clear();
    string_stream2 >> number;

    filename.clear();
    filename.append(prefix);
    filename.append(number);
    filename.append(".ppm");

    file.open(filename);
  }
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  
  std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "s" << std::endl;

  std::cout << " ===" << std::endl;
  cudaDeviceSynchronize();
  cudaError_t error = cudaPeekAtLastError();
  std::cout << "Error: " << cudaGetErrorString(error) << std::endl;

  return 0;
}

void print_help() {
  std::cout << "Usage: ./main [OPTION] file" << std::endl;
  std::cout << "\t--cutout-start-pixel <x> <y>\t\tPixel coordinates where the cutout algorithm should start. (default: 0 0)" << std::endl;
  std::cout << "\t--tracking-start-pixel <x> <y>\t\tPixel coordinates inside the object to track. (default: center of the image)" << std::endl;

  std::cout << "\t--canny-thresholds <min> <max>\t\tSpecify the thresholds that have to be used by the Canny edge detector (default: 50 100)" << std::endl;
  std::cout << "\t\t\t\t\t\tPermissible values are integer between 0 and 255." << std::endl;
  std::cout << "\t--processing-unit <processing-unit>\tSpecify where the cutout process has to be executed. (default: device)" << std::endl;
  std::cout << "\t\t\t\t\t\tPermissible processing units are 'host' (CPU) and 'device' (GPU)." << std::endl;
  std::cout << "\t--canny-sampling-offset <offset>\tSpecify that canny should produce multiple outputs, " \
  "starting from the minimum value threshold up to the maximum value" << std::endl;
  std::cout << "\t\t\t\t\t\twith an offset of 'offset' between each sample." << std::endl;

  std::cout << "\t--help\t\t\t\t\tDisplay this help and exit." << std::endl;
}

void print_bad_usage() {
  std::cout << "Usage: ./main [OPTION] file" << std::endl;
  std::cout << "Try './main --help' for more information." << std::endl;
}
