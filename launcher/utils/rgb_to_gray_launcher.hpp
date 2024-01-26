#ifndef RGB_TO_GRAY_LAUNCHER_HPP
#define RGB_TO_GRAY_LAUNCHER_HPP

#include "../../img.h"


namespace ProcessingUnitDevice {
  void rgb_to_gray(RGBImage *h_rgb_image, GrayImage *h_gray_image);
}


namespace ProcessingUnitHost {
  void rgb_to_gray(RGBImage *rgb_image, GrayImage *gray_image);
}

#endif // RGB_TO_GRAY_LAUNCHER_HPP
