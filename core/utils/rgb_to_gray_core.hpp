#ifndef RGB_TO_GRAY_CORE_HPP
#define RGB_TO_GRAY_CORE_HPP

__device__ __host__ unsigned char rgb_to_gray_core(int2 index, unsigned char *rgb_image, dim3 image_dim);

#endif // RGB_TO_GRAY_CORE_HPP
