#ifndef RGB_TO_GRAY_KERNEL_HPP
#define RGB_TO_GRAY_KERNEL_HPP

__global__ void rgb_to_gray_kernel(unsigned char *rgb_image, unsigned char *gray_image, dim3 image_dim);

#endif // RGB_TO_GRAY_KERNEL_HPP
