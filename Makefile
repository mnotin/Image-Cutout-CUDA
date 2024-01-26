SRC=main.cu \
    img.cpp \
    tests.cu \
    launcher/edge_detection/sobel_feldman_launcher.cu \
    kernel/edge_detection/sobel_feldman_kernel.cu \
    core/edge_detection/sobel_feldman_core.cu \
    launcher/edge_detection/canny_launcher.cu \
    kernel/edge_detection/canny_kernel.cu \
    core/edge_detection/canny_core.cu \
    kernel/utils/convolution_kernel.cu \
    core/utils/convolution_core.cu \
    launcher/utils/gaussian_blur_launcher.cu \
    launcher/utils/rgb_to_gray_launcher.cu \
    kernel/utils/rgb_to_gray_kernel.cu \
    core/utils/rgb_to_gray_core.cu \
    launcher/cutout_launcher.cu \
    kernel/cutout_kernel.cu \
    core/cutout_core.cu

main: $(SRC)
	nvcc --gpu-architecture=sm_50 --device-c $(SRC)
	nvcc -o main --gpu-architecture=sm_50 *.o -lm
	rm *.o

debug: $(SRC)
	nvcc --gpu-architecture=sm_50 --device-c -G -g $(SRC)
	nvcc -o main --gpu-architecture=sm_50 -G -g *.o -lm
	rm *.o
