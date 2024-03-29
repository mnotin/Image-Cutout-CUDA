OBJ=main.o \
    edge_detector.o \
    img.o \
    core/edge_detection/edge_detection_core.o \
    launcher/edge_detection/sobel_feldman_launcher.o \
    kernel/edge_detection/sobel_feldman_kernel.o \
    core/edge_detection/sobel_feldman_core.o \
    launcher/edge_detection/canny_launcher.o \
    kernel/edge_detection/canny_kernel.o \
    core/edge_detection/canny_core.o \
    kernel/utils/convolution_kernel.o \
    core/utils/convolution_core.o \
    launcher/utils/gaussian_blur_launcher.o \
    launcher/utils/rgb_to_gray_launcher.o \
    kernel/utils/rgb_to_gray_kernel.o \
    core/utils/rgb_to_gray_core.o \
    launcher/cutout_launcher.o \
    kernel/cutout_kernel.o \
    core/cutout_core.o

%.o: %.cu
	nvcc --gpu-architecture=sm_50 --device-c $< -o $@ $(SRC)

%.o: %.cpp
	nvcc -x cu --gpu-architecture=sm_50 --device-c $< -o $@ $(SRC)

main: $(OBJ)
	nvcc -o cutout --gpu-architecture=sm_50 $(OBJ) -lm

clean:
	rm -f $(OBJ) cutout
