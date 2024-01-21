SRC=main.cu \
    img.cpp \
    tests.cu \
    edge_detection/sobel_feldman.cu \
    edge_detection/canny.cu \
    utils/convolution.cu \
    utils/gaussian_blur.cu \
    utils/rgb_to_gray.cu \
    cutout.cu

SRC_OBJ=main.o \
        img.o \
        tests.o \
        sobel_feldman.o \
        canny.o \
        convolution.o \
        gaussian_blur.o \
        rgb_to_gray.o \
        cutout.o


main: $(SRC)
	nvcc --gpu-architecture=sm_50 --device-c $(SRC)
	nvcc -o main --gpu-architecture=sm_50 $(SRC_OBJ) -lm
	rm *.o
