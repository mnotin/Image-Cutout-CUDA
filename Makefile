SRC=main.cu \
    img.cpp \
    device/edge_detection/sobel_feldman.cu \
    device/edge_detection/canny.cu \
    device/cutout.cu \
    device/utils.cu \
    device/tests.cu

main: $(SRC)
	nvcc -o main $(SRC) -lm
