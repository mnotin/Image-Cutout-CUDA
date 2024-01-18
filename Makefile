SRC=main.cu \
    img.cpp \
    tests.cu \
    edge_detection/sobel_feldman.cu \
    edge_detection/canny.cu \
    cutout.cu \
    utils.cu

main: $(SRC)
	nvcc -o main $(SRC) -lm
