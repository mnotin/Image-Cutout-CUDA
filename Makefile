main: main.cu img.cpp sobel_feldman.cu cutout.cu utils.cu
	nvcc -o main main.cu img.cpp sobel_feldman.cu cutout.cu utils.cu -lm

tests: main.cu img.cpp tests.cu sobel_feldman.cu cutout.cu utils.cu
	nvcc -o main main.cu img.cpp tests.cu sobel_feldman.cu cutout.cu utils.cu -lm
