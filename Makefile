main: main.cu img.cpp edge_detection/sobel_feldman.cu edge_detection/canny.cu cutout.cu utils.cu
	nvcc -o main main.cu img.cpp edge_detection/sobel_feldman.cu edge_detection/canny.cu cutout.cu utils.cu -lm

tests: main.cu img.cpp tests.cu edge_detection/sobel_feldman.cu edge_detection/canny.cu cutout.cu utils.cu
	nvcc -o main main.cu img.cpp tests.cu edge_detection/sobel_feldman.cu edge_detection/canny.cu cutout.cu utils.cu -lm

