main: main.cu img.cpp edge_detection.cu
	nvcc -o main main.cu img.cpp edge_detection.cu -lm
