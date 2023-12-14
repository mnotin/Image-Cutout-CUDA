main: main.cu img.cpp edge_detection.cu
	nvcc -o main main.cu img.cpp edge_detection.cu -lm

tests: main.cu img.cpp edge_detection.cu tests.cu
	nvcc -o main main.cu img.cpp edge_detection.cu tests.cu -lm
