# Image Cutout CUDA

## Building the app
Ensure that the [NVIDIA CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) and [Make](https://www.gnu.org/software/make/) are installed on your machine.
Then clone the project and enter the directory using the `cd` command.
Finally, simply run `make` and the build process should start automatically.

## How to use the app
A folder named `input` has to be present in the same directory of the application.
Then simply put the images that have to be processed inside this folder using the following naming patern: `frameXXXXX.ppm`.

The app will then look for images in the folder by replacing the `XXXXX` pattern by a number with padding `0` on the left.
The app starts from `0` (i.e. `frame00000.ppm`) and increments this number until no more images following the pattern are found.

For more details about how to configure the cutout, run `./output --help`.
