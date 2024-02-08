# Image Cutout CUDA

## Building the app
Ensure that the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) and [Make](https://www.gnu.org/software/make/) are installed on your machine.
Then clone the project and enter the directory using the `cd` command.
Finally, simply run `make` and the build process should start automatically.

DISCLAIMER: the application has been tested using the [CUDA Toolkit 10.2](https://developer.nvidia.com/cuda-10.2-download-archive). Therefore, please note that it is not guaranteed to work properly if you are using a different version.

## How to use the app
A folder named `input` has to be present in the same directory of the application.
Then simply put the images that have to be processed inside this folder using the following naming patern: `frameXXXXX.ppm`.

The app will then look for images in the folder by replacing the `XXXXX` pattern by a number with padding `0` on the left.
The app starts from `1` (i.e. `frame00001.ppm`) and increments this number until no more images following the pattern are found.

The app will place the output images inside an `output` folder.
So please make sure that the `input` and `ouput` folders are present in the same directory of the application.

For more details about how to configure the cutout, run `./output --help`.

**Note**: for now, only images in the portable pixelmap (PPM) format are supported.

### How to use the app for a video
Since a video is a set of images, you can also apply the cutout process on a video.
For this simply extrat the different images from a video and place the images in the `input` folder as explained previously.
Then after running the app on these frames, reassemble the images into a video.

To execute this process, you can use [ffmpeg](https://ffmpeg.org/) for example.
To extract the images from an MP4 video for example run the following command:
```
ffmpeg -i "video.mp4" input/frame%05d.ppm
```
And the you can recreate the video using this command:
```
ffmpeg -i output/frame%05d.ppm -c:v libx264 -r 25 -pix_fmt yuv420p -c:a copy -shortest "video_processed.mp4"
```
