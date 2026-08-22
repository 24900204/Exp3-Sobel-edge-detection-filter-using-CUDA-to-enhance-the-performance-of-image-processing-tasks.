# Exp3-Sobel-edge-detection-filter-using-CUDA-to-enhance-the-performance-of-image-processing-tasks.
<h3>AIM:</h3>
<h3>Rithika L</h3>
<h3>212224040231</h3>
<h3>EX. NO 3</h3>
<h3>20.08.2026</h3>
<h1> <align=center> Sobel edge detection filter using CUDA </h3>
  Implement Sobel edge detection filtern using GPU.</h3>
Experiment Details:
  
## AIM:
  The Sobel operator is a popular edge detection method that computes the gradient of the image intensity at each pixel. It uses convolution with two kernels to determine the gradient in both the x and y directions. This lab focuses on utilizing CUDA to parallelize the Sobel filter implementation for efficient processing of images.

Code Overview: You will work with the provided CUDA implementation of the Sobel edge detection filter. The code reads an input image, applies the Sobel filter in parallel on the GPU, and writes the result to an output image.
## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
CUDA Toolkit and OpenCV installed.
A sample image for testing.

## PROCEDURE:
Tasks: 
a. Modify the Kernel:

Update the kernel to handle color images by converting them to grayscale before applying the Sobel filter.
Implement boundary checks to avoid reading out of bounds for pixels on the image edges.

b. Performance Analysis:

Measure the performance (execution time) of the Sobel filter with different image sizes (e.g., 256x256, 512x512, 1024x1024).
Analyze how the block size (e.g., 8x8, 16x16, 32x32) affects the execution time and output quality.

c. Comparison:

Compare the output of your CUDA Sobel filter with a CPU-based Sobel filter implemented using OpenCV.
Discuss the differences in execution time and output quality.

## PROGRAM:
```
!pip install git+https://github.com/andreinechaev/nvcc4jupyter.git
%load_ext nvcc4jupyter
```



<img width="788" height="334" alt="image" src="https://github.com/user-attachments/assets/f70aa433-fad1-4e67-8e87-f2d606a852f9" />



```
!nvcc --version
```



<img width="602" height="108" alt="image" src="https://github.com/user-attachments/assets/3d64dbab-181f-4692-a295-491b5265b1ca" />



```
%load_ext nvcc4jupyter
```



<img width="654" height="52" alt="image" src="https://github.com/user-attachments/assets/6a6dac20-e83d-4c5a-8e65-32c730b7fae7" />



```
from pathlib import Path

file_path = Path('/absolute/path/to/images.jpeg')
if file_path.exists():
    print("File exists!")
else:
    print("File does not exist!")
```




<img width="378" height="43" alt="image" src="https://github.com/user-attachments/assets/c47b9994-5a5a-4f53-8722-b0dcafd47b1a" />



```
import os
print("Current Working Directory:", os.getcwd())
```



<img width="484" height="36" alt="image" src="https://github.com/user-attachments/assets/a20aa44d-5668-4803-82f1-ae859993db1a" />



```
from google.colab import files
uploaded = files.upload()
```



<img width="747" height="80" alt="image" src="https://github.com/user-attachments/assets/d88d4991-f7e8-44d0-8e21-eb67b52d21e5" />



```
from pathlib import Path

file_path = Path("images.jpeg")

if file_path.exists():
    print("File exists!")
else:
    print("File does not exist!")
```



<img width="249" height="37" alt="image" src="https://github.com/user-attachments/assets/229caa38-399b-48ab-a82e-017de1981638" />



```
pwd
```



<img width="199" height="39" alt="image" src="https://github.com/user-attachments/assets/d0420344-922f-458a-af49-3e87719a2955" />



```
ls "/content/images.jpeg"
```



<img width="263" height="31" alt="image" src="https://github.com/user-attachments/assets/fcfebbf1-617a-449c-bd89-40f4d9be275e" />



```
#ls -l /content/66666.jpg
import cv2
image = cv2.imread('/content/images.jpeg')
if image is None:
    print("Error: Image not found or unable to read the image.")
else:
    print("Image read successfully.")
```



<img width="283" height="36" alt="image" src="https://github.com/user-attachments/assets/c5636de8-a861-4bca-8ea0-234f3cd57c3f" />



```
%%writefile sobelEdgeDetectionFilter.cu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

using namespace cv;

__global__ void sobelFilter(unsigned char *srcImage,
                            unsigned char *dstImage,
                            unsigned int width,
                            unsigned int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float Kx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    float Ky[3][3] = {
        {1, 2, 1},
        {0, 0, 0},
        {-1, -2, -1}
    };

    // Only threads inside the valid image area
    if (x >= 1 && x < width - 1 &&
        y >= 1 && y < height - 1)
    {
        // Gradient in x-direction
        float Gx = 0;

        for (int ky = -1; ky <= 1; ky++)
        {
            for (int kx = -1; kx <= 1; kx++)
            {
                float fl = srcImage[
                    (y + ky) * width + (x + kx)
                ];

                Gx += fl * Kx[ky + 1][kx + 1];
            }
        }

        float Gx_abs = Gx < 0 ? -Gx : Gx;

        // Gradient in y-direction
        float Gy = 0;

        for (int ky = -1; ky <= 1; ky++)
        {
            for (int kx = -1; kx <= 1; kx++)
            {
                float fl = srcImage[
                    (y + ky) * width + (x + kx)
                ];

                Gy += fl * Ky[ky + 1][kx + 1];
            }
        }

        float Gy_abs = Gy < 0 ? -Gy : Gy;

        float result = Gx_abs + Gy_abs;

        if (result > 255)
            result = 255;

        dstImage[y * width + x] =
            (unsigned char)result;
    }
    else if (x < width && y < height)
    {
        dstImage[y * width + x] = 0;
    }
}

void checkCudaErrors(cudaError_t r)
{
    if (r != cudaSuccess)
    {
        fprintf(stderr,
                "CUDA Error: %s\n",
                cudaGetErrorString(r));

        exit(EXIT_FAILURE);
    }
}

int main()
{
    // Read input image
    Mat image = imread(
        "/content/66666.jpg",
        IMREAD_GRAYSCALE
    );

    if (image.empty())
    {
        printf("Error: Image not found.\n");
        return -1;
    }

    int width = image.cols;
    int height = image.rows;

    size_t imageSize =
        width * height * sizeof(unsigned char);

    // Allocate host memory for output image
    unsigned char *h_outputImage =
        (unsigned char *)malloc(imageSize);

    if (h_outputImage == nullptr)
    {
        fprintf(stderr,
                "Failed to allocate host memory\n");

        return -1;
    }

    // Allocate device memory
    unsigned char *d_inputImage;
    unsigned char *d_outputImage;

    checkCudaErrors(
        cudaMalloc(&d_inputImage, imageSize)
    );

    checkCudaErrors(
        cudaMalloc(&d_outputImage, imageSize)
    );

    checkCudaErrors(
        cudaMemcpy(
            d_inputImage,
            image.data,
            imageSize,
            cudaMemcpyHostToDevice
        )
    );

    // Define CUDA events for timing
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch kernel
    dim3 blockSize(16, 16);

    dim3 gridSize(
        (width + 15) / 16,
        (height + 15) / 16
    );

    cudaEventRecord(start);

    sobelFilter<<<gridSize, blockSize>>>(
        d_inputImage,
        d_outputImage,
        width,
        height
    );

    cudaEventRecord(stop);

    // Synchronize events
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;

    cudaEventElapsedTime(
        &milliseconds,
        start,
        stop
    );

    // Copy result back to host
    checkCudaErrors(
        cudaMemcpy(
            h_outputImage,
            d_outputImage,
            imageSize,
            cudaMemcpyDeviceToHost
        )
    );

    // Write output image
    Mat outputImage(
        height,
        width,
        CV_8UC1,
        h_outputImage
    );

    imwrite(
        "/content/output_sobel.jpeg",
        outputImage
    );

    // Free memory
    free(h_outputImage);

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Print elapsed time
    printf("Total time taken: %f milliseconds\n",
           milliseconds);

    return 0;
}
```




<img width="411" height="32" alt="image" src="https://github.com/user-attachments/assets/a203c938-528e-49ad-9316-5dc8ec1f5423" />



```

!nvcc -o sobelEdgeDetectionFilter sobelEdgeDetectionFilter.cu `pkg-config --cflags --libs opencv4`
```



<img width="773" height="181" alt="image" src="https://github.com/user-attachments/assets/5fce4487-4455-4b76-b639-e1e789b32dae" />



```
!./sobelEdgeDetectionFilter
```




<img width="701" height="34" alt="image" src="https://github.com/user-attachments/assets/3ea44c2e-3aee-48f3-9b3e-bad3dba030c8" />




```
import cv2
from matplotlib import pyplot as plt
```



```
output_image_path = '/content/images.jpeg'
output_image = cv2.imread(output_image_path, cv2.IMREAD_GRAYSCALE)  # Use IMREAD_GRAYSCALE if it's a single-channel image
```



```
plt.imshow(output_image, cmap='gray')
plt.title('Edge Detection Output')
plt.axis('off')  # Hide the axes
plt.show()
```




<img width="728" height="478" alt="image" src="https://github.com/user-attachments/assets/8067b3e6-eb21-4dc2-9199-702b03224a7d" />


## RESULT:
Thus the program has been executed by using CUDA to Edge detection.

Questions:

What challenges did you face while implementing the Sobel filter for color images?
How did changing the block size influence the performance of your CUDA implementation?
What were the differences in output between the CUDA and CPU implementations? Discuss any discrepancies.
Suggest potential optimizations for improving the performance of the Sobel filter.

Deliverables:

Modified CUDA code with comments explaining your changes.
A report summarizing your findings, including graphs of execution times and a comparison of outputs.
Answers to the questions posed in the experiment.
Tools Required:

