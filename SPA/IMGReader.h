#pragma once
#include <opencv2/opencv.hpp>
#include "SPALib.h"
void convertImageToDoubleMatrix(const std::string& imagePath, int targetRows, int targetCols, std::vector<std::vector<std::vector<Num>>>& inp) {
    // Load the image in grayscale
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);

    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image: " << imagePath << std::endl;
        return; // Return an empty matrix on error
    }

    // Resize the image to the target size
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(targetCols, targetRows));

    // Convert the resized image to double
    cv::Mat doubleImage;
    resizedImage.convertTo(doubleImage, CV_64F); // Convert pixel values to double
    for (int i = 0; i < doubleImage.rows; ++i) {
        double* rowPtr = doubleImage.ptr<double>(i); // Pointer to the current row
        for (int j = 0; j < doubleImage.cols; ++j) {
            double pixelValue = rowPtr[j];
            pixelValue = AddtionalFunctions::MapRange(pixelValue, 0, 255, 0, 1);
            if (inp[0][i][j].curNode)
            {
                inp[0][i][j].val = pixelValue;
            }
            else
            {
                inp[0][i][j] = pixelValue;
            }
        }
    }

    return;
}