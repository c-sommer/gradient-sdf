/**
BSD 3-Clause License

This file is part of the code accompanying the paper
Gradient-SDF: A Semi-Implicit Surface Representation for 3D Reconstruction
by Christiane Sommer*, Lu Sang*, David Schubert, and Daniel Cremers (* denotes equal contribution).

Copyright (c) 2021, Christiane Sommer and Lu Sang.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#ifndef SHARP_DETECTOR_H_
#define SHARP_DETECTOR_H_

#include <opencv2/core/core.hpp>

float modifiedLaplacian(const cv::Mat& src);

bool sharpDetector(const cv::Mat& img, float threshold){
    float measure = modifiedLaplacian(img);
    std::cout << "the sharpness measure is " << measure << "." << std::endl;
    if( measure < threshold)
        return false;
    return true;
}


// OpenCV port of 'LAPM' algorithm (Nayar89)
float modifiedLaplacian(const cv::Mat& src)
{
    cv::Mat M = (cv::Mat_<float>(3, 1) << -1, 2, -1);
    cv::Mat G = cv::getGaussianKernel(3, -1, CV_32F);

    cv::Mat Lx;
    cv::sepFilter2D(src, Lx, CV_32F, M, G);

    cv::Mat Ly;
    cv::sepFilter2D(src, Ly, CV_32F, G, M);

    cv::Mat FM = cv::abs(Lx) + cv::abs(Ly);

    float focusMeasure = cv::mean(FM).val[0];
    return focusMeasure;
}

#endif // SHARP_DETECTOR_H_