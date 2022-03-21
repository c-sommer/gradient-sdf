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


#ifndef PRINTED3D_LOADER_H_
#define PRINTED3D_LOADER_H_

#include <iomanip>
#include "ImageLoader.h"

class Printed3dLoader : public ImageLoader {

private:

    size_t counter;

public:

    Printed3dLoader() :
        ImageLoader(1./1000, true),
        counter(0)
    {}
    
    Printed3dLoader(const std::string& path) :
        ImageLoader(1./1000, path, true),
        counter(0)
    {}
    
    ~Printed3dLoader() {}
    
    bool load_next(cv::Mat& color, cv::Mat& depth) {

        std::stringstream ss;
        ss << std::setfill('0') << std::setw(6) << counter;

        timestamp_rgb_ = ss.str();
        timestamp_depth_ = timestamp_rgb_;

        const std::string filename = timestamp_rgb_ + ".png";
    
        if (!load_depth("depth_" + filename, depth))
            return false;
        
        if (!load_color("color_" + filename, color))
            return false;
        
        ++counter;
        
        return true;
    }
    bool load_keyframe(cv::Mat &color, cv::Mat &depth, const int frame)
    {
        
        std::stringstream ss;
        ss << std::setfill('0') << std::setw(6) << frame;

        timestamp_rgb_ = ss.str();
        timestamp_depth_ = timestamp_rgb_;

        const std::string filename = timestamp_rgb_ + ".png";
        timestamps_depth_.push_back(timestamp_depth_);
        timestamps_rgb_.push_back(timestamp_rgb_);
    
        if (!load_depth("depth_" + filename, depth))
            return false;
        
        if (!load_color("color_" + filename, color))
            return false;
        
        
        return true;
    }

    void reset()
    {
        counter = 0;
        timestamps_depth_.clear();
        timestamps_rgb_.clear();
    }

};

#endif // PRINTED3D_LOADER_H_
