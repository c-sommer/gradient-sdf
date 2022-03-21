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

#ifndef TUMRGBD_LOADER_H_
#define TUMRGBD_LOADER_H_

#include "ImageLoader.h"
#include <fstream>
#include <sstream>

class TumrgbdLoader : public ImageLoader {

private:

    std::ifstream depth_file_;
    std::ifstream color_file_;
    std::ifstream assoc_file_;
    
    void init() {
        depth_file_.open(path_ + "depth.txt");
        color_file_.open(path_ + "rgb.txt");
        assoc_file_.open(path_ + "associated.txt");
        return;
    }

public:

    TumrgbdLoader() :
        ImageLoader(1./5000, false)
    {
        init();
    }
    
    TumrgbdLoader(const std::string& path) :
        ImageLoader(1./5000, path, false)
    {
        init();
    }
    
    ~TumrgbdLoader() {
        depth_file_.close();
        color_file_.close();
        assoc_file_.close();
    }
    

    bool load_next(cv::Mat& color, cv::Mat& depth) {
                
        std::string line, tmp, rgb_filename, depth_filename;
        
        line = "#";
        while (line.at(0) == '#') {
            if(!std::getline(assoc_file_, line))
                return false;
        }
        
        std::istringstream dss(line);
        dss >> timestamp_rgb_ >> rgb_filename >> timestamp_depth_ >> depth_filename;
        
        std::cout << "load image " << timestamp_rgb_ << std::endl;
        
        timestamps_depth_.push_back(timestamp_depth_);
        timestamps_rgb_.push_back(timestamp_rgb_);
        
        if (!load_depth(depth_filename, depth))
            return false;        

        if (!load_color(rgb_filename, color))
            return false;
        return true;
    }

    void reset() {
        depth_file_.close();
        color_file_.close();
        assoc_file_.close();
        depth_file_.open(path_ + "depth.txt");
        color_file_.open(path_ + "rgb.txt");
        assoc_file_.open(path_ + "associated.txt");
        timestamps_depth_.clear();
        timestamps_rgb_.clear();
    }

};

#endif // TUMRGBD_LOADER_H_
