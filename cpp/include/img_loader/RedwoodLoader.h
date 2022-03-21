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


#ifndef REDWOOD_LOADER_H_
#define REDWOOD_LOADER_H_

#include "ImageLoader.h"
#include <fstream>
#include <sstream>
#include <dirent.h>

class RedwoodLoader : public ImageLoader {

private:

    DIR *depth_dir_;
    dirent *depth_ent_;
    DIR *rgb_dir_;
    dirent *rgb_ent_;
    std::vector<std::string> depth_filenames_, rgb_filenames_;
    size_t counter;
    
    void init() {
        depth_dir_ = opendir((path_ + "depth/").c_str());
        rgb_dir_   = opendir((path_ + "rgb/").c_str());
        readdir(depth_dir_);
        readdir(depth_dir_);
        readdir(rgb_dir_);
        readdir(rgb_dir_);
        load_file_name();
        return;
    }

public:

    RedwoodLoader() :
        ImageLoader(1./1000, true),
        counter(0)
    {
        init();
    }
    
    RedwoodLoader(const std::string& path) :
        ImageLoader(1./1000, path, true),
        counter(0)
    {
        init();
    }

    ~RedwoodLoader()
    {
        closedir(depth_dir_);
        closedir(rgb_dir_);
    }
    
    void load_file_name(){

        while ((depth_ent_ = readdir(depth_dir_))!= NULL && (rgb_ent_ = readdir(rgb_dir_)) != NULL){
            std::string depthfiles, rgbfiles;
            depthfiles = depth_ent_->d_name;
            rgbfiles = rgb_ent_->d_name;

            if(depthfiles.substr(depthfiles.length()-3)=="png")
                depth_filenames_.push_back(depthfiles);
            if(rgbfiles.substr(rgbfiles.length()-3)=="jpg")
                rgb_filenames_.push_back(rgbfiles);
        }

        std::sort(depth_filenames_.begin(), depth_filenames_.end());
        std::sort(rgb_filenames_.begin(),rgb_filenames_.end());

    }

    bool load_next(cv::Mat& color, cv::Mat& depth) {

        std::string filename;

        // if (!(depth_ent_ = readdir(depth_dir_)) || !(rgb_ent_ = readdir(rgb_dir_)))
            // return false;
        
        filename = depth_filenames_[counter];
        timestamp_depth_ = filename; //TODO get rid of suffix
        timestamps_depth_.push_back(filename);

        std::cout << "------load depth " << filename << ". " << std::endl;
        
        if (!load_depth("depth/" + filename, depth))
            return false;        
        
        filename = rgb_filenames_[counter];
        timestamps_rgb_.push_back(filename);

         std::cout << "------load rgb " << filename << ". " << std::endl;

        if (!load_color("rgb/" + filename, color))
            return false;
        
        ++counter;
        return true;
    }

    void reset() {
        init();
        timestamps_depth_.clear();
        timestamps_rgb_.clear();
        counter = 0;
    }

};

#endif // REDWOOD_LOADER_H_
