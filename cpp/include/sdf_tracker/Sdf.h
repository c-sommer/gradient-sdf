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


#ifndef SDF_H_
#define SDF_H_

// includes
#include <iostream>

namespace cv {
    template <typename T>
    class NormalEstimator;
}

/**
 * class declaration
 */
class Sdf {

protected:

// friends

    friend class RigidPointOptimizer;
    friend class RigidSdfOptimizer;

// variables
    
    float T_; // truncation distance in meters
    float inv_T_;
    size_t counter_; // frame counter

    float z_min_ = 0.5;
    float z_max_ = 3.5;
    
// methods
    
    float truncate(float sdf) const {
        return std::max(-T_, std::min(T_, sdf));
    }
    
    float weight(float sdf) const {
        float w = 0.f;
        if (sdf<=0.) {
            w = 1.f;
        }
        else if (sdf<=T_) {
            w = 1.f - sdf*inv_T_;
        }
        return w;
    }

    void increase_counter() {
        ++counter_;
    }

//    void init();
    
public:

// constructors / destructor
    
    Sdf() :
        T_(0.05),
        inv_T_(1./T_),
        counter_(0)
    {}
    
    Sdf(float T) :
        T_(T),
        inv_T_(1./T_),
        counter_(0)
    {}
    
    virtual ~Sdf() {}
    
// methods
    
    virtual float tsdf(Vec3f point, Vec3f* grad_ptr = nullptr) const = 0;

    virtual float weights(Vec3f point) const = 0;
    
    virtual void update(const cv::Mat &color, const cv::Mat &depth, const Mat3f K, const SE3 &pose, cv::NormalEstimator<float>* NEst = nullptr) = 0;
    
    virtual void setup(const cv::Mat &color, const cv::Mat &depth, const Mat3f K, cv::NormalEstimator<float>* NEst = nullptr) {
        update(color, depth, K, SE3(), NEst);
    }

    void set_zmin(float z_min) {
        z_min_ = z_min;
    }

    void set_zmax(float z_max) {
        z_max_ = z_max;
    }



// visualization / debugging

    virtual bool extract_mesh(std::string filename = "mesh.ply") {
        return false;
    }
    
    virtual bool extract_pc(std::string filename = "cloud.ply") {
        return false;
    }

    virtual bool save_sdf(std::string filename ="sdf.txt"){
        return false;
    }

    // virtual void write() {}

};

#endif // SDF_H_
