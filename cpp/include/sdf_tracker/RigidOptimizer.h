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


#ifndef RIGID_OPTIMIZER_H_
#define RIGID_OPTIMIZER_H_

// includes
#include <iostream>
#include <opencv2/core/core.hpp>
#include "mat.h"
//class includes
#include "Sdf.h"

/**
 * class declaration
 */
class RigidOptimizer {

protected:

// variables
    
    int num_iterations_;
    float conv_threshold_;
    float conv_threshold_sq_;
    float damping_;
    
    Sdf* tSDF_;
    
    SE3 pose_ = SE3();

public:

// constructors / destructor

    RigidOptimizer(Sdf* tSDF) :
        num_iterations_(25),
        conv_threshold_(1e-3),
        conv_threshold_sq_(conv_threshold_ * conv_threshold_),
        damping_(1.),
        tSDF_(tSDF)
    {}
    
    RigidOptimizer(int num_iterations, float conv_threshold, float damping, Sdf* tSDF) :
        num_iterations_(num_iterations),
        conv_threshold_(conv_threshold),
        conv_threshold_sq_(conv_threshold_ * conv_threshold_),
        damping_(damping),
        tSDF_(tSDF)
    {}
    
    virtual ~RigidOptimizer() {}
    
// member functions
    
    void set_num_iterations(int num_iterations) {
        num_iterations_ = num_iterations;
    }
    
    void set_conv_threshold(float conv_threshold) {
        conv_threshold_ = conv_threshold;
        conv_threshold_sq_ = conv_threshold_ * conv_threshold_;
    }
    
    void set_damping(float damping) {
        damping_  = damping;
    }
    
    void set_pose(SE3 pose) { pose_ = pose; }
    
    SE3 pose() {
        return pose_;
    }
    
    // virtual bool optimize() {}
    virtual bool optimize(const cv::Mat &depth, const Mat3f K) = 0;

};

#endif // RIGID_OPTIMIZER_H_
