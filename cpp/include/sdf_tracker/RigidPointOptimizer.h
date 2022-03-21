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


#ifndef RIGID_POINT_OPTIMIZER_H_
#define RIGID_POINT_OPTIMIZER_H_

// includes
#include <iostream>
//class includes
#include "RigidOptimizer.h"

/**
 * class declaration
 */
class RigidPointOptimizer : public RigidOptimizer {

public:

// constructors / destructor

    RigidPointOptimizer(Sdf* tSDF) :
        RigidOptimizer(tSDF)
    {}
    
    RigidPointOptimizer(int num_iterations, float conv_threshold, float damping, Sdf* tSDF) :
        RigidOptimizer(num_iterations, conv_threshold, damping, tSDF)
    {}
    
    ~RigidPointOptimizer() {}
    
    bool optimize_sampled(const cv::Mat &depth, const Mat3f K, size_t sampling);
    
// member functions
    
    bool optimize(const cv::Mat &depth, const Mat3f K) {
        // tSDF_->increase_counter();
        return optimize_sampled(depth, K, 1);
    }

};

#endif // RIGID_POINT_OPTIMIZER_H_