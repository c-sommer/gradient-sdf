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


#include "RigidPointOptimizer.h"

bool RigidPointOptimizer::optimize_sampled(const cv::Mat &depth, const Mat3f K, size_t sampling) {

    const float z_min = tSDF_->z_min_, z_max = tSDF_->z_max_;
    const int w = depth.cols;
    const int h = depth.rows;
    const float fx = K(0,0), fy = K(1,1);
    const float cx = K(0,2), cy = K(1,2);
    const float fx_inv = 1.f / fx;
    const float fy_inv = 1.f / fy;
    const float* depth_ptr = (const float*)depth.data;
    
    for (size_t k=0; k<num_iterations_; ++k) {

        Mat3f R = pose_.rotationMatrix();
        Vec3f t = pose_.translation();
        
        float E = 0; // cost
        Vec6f g = Vec6f::Zero(); // gradient
        Mat6f H = Mat6f::Zero(); // approximate Hessian
        
        size_t counter = 0;
        
        for (int y=0; y<h; y+=sampling) for (int x=0; x<w; x+=sampling) {
        
            const float z = depth_ptr[y*w + x];
            if (z<=z_min || z>=z_max) continue;
            
            const float x0 = (float(x) - cx) * fx_inv;
            const float y0 = (float(y) - cy) * fy_inv;
            Vec3f point(x0*z, y0*z, z);
            point = R * point + t;
            
            float w0 = tSDF_->weights(point);
            if (w0>0) {
                Vec3f grad_curr;
                float phi0 = tSDF_->tsdf(point, &grad_curr);
                E += phi0 * phi0;
                Vec6f grad_xi;
                grad_xi << grad_curr, point.cross(grad_curr);
                g += phi0 * grad_xi;
                H += grad_xi * grad_xi.transpose();
                ++counter;
            }
          
        }
        
        Vec6f xi = damping_ * H.llt().solve(g); // Gauss-Newton
        
        if (xi.squaredNorm() < conv_threshold_sq_) {
            std::cout << "... Convergence after " << k << " iterations!" << std::endl;
            return true;
        }
        
        // update pose
        if(!xi.array().isNaN().any())
            pose_ = SE3::exp(-xi) * pose_;
    }
    
    return false;
}