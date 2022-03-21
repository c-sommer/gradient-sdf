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


#ifndef MAP_PIXEL_SDF_H_
#define MAP_PIXEL_SDF_H_

// includes
#include <iostream>
#include "mat.h"
// #include <opencv/cv.h>
#include <opencv2/core/core.hpp>
// class includes
#include "Sdf.h"
#include "hash_map.h"
#include "sdf_voxel/SdfVoxel.h"

/**
 * class declaration
 */
class MapPixelSdf :  public Sdf {

// variables

    const float voxel_size_;
    const float voxel_size_inv_;

    phmap::parallel_node_hash_map<Vec3i, SdfVoxel,
        phmap::priv::hash_default_hash<Vec3i>,
        phmap::priv::hash_default_eq<Vec3i>,
        Eigen::aligned_allocator<std::pair<const Vec3i, SdfVoxel>>> tsdf_;
    
// methods

    // world point to voxel idx
    Vec3i float2vox(Vec3f point) const {
        Vec3f pv = voxel_size_inv_ * point;
        return Vec3i(std::round(pv[0]), std::round(pv[1]), std::round(pv[2]));
    }
    
    Vec3f vox2float(Vec3i idx) const {
        return voxel_size_ * idx.cast<float>();
    }

    float interp3(Vec3f point, float extrap, Vec3f& grad) const;

    
public:

// constructors / destructor
    
    MapPixelSdf() :
        Sdf(),
        voxel_size_(0.02),
        voxel_size_inv_(1./voxel_size_)
    {}
    
    MapPixelSdf(float voxel_size) :
        Sdf(),
        voxel_size_(voxel_size),
        voxel_size_inv_(1./voxel_size_)
    {}
    
    MapPixelSdf(float voxel_size, float T) :
        Sdf(T),
        voxel_size_(voxel_size),
        voxel_size_inv_(1./voxel_size_)
    {}
    
    ~MapPixelSdf() {}
    
// methods
    
    virtual float tsdf(Vec3f point, Vec3f* grad_ptr) const {
        Vec3f grad = Vec3f::Zero();
        float phi0 = interp3(point, -T_, grad);
        if (grad_ptr) {
            (*grad_ptr)[0] = grad[0];
            (*grad_ptr)[1] = grad[1];
            (*grad_ptr)[2] = grad[2];
        }
        return phi0;
    }
    
    virtual float weights(Vec3f point) const {
        float w = 0;
        Vec3f pv = voxel_size_inv_ * point;
        
        // remove out of volume cases
        const int im = std::floor(pv[0]);
        const int jm = std::floor(pv[1]);
        const int km = std::floor(pv[2]);

        bool A = tsdf_.find(Vec3i(im,jm,km)) != tsdf_.end();
        bool B = tsdf_.find(Vec3i(im,jm,km+1)) != tsdf_.end();
        bool C = tsdf_.find(Vec3i(im,jm+1,km)) != tsdf_.end();
        bool D = tsdf_.find(Vec3i(im,jm+1,km+1)) != tsdf_.end();
        bool E = tsdf_.find(Vec3i(im+1,jm,km)) != tsdf_.end();
        bool F = tsdf_.find(Vec3i(im+1,jm,km+1)) != tsdf_.end();
        bool G = tsdf_.find(Vec3i(im+1,jm+1,km)) != tsdf_.end();
        bool H = tsdf_.find(Vec3i(im+1,jm+1,km+1)) != tsdf_.end();

       if (A && B && C && D && E && F && G && H) {
            const Vec3i idx = float2vox(point);
            auto pair = tsdf_.find(idx);
            w = pair->second.weight;
        }
        return w;
        
    }
    
    virtual void update(const cv::Mat &color, const cv::Mat &depth, const Mat3f K, const SE3 &pose, cv::NormalEstimator<float>* NEst);

// visualization / debugging
    
    virtual bool extract_mesh(std::string filename);
    
    virtual bool extract_pc(std::string filename);

    virtual bool save_sdf(std::string filename);

};

#endif // MAP_PIXEL_SDF_H_
