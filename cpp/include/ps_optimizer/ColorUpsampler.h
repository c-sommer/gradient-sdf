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


#ifndef COLOR_UPSAMPLER_H_
#define COLOR_UPSAMPLER_H_

#include "mat.h"
#include "sdf_voxel/SdfVoxel.h"

#include <Eigen/StdVector>
#include <opencv2/core/core.hpp>
#include "loss.h"

using Mat3x8f = Eigen::Matrix<float, 3, 8>;
using Mat8f = Eigen::Matrix<float, 8, 8>;

class ColorUpsampler {

// ========== members ==========

    // ..... problem properties .....

    size_t num_frames_;
    size_t num_voxels_;
    const float voxel_size_;
    const float voxel_size_inv_;
    Mat3f K_; // camera intrinsics

    std::vector<int> frame_idx_; // indices of keyframes to be taken into account
    std::vector<Vec3i> indices_; // 3D integer indices of voxels, stored in vector
    std::vector<std::vector<bool>> vis_; // visibility per voxel and frame

    std::vector<std::shared_ptr<cv::Mat>> images_; // vector of pointers to keyframes

    // ..... variables to be optimized .....

    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > poses_;

    SdfHrMap sdf_; // SDF map

// ========== private member functions ==========

    // ..... initialization internals .....

    void init(const SdfLrMap& sdf_lr, phmap::parallel_flat_hash_map<Vec3i, std::vector<bool>>& vis_map);

    bool getIntensity(const int frame, const Vec3i& voxel, const Mat3f& R, const Vec3f& t, Mat3x8f& intensity);
    Mat3x8f getSubvoxelFloat(const Vec3i& voxel_in);
    
    void setAlbedo(const Vec3i& idx, const Vec8f& r, const Vec8f& g, const Vec8f& b);

public:

// ========== constructors ==========
    
    ColorUpsampler(const SdfLrMap& sdf_lr,
					phmap::parallel_flat_hash_map<Vec3i, std::vector<bool>>& vis_map,
					std::vector<std::shared_ptr<cv::Mat>>& images,
					std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>& poses,
					std::vector<int>& frame_idx,
					const float voxel_size,
                    const Mat3f& K);


// ========== public member functions ==========

    // ..... inline get functions .....

    size_t getFrameNumber() const
    {
        return num_frames_;
    }

    size_t getVoxelNumber() const
    {
        return num_voxels_;
    }

    Eigen::Matrix3f getRotation(int frame_id) const
    {
        return poses_[frame_id].topLeftCorner(3,3);
    }

    Vec3f getTranslation(int frame_id) const
    {
        return poses_[frame_id].topRightCorner(3,1);
    }

    cv::Mat getFrame(const int frame) const
    {
        return *(images_[frame]);
    }

    std::vector<bool> getVis(int voxel_id) const
    {
        return vis_[voxel_id];
    }

    SdfVoxelHr getSdf(const Vec3i& voxel) const
    {
        SdfVoxelHr v(sdf_.at(voxel));
        return v;
    }

    
    void computeColor();
    bool extractMesh(std::string filename);
    bool extractCloud(std::string filename);
    
};

#endif // COLOR_UPSAMPLER_H_