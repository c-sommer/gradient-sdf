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


#ifndef PHOTOMETRIC_OPTIMIZER_H_
#define PHOTOMETRIC_OPTIMIZER_H_

#include "mat.h"
#include "sdf_voxel/SdfVoxel.h"
#include "sdf_tracker/MapGradPixelSdf.h"

#include <Eigen/StdVector>
#include <opencv2/core/core.hpp>
#include "loss.h"


struct OptSettings {
    int max_it; // maximum number of iterations
    float conv_threshold; // threshold for convergence
    float damping; // LM damping term (often referred to as lambda)
    float lambda; // lambda for weight function (not for LM!)
    float lambda_sq;
    float reg_weight;
    LossFunction loss; // type of (robust) loss
    OptSettings() :
        max_it(25),
        conv_threshold(1e-4),
        damping(1.0),
        lambda(0.5),
        lambda_sq(lambda * lambda),
        reg_weight(10.0),
        loss(LossFunction::CAUCHY)
    {}
};

class PhotometricOptimizer {

// ========== members ==========

    // ..... problem properties .....

    size_t num_voxels_;
    const float voxel_size_;
    const float voxel_size_inv_;
    Mat3f K_; // camera intrinsics
    std::string save_path_;


    OptSettings settings_;

    // ..... variables to be optimized .....

    MapGradPixelSdf* tSDF_;
    std::vector<int> frame_idx_; // indices of keyframes to be taken into account
    std::vector<std::shared_ptr<cv::Mat>> images_; // vector of pointers to keyframes
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > poses_; // poses from tracking part
    std::vector<std::string> key_stamps_;

// ========== private member functions ==========

    // ..... initialization internals .....

    void init(const SdfLrMap& sdf_lr, phmap::parallel_flat_hash_map<Vec3i, std::vector<bool>>& vis_map);

    // ..... pose related .......
    Mat3f getRotation(const int frame_id)
    {
        return poses_[frame_id].topLeftCorner(3,3);
    }

    Vec3f getTranslation(const int frame_id)
    {
        return poses_[frame_id].topRightCorner(3,1);
    }

    cv::Mat getImage(const int frame_id) const
    {
        return *(images_[frame_id]);
    }

    // ..... Jacobian computations .....

    // distance Jacobian
    bool computeJdOneFrame(const Vec3i& idx, const SdfVoxel& voxel, const Mat3f& R, const Vec3f& t, const cv::Mat& img, Vec3f& Jd);
    // camera pose Jacobian
    bool computeJc(const Vec3i& idx, const SdfVoxel& voxel, const cv::Mat& img, const Mat3f& R, const Vec3f& t, Eigen::Matrix<float, 3, 6>& Jc);

    // ..... helper functions for Jacobians .....

    bool getIntensity(const cv::Mat& img, const Vec3i& voxel, const Mat3f& R, const Vec3f& t, Vec3f& intensity);

    // ..... set values for individual voxels .....

    void updateDist(const Vec3i& idx, const float delta_d);

public:

// ========== constructors ==========

    PhotometricOptimizer(MapGradPixelSdf* tSDF,
					const float voxel_size,
                    const Mat3f& K,
                    std::string save_path,
					OptSettings settings = OptSettings());


// ========== public member functions ==========

    // .... initialization functions .....
    void setImages(std::vector<std::shared_ptr<cv::Mat>> images)
    {
        images_ = images;
    }

    void setPoses(std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >& pose)
    {
        poses_ = pose;
    }

    void setKeyframes( std::vector<int>& keyframes)
    {
        frame_idx_ = keyframes;
    }
    void setKeytimestamps(std::vector<std::string>& keystamps)
    {
        key_stamps_ = keystamps;
    }

    // ..... inline get functions .....

    size_t getVoxelNumber() const
    {
        return num_voxels_;
    }

    SdfVoxel getSdf(const Vec3i& voxel) const
    {
        SdfVoxel v(tSDF_->tsdf_.at(voxel));
        return v;
    }

    // ..... optimization and energy computation .....

    float getEnergy();
    void solveDist(float damping = 1.0);
    void solvePose(float damping = 1.0);
    void solvePoseFull(float damping = 1.0);

    bool optimize();

    // ..... debugging .....

    bool savePoses(std::string filename);
};

#endif // PHOTOMETRIC_OPTIMIZER_H_