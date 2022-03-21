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


#ifndef SDF_VOXEL_H_
#define SDF_VOXEL_H_

#include <vector>
#include <Eigen/Dense>
#include "hash_map.h"

struct SdfVoxel {
EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    float dist;
    Eigen::Vector3f grad;
    float weight;

    SdfVoxel() :
        dist(0.),
        grad(Eigen::Vector3f::Zero()),
        weight(0.)
    {}
};

using Vec8f = Eigen::Matrix<float, 8, 1>;

struct SdfVoxelHr {

EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Vec8f d = Vec8f::Zero();
    Vec8f r = Vec8f::Zero();
    Vec8f g = Vec8f::Zero();
    Vec8f b = Vec8f::Zero();

    const float dist;
    const float weight;
    const Eigen::Vector3f grad;

    SdfVoxelHr() :
        dist(0.),
        grad(Eigen::Vector3f::Zero()),
        weight(0.),
        d(Vec8f::Zero()),
        r(Vec8f::Zero()),
        g(Vec8f::Zero()),
        b(Vec8f::Zero())
    {}

    SdfVoxelHr(const SdfVoxel &voxel, const float voxel_size) :
        dist(voxel.dist),
        grad(voxel.grad.normalized()),
        weight(voxel.weight),
        r(Vec8f::Zero()),
        g(Vec8f::Zero()),
        b(Vec8f::Zero())
    {
        const float voxel_size_4 = 0.25 * voxel_size;
        d[0] = dist + voxel_size_4 * (-grad[0] - grad[1] - grad[2]);
        d[1] = dist + voxel_size_4 * ( grad[0] - grad[1] - grad[2]);
        d[2] = dist + voxel_size_4 * (-grad[0] + grad[1] - grad[2]);
        d[3] = dist + voxel_size_4 * ( grad[0] + grad[1] - grad[2]);
        d[4] = dist + voxel_size_4 * (-grad[0] - grad[1] + grad[2]);
        d[5] = dist + voxel_size_4 * ( grad[0] - grad[1] + grad[2]);
        d[6] = dist + voxel_size_4 * (-grad[0] + grad[1] + grad[2]);
        d[7] = dist + voxel_size_4 * ( grad[0] + grad[1] + grad[2]);
    }

    SdfVoxelHr(const SdfVoxelHr &voxel) :
        dist(voxel.dist),
        grad(voxel.grad),
        weight(voxel.weight),
        d(voxel.d),
        r(voxel.r),
        g(voxel.g),
        b(voxel.b)
    {}

};

using Vec3i = Eigen::Vector3i;
using SdfLrMap = phmap::parallel_node_hash_map<Vec3i, SdfVoxel,   phmap::priv::hash_default_hash<Vec3i>, phmap::priv::hash_default_eq<Vec3i>, Eigen::aligned_allocator<std::pair<const Vec3i, SdfVoxel>>>;
using SdfHrMap = phmap::parallel_node_hash_map<Vec3i, SdfVoxelHr, phmap::priv::hash_default_hash<Vec3i>, phmap::priv::hash_default_eq<Vec3i>, Eigen::aligned_allocator<std::pair<const Vec3i, SdfVoxelHr>>>;

#endif // SDF_VOXEL_H_

