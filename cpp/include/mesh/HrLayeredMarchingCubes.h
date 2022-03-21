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


#ifndef HR_LAYERED_MARCHING_CUBES_H
#define HR_LAYERED_MARCHING_CUBES_H

#include <vector>

#include "mat.h"

#include <map>

#include "sdf_voxel/SdfVoxel.h"

// class of high resolution layered marching cubes with colors.
class HrLayeredMarchingCubes
{
public:

    using voxel_phmap = phmap::parallel_node_hash_map<Vec3i, SdfVoxelHr,
                                                    phmap::priv::hash_default_hash<Vec3i>,
                                                    phmap::priv::hash_default_eq<Vec3i>,
                                                    Eigen::aligned_allocator<std::pair<const Vec3i, SdfVoxelHr>>>;

    // HrLayeredMarchingCubes(const Vec3i &dimensions, const Vec3f &size);

    HrLayeredMarchingCubes(const Vec3f &voxelSize);

    ~HrLayeredMarchingCubes();

    bool computeIsoSurface(const voxel_phmap* sdf_map, float isoValue = 0.0f);

    bool savePly(const std::string &filename) const;

protected:

    static int edgeTable[256];
    
    static int triTable[256][16];

    inline int computeLutIndex(int i, int j, int k, float isoValue);

    void copyLayer(int z, const voxel_phmap* sdf_map);

    inline void zeroWeights(int x, int y, int z);

    inline void copyCube(int x, int y, int z, const float w,
                        const Eigen::Matrix<float, 8, 1>& d,
                        const Eigen::Matrix<float, 8, 1>& r,
                        const Eigen::Matrix<float, 8, 1>& g,
                        const Eigen::Matrix<float, 8, 1>& b
                        );

    inline void setVoxel(const size_t off, const size_t corner, const float w,
                        const Eigen::Matrix<float, 8, 1>& d,
                        const Eigen::Matrix<float, 8, 1>& r,
                        const Eigen::Matrix<float, 8, 1>& g,
                        const Eigen::Matrix<float, 8, 1>& b
                        );

    Vec3f interpolate(float tsdf0, float tsdf1, const Vec3f &val0, const Vec3f &val1, float isoValue);

    Vec3f getVertex(int i1, int j1, int k1, int i2, int j2, int k2, float isoValue);

    Vec3b getColor(int x1, int y1, int z1, int x2, int y2, int z2, float isoValue);

    void computeTriangles(int cubeIndex, const Vec3f edgePoints[12], const Vec3b edgeColors[12]);

    inline unsigned int addVertex(const Vec3f &v, const Vec3b &c);

    Vec3f voxelToWorld(int i, int j, int k) const;

    std::vector<Vec3f> vertices_;
    std::vector<Vec3b> colors_;
    std::vector<Vec3i> faces_;
    Vec3i dim_;
    Vec3f size_;
    Vec3f voxelSize_;
    Vec3f origin_;
    Vec3i min_;

    // layers
    float* tsdf_;
    float* weights_;
    unsigned char* red_;
    unsigned char* green_;
    unsigned char* blue_;
};

#endif
