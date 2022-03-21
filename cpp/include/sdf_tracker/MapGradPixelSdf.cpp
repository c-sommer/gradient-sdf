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


#include "MapGradPixelSdf.h"
#include "normals/NormalEstimator.h"
#include "mesh/LayeredMarchingCubesNoColor.h"
#include <fstream>

void MapGradPixelSdf::update(const cv::Mat &color, const cv::Mat &depth, const Mat3f K, const SE3 &pose, cv::NormalEstimator<float>* NEst) {

    const float fx = K(0,0), fy = K(1,1);
    const float cx = K(0,2), cy = K(1,2);
    
    const float z_min = z_min_, z_max = z_max_;
    
    size_t lin_index = 0;
    
    cv::Mat nx, ny, nz, med_depth;
    cv::medianBlur(depth, med_depth, 5); // median filtering
    
    if (!NEst) { 
        std::cerr << "No normal estimation possible - cannot update SDF volume!" << std::endl;
        return;
    }
    
    NEst->compute(depth, nx, ny, nz);
    
    const Mat3f R = pose.rotationMatrix();
    const Mat3f Rt = pose.rotationMatrix().transpose();
    const Vec3f t = pose.translation();
    
    cv::Mat* x0_ptr = NEst->x0_ptr();
    cv::Mat* y0_ptr = NEst->y0_ptr();
    cv::Mat* n_sq_inv_ptr = NEst->n_sq_inv_ptr();
    const float* x_hom_ptr = (const float*)x0_ptr->data;
    const float* y_hom_ptr = (const float*)y0_ptr->data;
    const float* hom_inv_ptr = (const float*)n_sq_inv_ptr->data;
    const float* z_ptr = (const float*)depth.data;
    const float* zm_ptr = (const float*)med_depth.data;
    
    const float* nx_ptr = (const float*)nx.data;
    const float* ny_ptr = (const float*)ny.data;
    const float* nz_ptr = (const float*)nz.data;
    
    const int factor = std::floor(T_ / voxel_size_);
    
    for (size_t m=0; m < depth.rows; ++m) for (size_t n=0; n < depth.cols; ++n) {
    
        const size_t idx = m * depth.cols + n;
        
        const float z = z_ptr[idx];
        
        if (z <= z_min || z >= z_max ) // z out of range or unreliable z
            continue;

        const Vec3f xy_hom(x_hom_ptr[idx], y_hom_ptr[idx], 1.);
        const Vec3f R_xy_hom(R * xy_hom);
        const Vec3f normal(nx_ptr[idx], ny_ptr[idx], nz_ptr[idx]);
        const Vec3f Rn(R * normal);
        
        if (normal.squaredNorm() < .1) // invalid normal
            continue;
        
        if (normal.dot(xy_hom) * normal.dot(xy_hom) * hom_inv_ptr[idx] < .25) // normal direction too far from viewing ray direction (>72.5°)
            continue;
        
        for (float k = -factor; k <= factor; ++k) { // loop along ray
        
            Vec3f point = (z + k*voxel_size_) * R_xy_hom + t; // convert point into Sdf coordinate system
            const Vec3i vi = float2vox(point);
            point = Rt * (vox2float(vi) - t);
            const float sdf = point[2] - z;
            const float w = weight(sdf);
            if (w>0) {
                SdfVoxel& v = tsdf_[vi];
                v.weight += w;
                v.dist += (truncate(sdf) - v.dist) * w / v.weight;
                v.grad += w * Rn; // normals are inward-pointing!
                std::vector<bool>& vis = vis_[vi];
                vis.resize(counter_);
                vis.push_back(true);
            }
        }
    }

    increase_counter();
    std::cout << "Current frame counter: " << counter_ << std::endl; // DEBUG
}

bool MapGradPixelSdf::extract_mesh(std::string filename) {

    // compute dimensions (and, from that, size)
    const int pos_inf = std::numeric_limits<int>::max();
    const int neg_inf = std::numeric_limits<int>::min();
    int xmin, xmax, ymin, ymax, zmin, zmax;
    xmin = pos_inf;
    xmax = neg_inf;
    ymin = pos_inf;
    ymax = neg_inf;
    zmin = pos_inf;
    zmax = neg_inf;
    for (auto v : tsdf_) {
        if (v.first[0] < xmin) xmin = v.first[0];
        if (v.first[0] > xmax) xmax = v.first[0];
        if (v.first[1] < ymin) ymin = v.first[1];
        if (v.first[1] > ymax) ymax = v.first[1];
        if (v.first[2] < zmin) zmin = v.first[2];
        if (v.first[2] > zmax) zmax = v.first[2];
    }

    // create input that can handle MarchingCubes class
    const Vec3i dim(xmax - xmin + 1, ymax - ymin + 1, zmax - zmin + 1);
    const size_t num_voxels = dim[0] * dim[1] * dim[2];
    float* dist = new float[num_voxels];
    float* weights = new float[num_voxels];
    size_t lin_index = 0;
    for (int k=zmin; k<=zmax; ++k) for (int j=ymin; j<=ymax; ++j) for (int i=xmin; i<=xmax; ++i) {
        Vec3i idx(i, j, k);
        auto pair = tsdf_.find(idx);
        if (pair != tsdf_.end()) {
            dist[lin_index] = pair->second.dist;
            weights[lin_index] = pair->second.weight;
        }
        else {
            dist[lin_index] = T_;
            weights[lin_index] = 0;
        }
        ++lin_index;
    }
    
    // call marching cubes
    LayeredMarchingCubesNoColor lmc(Vec3f(voxel_size_, voxel_size_, voxel_size_));
    lmc.computeIsoSurface(&tsdf_);
    bool success = lmc.savePly(filename);
    
    // delete temporary arrays  
    delete[] dist;
    delete[] weights;
    
    return success;
}

bool MapGradPixelSdf::extract_pc(std::string filename) {

	const float voxel_size_2 = .5 * voxel_size_;

    std::vector<Vec6f> points_normals;
    for (const auto& el : tsdf_) {
        const SdfVoxel& v = el.second;
        if (v.weight < 5)
            continue;
        Vec3f g = 1.2*v.grad.normalized();
        Vec3f d = v.dist * g;
        if (std::fabs(d[0]) < voxel_size_2 && std::fabs(d[1]) < voxel_size_2 && std::fabs(d[2]) < voxel_size_2)
        {
            Vec6f pn;
            pn.segment<3>(0) = vox2float(el.first) - d;
            pn.segment<3>(3) = -g;
            points_normals.push_back(pn);
        }
    }    

    std::ofstream plyFile;
    plyFile.open(filename.c_str());
    if (!plyFile.is_open())
        return false;
        
    plyFile << "ply" << std::endl;
    plyFile << "format ascii 1.0" << std::endl;
    plyFile << "element vertex " << points_normals.size() << std::endl;
    plyFile << "property float x" << std::endl;
    plyFile << "property float y" << std::endl;
    plyFile << "property float z" << std::endl;
    plyFile << "property float nx" << std::endl;
    plyFile << "property float ny" << std::endl;
    plyFile << "property float nz" << std::endl;
    plyFile << "end_header" << std::endl;
    
    for (const Vec6f& p : points_normals) {
        plyFile << p[0] << " " << p[1] << " " << p[2] << " " << p[3] << " " << p[4] << " " << p[5] << std::endl;
    }
    
    plyFile.close();

    return true;
}

bool MapGradPixelSdf::save_sdf(std::string filename)
{
    // compute dimensions (and, from that, size)
    const int pos_inf = std::numeric_limits<int>::max();
    const int neg_inf = std::numeric_limits<int>::min();
    int xmin, xmax, ymin, ymax, zmin, zmax;
    xmin = pos_inf;
    xmax = neg_inf;
    ymin = pos_inf;
    ymax = neg_inf;
    zmin = pos_inf;
    zmax = neg_inf;
    for (auto v : tsdf_) {
        if (v.first[0] < xmin) xmin = v.first[0];
        if (v.first[0] > xmax) xmax = v.first[0];
        if (v.first[1] < ymin) ymin = v.first[1];
        if (v.first[1] > ymax) ymax = v.first[1];
        if (v.first[2] < zmin) zmin = v.first[2];
        if (v.first[2] > zmax) zmax = v.first[2];
    }
  

    // create input that can handle MarchingCubes class
    const Vec3i dim(xmax - xmin + 1, ymax - ymin + 1, zmax - zmin + 1);
    const size_t num_voxels = dim[0] * dim[1] * dim[2];

    // --------------save grid info ----------------------------
    std::ofstream grid_file;
    grid_file.open((filename + "_grid_info.txt").c_str());
    if (!grid_file.is_open()){
        std::cerr << "couldn't save grid_info file!" << std::endl;
        return false;
    }
    grid_file << "voxel size: " << voxel_size_ << std::endl 
    << "voxel dim: " << dim[0] << " " << dim[1] << " " << dim[2] << std::endl
    << "voxel min: " << xmin << " " << ymin << " " << zmin << std::endl
    << "voxel max: " << xmax << " " << ymax << " " << zmax << std::endl;
    grid_file.close();

    // ---------------- save sdf dist -------------------------------
    std::ofstream file;
    std::ofstream weight_file;
    file.open((filename + "_sdf_d.txt").c_str());
    weight_file.open((filename + "_sdf_weight.txt").c_str());

    // -------------- save sdf grad ------------
    std::ofstream sdf_n0, sdf_n1, sdf_n2;
    sdf_n0.open((filename + "_sdf_n0.txt").c_str());
    sdf_n1.open((filename + "_sdf_n1.txt").c_str());
    sdf_n2.open((filename + "_sdf_n2.txt").c_str());

    if (!file.is_open() || !weight_file.is_open() || !sdf_n0.is_open() || !sdf_n1.is_open() || !sdf_n2.is_open()){
        std::cerr << "couldn't save sdf or sdf weight file!" << std::endl;
        return false;
    }

    for(const auto& pair : tsdf_){
        Vec3i idx = pair.first;
        const SdfVoxel& v = pair.second;
        int lin_idx = dim[0]*dim[1]*(idx[2]-zmin) + dim[0]*(idx[1]-ymin) + idx[0]-xmin;
        file << lin_idx << " " << v.dist << "\n";
        weight_file << lin_idx << " " << v.weight << "\n";
        sdf_n0 << lin_idx << " " << v.grad[0] << "\n";
        sdf_n1 << lin_idx << " " << v.grad[1] << "\n";
        sdf_n2 << lin_idx << " " << v.grad[2] << "\n";
        
    }

    file.close();
    weight_file.close();
    sdf_n0.close();
    sdf_n1.close();
    sdf_n2.close();
    return true;
}
