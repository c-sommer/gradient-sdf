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


#include "MapPixelSdf.h"
#include "normals/NormalEstimator.h"
#include "mesh/LayeredMarchingCubesNoColor.h"
#include <fstream>

float MapPixelSdf::interp3(Vec3f point, float extrap, Vec3f& grad) const{
    Vec3f pv = voxel_size_inv_ * point;
    const float i = pv[0], j= pv[1], k = pv[2];
    
    // remove out of volume cases
    const int im = std::floor(i);
    const int jm = std::floor(j);
    const int km = std::floor(k);


    const float dx = i-im;
    const float dy = j-jm;
    const float dz = k-km;

    // get 8 corner
    Eigen::Matrix<float, 8, 1> d = extrap*Eigen::Matrix<float, 8, 1>::Ones();
    std::vector<bool> corner;
    for(size_t k0 = 0; k0 < 2; k0++)for(size_t j0 = 0; j0 < 2; j0++)for(size_t i0 = 0; i0 < 2; i0++){
        Vec3i idx(im+i0, jm+j0 , km+k0);
        auto pair = tsdf_.find(idx);
        if (pair != tsdf_.end()){
            d[i0 + j0*2 + k0*2*2] = pair->second.dist;
            corner.push_back(true);
        }
        else
            corner.push_back(false);
    }

    if (std::all_of(corner.begin(), corner.end(), [](bool v) { return !v; })) // if all false
        return extrap;
    
    if (std::all_of(corner.begin(), corner.end(), [](bool v) { return v; })){ //if all true
        // std::cout << "8 corner d: " << d.transpose() << std::endl;
        // interpolate in x direction
        const float d01 = (1-dx)*d[0] + dx*d[1];
        const float d23 = (1-dx)*d[2] + dx*d[3];
        const float d45 = (1-dx)*d[4] + dx*d[5];
        const float d67 = (1-dx)*d[6] + dx*d[7];

        // interpolate in y direction
        const float d02 = (1-dy)*d[0] + dy*d[2];
        const float d13 = (1-dy)*d[1] + dy*d[3];
        const float d46 = (1-dy)*d[4] + dy*d[6];
        const float d57 = (1-dy)*d[5] + dy*d[7];

        // interpolate in z direction
        const float d04 = (1-dz)*d[0] + dz*d[4];
        const float d15 = (1-dz)*d[1] + dz*d[5];
        const float d26 = (1-dz)*d[2] + dz*d[6];
        const float d37 = (1-dz)*d[3] + dz*d[7];

        // calculate gradient
        grad[0] = voxel_size_inv_*((1-dz)*d13 + dz*d57 - (1-dz)*d02 - dz*d46);
        grad[1] = voxel_size_inv_*((1-dz)*d23 + dz*d67 - (1-dz)*d01 - dz*d45);
        grad[2] = voxel_size_inv_*((1-dy)*d45 + dy*d67 - (1-dy)*d01 - dy*d23);


        // interpolate in  y direction
        const float dy0 = (1-dy)*d01 + dy*d23;
        const float dy1 = (1-dy)*d45 + dy*d67;

        //interpolate in z direction
        return (1-dz)*dy0 + dz*dy1;
    }

    // if not all 8 exists
    return 0.0;
    
}


void MapPixelSdf::update(const cv::Mat &color, const cv::Mat &depth, const Mat3f K, const SE3 &pose, cv::NormalEstimator<float>* NEst) {

    const float fx = K(0,0), fy = K(1,1);
    const float cx = K(0,2), cy = K(1,2);
    
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
    
    size_t counter = 0; 
    
    for (size_t m=0; m < depth.rows; ++m) for (size_t n=0; n < depth.cols; ++n) {
    
        const size_t idx = m * depth.cols + n;
        
        const float z = z_ptr[idx];
        
        if (z <= z_min_ || z >= z_max_ ) // z out of range or unreliable z
            continue;

        const Vec3f xy_hom(x_hom_ptr[idx], y_hom_ptr[idx], 1.);
        const Vec3f R_xy_hom(z * R * xy_hom + t);
        const Vec3f normal(nx_ptr[idx], ny_ptr[idx], nz_ptr[idx]);
        
        if (normal.squaredNorm() < .1) // invalid normal
            continue;
        
        if (normal.dot(xy_hom) * normal.dot(xy_hom) * hom_inv_ptr[idx] < .25) // normal direction too far from viewing ray direction (>75.5°)
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
            }
        }
        
        ++counter; 
    
    }
    
    std::cout << counter << " points integrated into tSDF volume!" << std::endl; 
}

bool MapPixelSdf::extract_mesh(std::string filename) {

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

bool MapPixelSdf::extract_pc(std::string filename) {

	const float voxel_size_2 = .5 * voxel_size_;

    std::vector<Vec3f> points;
    for (const auto& el : tsdf_) {
        const SdfVoxel& v = el.second;
        if (v.weight < 5)
            continue;
        if(std::fabs(v.dist) < std::sqrt(3)*voxel_size_)
        {
            Vec3f pn;
            pn.segment<3>(0) = vox2float(el.first);
            points.push_back(pn);
        }
    }    

    std::ofstream plyFile;
    plyFile.open(filename.c_str());
    if (!plyFile.is_open())
        return false;
        
    plyFile << "ply" << std::endl;
    plyFile << "format ascii 1.0" << std::endl;
    plyFile << "element vertex " << points.size() << std::endl;
    plyFile << "property float x" << std::endl;
    plyFile << "property float y" << std::endl;
    plyFile << "property float z" << std::endl;
    
    plyFile << "end_header" << std::endl;
    
    for (const Vec3f& p : points) {
        plyFile << p[0] << " " << p[1] << " " << p[2] << std::endl;
    }
    
    plyFile.close();

    return true;
}

bool MapPixelSdf::save_sdf(std::string filename)
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


    if (!file.is_open() || !weight_file.is_open()){
        std::cerr << "couldn't save sdf or sdf weight file!" << std::endl;
        return false;
    }

    for(const auto& pair : tsdf_){
        Vec3i idx = pair.first;
        const SdfVoxel& v = pair.second;
        int lin_idx = dim[0]*dim[1]*(idx[2]-zmin) + dim[0]*(idx[1]-ymin) + idx[0]-xmin;
        file << lin_idx << " " << v.dist << "\n";
        weight_file << lin_idx << " " << v.weight << "\n";
    }

    file.close();
    weight_file.close();
    return true;
}
