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


#include "ColorUpsampler.h"
#include "mesh/HrLayeredMarchingCubes.h"

#include <fstream>
// #include <Eigen/Sparse>

using Vec8f = Vec8f;
using Arr8f = Eigen::Array<float, 8, 1>;

// ========== non-class functions ==========

//! from pose matrix to 6-DoF
Eigen::Matrix<float, 1, 7> MatrixTo7DoF(Mat4f& pose){
	Eigen::Matrix<float, 1, 7> xi;
	Vec3f t(pose.topRightCorner(3,1));
	Eigen::Quaternion<float> q(pose.topLeftCorner<3,3>());
	xi << t[0], t[1], t[2],  q.x(), q.y(), q.z(), q.w();
	return xi;
}

template<typename T>
static void printVec(std::vector<T> vec){
	for (const auto& i: vec)
  		std::cout << i << " ";
}


//! check for NaN values inside Eigen object
template<class S> 
bool checkNan(S vector)
{
    return vector.array().isNaN().sum(); 
}

//! interpolation for RGB image
static Vec3f interpolateImage(const float m, const float n, const cv::Mat& img)
{
	int x = std::floor(m);
	int y = std::floor(n);
	cv::Vec3f tmp;
	if ((x+1) < img.rows && (y+1) < img.cols){
		tmp = (y+1.0-n)*(m-x)*img.at<cv::Vec3f>(x+1,y) + (y+1.0-n)*(x+1.0-m)*img.at<cv::Vec3f>(x,y) + (n-y)*(m-x)*img.at<cv::Vec3f>(x+1,y+1) + (n-y)*(x+1.0-m)*img.at<cv::Vec3f>(x,y+1);
	}
	else if ((y+1) < img.cols && x >= img.rows){
		tmp = (y+1.0-n)*img.at<cv::Vec3f>(x,y) + (n-y)*img.at<cv::Vec3f>(x,y+1);
	}
	else if ( y >= img.cols && (x+1) < img.rows){
		tmp = (m-x)*img.at<cv::Vec3f>(x+1,y) + (x+1.0-m)*img.at<cv::Vec3f>(x,y);
	}
	else{
		tmp = img.at<cv::Vec3f>(x,y);
	}

	Vec3f intensity(tmp[2],tmp[1],tmp[0]); // OpenCV stores image colors as BGR.
	return intensity;
}


//! corners of cube [-1, 1]^3
Eigen::Matrix<float, 3, 8> centeredCubeCorners()
{
	Eigen::Matrix<float, 3, 8> corners;
	
	corners.col(0) << -1.0, -1.0, -1.0;
	corners.col(1) << 1.0, -1.0, -1.0;
	corners.col(2) << -1.0, 1.0, -1.0;
	corners.col(3) << 1.0, 1.0, -1.0;
	corners.col(4) << -1.0, -1.0, 1.0;
	corners.col(5) << 1.0, -1.0, 1.0;
	corners.col(6) << -1.0, 1.0, 1.0;
	corners.col(7) << 1.0, 1.0, 1.0;

	return corners;
}


// ========== initialization ==========

//! constructor
ColorUpsampler::ColorUpsampler(const SdfLrMap& sdf_lr,
                            phmap::parallel_flat_hash_map<Vec3i, std::vector<bool>>& vis_map,
                            std::vector<std::shared_ptr<cv::Mat>>& images,
                            std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>& poses,
                            std::vector<int>& frame_idx,
                            const float voxel_size,
                            const Mat3f& K) :
    images_(images),
    poses_(poses),
    frame_idx_(frame_idx),
    voxel_size_(voxel_size),
    voxel_size_inv_(1.f / voxel_size),
    K_(K)
{
    num_frames_ = frame_idx_.size(); // does not work inside colon initializer
    init(sdf_lr, vis_map);
    // selectPoses();
}

//! init
void ColorUpsampler::init(const SdfLrMap& sdf_lr, phmap::parallel_flat_hash_map<Vec3i, std::vector<bool>>& vis_map)
{
	indices_.clear();
	vis_.clear();
    sdf_.clear();
	// convert low-res voxels to high-res voxels
    const float voxel_diameter = std::sqrt(3.) * voxel_size_;
	for (const auto& voxel: sdf_lr) {
        if (std::fabs(voxel.second.dist) < voxel_diameter) { // only accept voxels close to surface
            sdf_.emplace(voxel.first, SdfVoxelHr(voxel.second, voxel_size_));
        }
	}

	num_voxels_ = sdf_.size();
	indices_.resize(num_voxels_);
	vis_.resize(num_voxels_);

	// store indices and visibility vectors in correct order (that of high-res voxel map)
	int count = 0;
	for (const auto& voxel: sdf_) {
		indices_[count] = voxel.first;
		vis_[count] = vis_map[voxel.first];
		vis_[count].resize(frame_idx_.back()+1);
		++count;
    }
}


// ========== helper functions for Jacobians ==========

//! get I_i(R_iv+t) RGB/gray value for 8 subvoxels 
bool ColorUpsampler::getIntensity(const int frame, const Vec3i& idx, const Mat3f& R, const Vec3f& t, Mat3x8f& intensity)
{
	const float fx = K_(0,0);
	const float fy = K_(1,1);
	const float cx = K_(0,2);
	const float cy = K_(1,2);

	SdfVoxelHr voxel(getSdf(idx));

	
	Mat3x8f point = R.transpose()*(getSubvoxelFloat(idx) - voxel.grad * voxel.d.transpose() - t.replicate(1,8));

    auto m = fx * point.row(0).array()/point.row(2).array() + cx;
    auto n = fy * point.row(1).array()/point.row(2).array() + cy;

	//DEBUG
	if (m.hasNaN()|| n.hasNaN()){
		std::cout << "invalid pixel coordinate at voxel "<< idx.transpose() << " at frame " << frame << std::endl; 
		return false;
	}

    const cv::Mat& img = getFrame(frame);

	// Mat3x8f intensity;
	for (size_t i = 0; i < 8; i++){
        if (m(i)<0 || m(i)>=img.cols || n(i)<0 || n(i)>=img.rows) {
			
			return false;
		}
        else {
            intensity.col(i) = interpolateImage(n(i), m(i), img);
		}
	}

	return true;
}



//! coordinates of subvoxel centers
Mat3x8f ColorUpsampler::getSubvoxelFloat(const Vec3i& voxel_in)
{
	Vec3f voxel_float = voxel_in.cast<float>();
    return voxel_size_ * (.25 * centeredCubeCorners() + voxel_float.replicate<1,8>());
}

// ========== set values for individual voxels ==========

//! set albedo
void ColorUpsampler::setAlbedo(const Vec3i& idx, const Vec8f& r, const Vec8f& g, const Vec8f& b)
{
	Vec8f rr, gg, bb;
	rr = r;
	rr = rr.cwiseMax(0.0);
	rr = rr.cwiseMin(1.0);

	gg = g;
	gg = gg.cwiseMax(0.0);
	gg = gg.cwiseMin(1.0);

	bb = b;
	bb = bb.cwiseMax(0.0);
	bb = bb.cwiseMin(1.0);

	sdf_.at(idx).r = rr;
	sdf_.at(idx).g = gg;
	sdf_.at(idx).b = bb;

}


//! extract mesh from SDF to debug geometry
bool ColorUpsampler::extractMesh(std::string filename)
{
    HrLayeredMarchingCubes lmc(Vec3f(voxel_size_, voxel_size_, voxel_size_));
    lmc.computeIsoSurface(&sdf_);
    bool success = lmc.savePly(filename + ".ply");
	if (success)
		std::cout << "Mesh " << filename << ".ply successfully saved." << std::endl;

	return success;
}

bool ColorUpsampler::extractCloud(std::string filename)
{

	const float voxel_size_4 = .25 * voxel_size_;
	filename += ".ply";

	int voxel_id = 0;
    std::vector<Eigen::Matrix<float, 9, 1>> points_normals_colors;
    for (const auto& el : sdf_) {

		std::vector<bool> vis = vis_[voxel_id];
		bool visible = false;
		for (size_t i = 0; i < num_frames_; i++){
			if (vis[frame_idx_[i]]){
				visible = true;
				break;
			}	
		}

		if(!visible){
			voxel_id++;
			continue;
		}

        const SdfVoxelHr& v = el.second;
        if (v.weight < 5){
			voxel_id++;
            continue;
		}

		Mat3x8f voxel_normal = -v.grad.normalized().replicate(1,8);
		Mat3x8f voxel_float = getSubvoxelFloat(el.first);
		Mat3x8f distances = voxel_normal * v.d.asDiagonal();
		for (int i=0; i<8; ++i)
		{
			Vec3f d = distances.col(i);
			if (std::fabs(d[0]) < voxel_size_4 && std::fabs(d[1]) < voxel_size_4 && std::fabs(d[2]) < voxel_size_4 && !std::isnan(v.r[i]) && !std::isnan(v.g[i]) && !std::isnan(v.b[i]))
			{
				Eigen::Matrix<float, 9, 1> pnc;
				pnc.segment<3>(0) = voxel_float.col(i) + d;
				pnc.segment<3>(3) = voxel_normal.col(i);
				pnc.segment<3>(6) = Vec3f(v.r[i], v.g[i], v.b[i]);
				points_normals_colors.push_back(pnc);
			}
		}
		voxel_id++;
    }    

    std::ofstream plyFile;
    plyFile.open(filename.c_str());
    if (!plyFile.is_open())
        return false;
        
    plyFile << "ply" << std::endl;
    plyFile << "format ascii 1.0" << std::endl;
    plyFile << "element vertex " << points_normals_colors.size() << std::endl;
    plyFile << "property float x" << std::endl;
    plyFile << "property float y" << std::endl;
    plyFile << "property float z" << std::endl;
    plyFile << "property float nx" << std::endl;
    plyFile << "property float ny" << std::endl;
    plyFile << "property float nz" << std::endl;
    plyFile << "property uchar red" << std::endl;
    plyFile << "property uchar green" << std::endl;
    plyFile << "property uchar blue" << std::endl;
    plyFile << "end_header" << std::endl;
    
    for (const Eigen::Matrix<float, 9, 1>& p : points_normals_colors)
	{
        plyFile << p[0] << " " << p[1] << " " << p[2] << " " << p[3] << " " << p[4] << " " << p[5] << " " << int(255 * p[6]) << " " << int(255 * p[7]) << " " << int(255 * p[8]) << std::endl;
    }
    
    plyFile.close();

	std::cout << "Cloud " << filename << " successfully saved." << std::endl;
    return true;
}


 // ------------------- solvers -----------------------------------------------------------------------------------------------------------------


//! compute albedo from ambient light assumption, i.e. average over all observations
void ColorUpsampler::computeColor()
{
	// size_t num_tot_frames = getVis(0).size();

	size_t count = 0; // counter (same for all three color channels)
    Vec8f br, bb, bg;
    int voxel_id = 0;

	for (const auto& idx : indices_) {

		br.setZero();
		bb.setZero();
		bg.setZero();

		std::vector<bool> vis = getVis(voxel_id);
		SdfVoxelHr voxel(getSdf(idx));
		
		for(size_t i = 0; i < num_frames_; i++){
			Mat3f R = getRotation(i);
			Vec3f t = getTranslation(i);
			
            if(vis[frame_idx_[i]]) {
				// r(v) = \sum_i (I(pi(Rv+t)) - pho(v)(<n(v),Rl>))
				Mat3x8f intensity;
				if(!getIntensity(i, idx, R, t, intensity)){
					continue;
				}
				br += intensity.row(0);
				bg += intensity.row(1);
				bb += intensity.row(2);
				++count;
			}
		}

		Vec8f r,g,b;
		float inv_count = 1.f / float(count);
		r = inv_count * br;
		g = inv_count * bg;
		b = inv_count * bb;
		setAlbedo(idx, r, g, b); // cut-off happens here
		count = 0;
		voxel_id++;
	}
}

