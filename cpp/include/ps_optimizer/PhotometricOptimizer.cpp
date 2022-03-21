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


#include <fstream>
#include "PhotometricOptimizer.h"
#include "Timer.h"

// ========== non-class functions ==========

//! skew-symmetric matrix of a 3D vector
static inline Mat3f skew(Vec3f v) {
	Mat3f vx = Mat3f::Zero();
	vx(0,1) = -v[2];
	vx(0,2) =  v[1];
	vx(1,0) =  v[2];
	vx(1,2) = -v[0];
	vx(2,0) = -v[1];
	vx(2,1) =  v[0];
	return vx;
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

//! gradient of image
static Vec3f computeImageGradient(const float m, const float n, const cv::Mat& img, int direction)
{
	float w00, w01, w10, w11;

	int x = std::floor(m);
	int y = std::floor(n);

	w01 = m - x;
	w11 = n - y;
	w00 = 1.0 - w01;
	w10 = 1.0 - w11;

    // compute gradient manually using finite differences
    
    cv::Vec3f v0, v1;

    if (direction == 0)
    {
		// x-direction
		if( (x+1)<img.rows && (y+1) < img.cols){
		
        	v0 = img.at<cv::Vec3f>(x,y+1) - img.at<cv::Vec3f>(x,y);
        	v1 = img.at<cv::Vec3f>(x+1,y+1) - img.at<cv::Vec3f>(x+1,y);
      
        	return w00 * Vec3f(v0[2],v0[1],v0[0]) + w01 *  Vec3f(v1[2],v1[1],v1[0]);
		}
		else if ( (x+1) >= img.rows){// && (y+1) < img.cols){

			v0 = img.at<cv::Vec3f>(x,y+1) - img.at<cv::Vec3f>(x,y);

			return Vec3f(v0[2],v0[1],v0[0]);
		}
		else if ( (x+1) < img.rows && (y+1) >= img.cols){

			v0 = -img.at<cv::Vec3f>(x,y-1) + img.at<cv::Vec3f>(x,y);
			v1 = -img.at<cv::Vec3f>(x+1,y-1) + img.at<cv::Vec3f>(x+1,y);

			return w00 * Vec3f(v0[2],v0[1],v0[0]) + w01 *  Vec3f(v1[2],v1[1],v1[0]);
		}
    }
    else
    {
        // y-direction
		if ((x+1)<img.rows && (y+1) < img.cols){
        	v0 = img.at<cv::Vec3f>(x+1,y) - img.at<cv::Vec3f>(x,y);
        	v1 = img.at<cv::Vec3f>(x+1,y+1) - img.at<cv::Vec3f>(x,y+1);
        	return w10 * Vec3f(v0[2],v0[1],v0[0]) + w11 * Vec3f(v1[2],v1[1],v1[0]);
		}

		else if ( (x+1) >= img.rows && (y+1) < img.cols){
			v0 = -img.at<cv::Vec3f>(x-1,y) + img.at<cv::Vec3f>(x,y);
        	v1 = -img.at<cv::Vec3f>(x-1,y+1) + img.at<cv::Vec3f>(x,y+1);
			return w10 * Vec3f(v0[2],v0[1],v0[0]) + w11 * Vec3f(v1[2],v1[1],v1[0]);
		}
		else if ( (y+1) >= img.cols){ // && (x+1) < img.rows){
			v0 = img.at<cv::Vec3f>(x+1,y) - img.at<cv::Vec3f>(x,y);
			return Vec3f(v0[2],v0[1],v0[0]);
		}
    }
}


// ========== initialization ==========

//! constructor
PhotometricOptimizer::PhotometricOptimizer(MapGradPixelSdf* tSDF,
                            const float voxel_size,
                            const Mat3f& K,
							std::string save_path,
                            OptSettings settings):
    tSDF_(tSDF),
    voxel_size_(voxel_size),
    voxel_size_inv_(1.f / voxel_size),
    K_(K),
	save_path_(save_path),
    settings_(settings)
{}

// ========== Jacobian computations ==========

//! Jacobian w.r.t. SDF when gradient norm treated as constant
bool PhotometricOptimizer::computeJdOneFrame(const Vec3i& idx, const SdfVoxel& voxel,  const Mat3f& R, const Vec3f& t, const cv::Mat& img, Vec3f& Jd)
{	
	const float fx = K_(0,0);
	const float fy = K_(1,1);
	const float cx = K_(0,2);
	const float cy = K_(1,2);

	Mat3f Rt = R.transpose();
	
    Vec3f point = Rt * (tSDF_->vox2float(idx) - voxel.dist * voxel.grad.normalized() - t);
    const float z_inv = 1. / point[2];
    const float z_inv_sq = z_inv * z_inv;
	float m = fx * point[0] * z_inv + cx;
    float n = fy * point[1] * z_inv + cy;

    if (m<0 || m>=img.cols || n<0 || n>=img.rows) {
        return false;
    }

	// Mat3x8f Rtn = Rt*getSubvoxelGrad(voxel.d);
	Vec3f Rtn = -Rt*voxel.grad;

    Eigen::Matrix<float,3,2> image_grad;
    image_grad.col(0) = computeImageGradient(n, m, img, 0); //3x2
    image_grad.col(1) = computeImageGradient(n, m, img, 1);

    Eigen::Matrix<float, 2, 3> pi_grad = Eigen::Matrix<float, 2, 3>::Zero();
    pi_grad(0,0) = fx * z_inv;
    pi_grad(0,2) = -fx * point[0] * z_inv_sq;
    pi_grad(1,1) = fy * z_inv;
    pi_grad(1,2) = -fy * point[1] * z_inv_sq;

    Jd = image_grad * pi_grad * Rtn;

	return true;
}


//! Jacobian w.r.t. camera poses
bool PhotometricOptimizer::computeJc(const Vec3i& idx, const SdfVoxel& voxel, const cv::Mat& img, const Mat3f& R, const Vec3f& t, Eigen::Matrix<float, 3, 6>& Jc)
{
	const float fx = K_(0,0);
	const float fy = K_(1,1);
	const float cx = K_(0,2);
	const float cy = K_(1,2);

	Mat3f Rt = R.transpose();
    Vec3f point = Rt * (tSDF_->vox2float(idx) - voxel.dist * voxel.grad.normalized() - t);
    const float z_inv = 1. / point[2];
    const float z_inv_sq = z_inv * z_inv;
	float m = fx * point[0] * z_inv + cx;
    float n = fy * point[1] * z_inv + cy;

    if (m<0 || m>=img.cols || n<0 || n>=img.rows) {
        return false;
    }


    Eigen::Matrix<float,3,2> image_grad;
    image_grad.col(0) = computeImageGradient(n, m, img, 0); //3x2
    image_grad.col(1) = computeImageGradient(n, m, img, 1);

    Eigen::Matrix<float, 2, 3> pi_grad = Eigen::Matrix<float, 2, 3>::Zero();
    pi_grad(0,0) = fx * z_inv;
    pi_grad(0,2) = -fx * point[0] * z_inv_sq;
    pi_grad(1,1) = fy * z_inv;
    pi_grad(1,2) = -fy * point[1] * z_inv_sq;

    Mat3f image_pi_grad = image_grad * pi_grad;
    Jc.block<3, 3>(0, 0) = -image_pi_grad * Rt;
    Jc.block<3, 3>(0, 3) = image_pi_grad * skew(point);
	return true;
}

// ========== helper functions for Jacobians ==========

//! get I_i(R_iv+t) RGB/gray value for 8 subvoxels 
bool PhotometricOptimizer::getIntensity(const cv::Mat& img, const Vec3i& idx, const Mat3f& R, const Vec3f& t, Vec3f& intensity)
{
	const float fx = K_(0,0);
	const float fy = K_(1,1);
	const float cx = K_(0,2);
	const float cy = K_(1,2);

	SdfVoxel voxel(getSdf(idx));

	Mat3f Rt = R.transpose();
    Vec3f point = Rt * (tSDF_->vox2float(idx) - voxel.dist * voxel.grad.normalized() - t);
    const float z_inv = 1. / point[2];
    const float z_inv_sq = z_inv * z_inv;
	float m = fx * point[0] * z_inv + cx;
    float n = fy * point[1] * z_inv + cy;

    if (m<0 || m>=img.cols || n<0 || n>=img.rows) {
        return false;
    }

    intensity = interpolateImage(n, m, img);
	return true;
}

// ========== set values for individual voxels ==========

//! update SDF value
void PhotometricOptimizer::updateDist(const Vec3i& idx, const float delta_d)
{
	tSDF_->tsdf_.at(idx).dist -= delta_d;
}

// ========== optimization and energy computation ==========

//! compute total energy
float PhotometricOptimizer::getEnergy()
{
	float E = 0.0;
	float E_reg = 0.0;
	size_t num_res = 0;
    const size_t num_frames = frame_idx_.size();

	for (const auto& vp : tSDF_->tsdf_){
		// each voxel sum over all frames
        Vec3i idx = vp.first;
		std::vector<bool> vis = tSDF_->vis_.at(idx);
		SdfVoxel voxel(vp.second);

        if (std::fabs(voxel.dist) > voxel_size_)
            continue;

		size_t Nj = 0; // number of frames where voxel is visible
		std::vector<size_t> vec_i;
		Vec3f mean_Aij = Vec3f::Zero();
		std::vector<Vec3f, Eigen::aligned_allocator<Vec3f>> vec_Aij;

		// photometric data term (photoconsistency) accumulate
		for(int i = 0; i < num_frames; ++i){
            if (vis.size()<=frame_idx_[i] || !vis[frame_idx_[i]]){
                continue;
			}

            Mat3f R = poses_[i].topLeftCorner(3, 3);
            Vec3f t = poses_[i].topRightCorner(3, 1);
            cv::Mat& img = *(images_[i]);

            Vec3f Aij;
            if(!getIntensity(img, idx, R, t, Aij)) {
                continue;
            }
            ++Nj;
            mean_Aij += Aij;
            vec_Aij.push_back(Aij);
		}

		num_res += Nj;

		mean_Aij = (1. / static_cast<float>(Nj)) * mean_Aij;
		for (size_t k=0; k<Nj; ++k) {
			E += (vec_Aij[k] - mean_Aij).squaredNorm(); // standard L2 loss
		}
	}
    return E;
}

// ------------------- solvers -----------------------------------------------------------------------------------------------------------------

//! compute SDF update in bundle adjustment manner
void PhotometricOptimizer::solveDist(float damping)
{
	float H_dd;
	float b_d;

    const size_t num_frames = frame_idx_.size();

	for (const auto& vp : tSDF_->tsdf_){
		// each voxel sum over all frames
        Vec3i idx = vp.first;
		std::vector<bool> vis = tSDF_->vis_.at(idx);
		SdfVoxel voxel(vp.second);

		// vector and number of visible frames
		size_t Nj = 0;
		// sum of intensities Aij
		Vec3f sum_Aij = Vec3f::Zero();
		// sum of derivatives d Aij / d dj
		Vec3f sum_Aijdj = Vec3f::Zero();
		// sum of product intensities * derivatives
		Vec3f sum_Aij_Aijdj = Vec3f::Zero();
		// sum of squared derivatives
		Vec3f sum_Aijdj_sq = Vec3f::Zero();

		for (size_t i = 0; i < num_frames; ++i)
		{
			// check visibility of voxel in frame i
			if (vis.size()<=frame_idx_[i] || !vis[frame_idx_[i]])
				continue;

            Mat3f R = poses_[i].topLeftCorner(3, 3);
            Vec3f t = poses_[i].topRightCorner(3, 1);
            cv::Mat& img = *(images_[i]);

			Vec3f Aij, Aijdj;
			if (!getIntensity(img, idx, R, t, Aij))
				continue;

			if (settings_.loss == LossFunction::TRUNC_L2 && Aij.array().square().maxCoeff() > settings_.lambda_sq) // ignore too large residuals
				continue;

			Nj++;
			computeJdOneFrame(idx, voxel, R, t, img, Aijdj);
			sum_Aij += Aij;
			sum_Aijdj += Aijdj;
			sum_Aij_Aijdj += Aij.cwiseProduct(Aijdj);
			sum_Aijdj_sq += Aijdj.cwiseProduct(Aijdj);
		}

		float inv_Nj = 1. / static_cast<float>(Nj);

		if (Nj == 0)
			continue;
		
		H_dd = sum_Aijdj_sq.sum() - inv_Nj * sum_Aijdj.cwiseProduct(sum_Aijdj).sum();
		b_d = sum_Aij_Aijdj.sum() - inv_Nj * sum_Aij.cwiseProduct(sum_Aijdj).sum();

		H_dd += settings_.reg_weight * voxel.weight; // damping

        if (H_dd != 0)
		    updateDist(idx, damping*b_d/H_dd);
	}
}


//! compute pose update in bundle adjustment manner
void PhotometricOptimizer::solvePoseFull(float damping)
{
    
    const size_t num_frames = frame_idx_.size();
	Eigen::Matrix<float, -1, -1> H(6*num_frames, 6*num_frames);
	Eigen::Matrix<float, -1, 1>  b(6*num_frames, 1);
	H.setZero();
	b.setZero();

	for (const auto& vp : tSDF_->tsdf_){
		// each voxel sum over all frames
        Vec3i idx = vp.first;
		std::vector<bool> vis = tSDF_->vis_.at(idx);
		SdfVoxel voxel(vp.second);
        if (std::fabs(voxel.dist) > voxel_size_)
            continue;

		// vector and number of visible frames
		std::vector<size_t> vec_i;
		size_t Nj = 0;
		// vector and mean of intensities Aij
		std::vector<Vec3f, Eigen::aligned_allocator<Vec3f>> vec_Aij;
		Vec3f mean_Aij = Vec3f::Zero();
		// vector of camera Jacobians d Aij / d Ti
		std::vector<Eigen::Matrix<float, 3, 6>, Eigen::aligned_allocator<Eigen::Matrix<float, 3, 6>>> vec_AijTi;
		vec_AijTi.clear();

		for (size_t i = 0; i < num_frames; ++i)
		{
			// check visibility of voxel in frame i
			if (vis.size()<=frame_idx_[i] || !vis[frame_idx_[i]])
				continue;				

            Mat3f R = poses_[i].topLeftCorner(3, 3);
            Vec3f t = poses_[i].topRightCorner(3, 1);
            cv::Mat& img = *(images_[i]);

			Vec3f Aij;
			Eigen::Matrix<float, 3, 6> AijTi;

			if (!getIntensity(img, idx, R, t, Aij) || !computeJc(idx, voxel, img, R, t, AijTi))
				continue;

			if (settings_.loss == LossFunction::TRUNC_L2 && Aij.array().square().maxCoeff() > settings_.lambda_sq) // ignore too large residuals
				continue;

			++Nj;
			vec_i.push_back(i);
			vec_Aij.push_back(Aij);
			mean_Aij += Aij;
			vec_AijTi.push_back(AijTi);
		}

		if (Nj == 0)
			continue;

		float inv_Nj = 1. / static_cast<float>(Nj);

		mean_Aij = inv_Nj * mean_Aij;

		for (size_t k1 = 0; k1 < Nj; ++k1)
		{
			size_t i1 = vec_i[k1];
			// extract elements from vectors
			Vec3f Ai1j = vec_Aij[k1];
			Eigen::Matrix<float, 3, 6> Ai1jTi1 = vec_AijTi[k1];
			Vec3f ri1j = Ai1j - mean_Aij;

			// add elements to b-vector
			Eigen::Matrix<float, 1, 6> bi = (ri1j.transpose().asDiagonal() * Ai1jTi1).colwise().sum(); // 1x3 x 3x6
			b.block<6,1>(6*i1, 0) += bi.transpose();
			// add diagonal blocks to Hessian
			Mat6f Hii = Ai1jTi1.transpose() * Ai1jTi1; // 6x3 x 3x6
			
			H.block<6,6>(6*i1, 6*i1) += (1 - inv_Nj) * Hii;

			for (size_t k2 = k1+1; k2 < Nj; ++k2)
			{
				size_t i2 = vec_i[k2];
				// extract elements from vectors
				Vec3f Ai2j = vec_Aij[k2];
				Eigen::Matrix<float, 3, 6> Ai2jTi2 = vec_AijTi[k2];
				// add off-diagonal blocks to Hessian
				Hii = Ai1jTi1.transpose() * Ai2jTi2; // 6x3 x 3x6
				

				H.block<6,6>(6*i1, 6*i2) += (-inv_Nj) * Hii;
				H.block<6,6>(6*i2, 6*i1) += (-inv_Nj) * Hii.transpose();
			}
		}
	}


	Eigen::Matrix<float, -1, 1> delta_pose = H.ldlt().solve(b);
	// delta_pose(delta_pose.array().isNaN()).select(0, delta_pose);
	// update poses_ accordingly
	for (int i = 0; i < num_frames; ++i) {
		if(delta_pose.hasNaN())
			continue;
        Vec3f delta_t = delta_pose.block<3,1>(6 * i, 0);
        Vec3f omega = delta_pose.block<3,1>(6*i + 3, 0);
        poses_[i].topRightCorner(3, 1) -= delta_t;
        poses_[i].topLeftCorner(3, 3) = poses_[i].topLeftCorner(3, 3) * SO3::exp(-omega).matrix();
	}
}

//! compute pose update in bundle adjustment manner
void PhotometricOptimizer::solvePose(float damping)
{
    
    const size_t num_frames = frame_idx_.size();
	Eigen::Matrix<float, -1, -1> H(6*num_frames, 6);
	Eigen::Matrix<float, -1, 1>  b(6*num_frames, 1);
	H.setZero();
	b.setZero();

	for (const auto& vp : tSDF_->tsdf_){
		// each voxel sum over all frames
        Vec3i idx = vp.first;
		std::vector<bool> vis = tSDF_->vis_.at(idx);
		SdfVoxel voxel(vp.second);
        if (std::fabs(voxel.dist) > voxel_size_)
            continue;

		// vector and number of visible frames
		std::vector<size_t> vec_i;
		size_t Nj = 0;
		// vector and mean of intensities Aij
		std::vector<Vec3f, Eigen::aligned_allocator<Vec3f>> vec_Aij;
		Vec3f mean_Aij = Vec3f::Zero();
		// vector of camera Jacobians d Aij / d Ti
		std::vector<Eigen::Matrix<float, 3, 6>, Eigen::aligned_allocator<Eigen::Matrix<float, 3, 6>>> vec_AijTi;
		vec_AijTi.clear();

		for (size_t i = 0; i < num_frames; ++i)
		{
			// check visibility of voxel in frame i
			if (vis.size()<=frame_idx_[i] || !vis[frame_idx_[i]])
				continue;				

            Mat3f R = poses_[i].topLeftCorner(3, 3);
            Vec3f t = poses_[i].topRightCorner(3, 1);
            cv::Mat& img = *(images_[i]);

			Vec3f Aij;
			Eigen::Matrix<float, 3, 6> AijTi;

			if (!getIntensity(img, idx, R, t, Aij) || !computeJc(idx, voxel, img, R, t, AijTi))
				continue;

			if (settings_.loss == LossFunction::TRUNC_L2 && Aij.array().square().maxCoeff() > settings_.lambda_sq) // ignore too large residuals
				continue;

			++Nj;
			vec_i.push_back(i);
			vec_Aij.push_back(Aij);
			mean_Aij += Aij;
			vec_AijTi.push_back(AijTi);
		}

		if (Nj == 0)
			continue;

		float inv_Nj = 1. / static_cast<float>(Nj);

		mean_Aij = inv_Nj * mean_Aij;

		for (size_t k1 = 0; k1 < Nj; ++k1)
		{
			size_t i1 = vec_i[k1];
			// extract elements from vectors
			Vec3f Ai1j = vec_Aij[k1];
			Eigen::Matrix<float, 3, 6> Ai1jTi1 = vec_AijTi[k1];
			Vec3f ri1j = Ai1j - mean_Aij;

			// add elements to b-vector
			// Eigen::Matrix<float, 1, 6> bi = ri1j.transpose() * Ai1jTi1; // 1x3 x 3x6
			Eigen::Matrix<float, 1, 6> bi = (ri1j.transpose().asDiagonal() * Ai1jTi1).colwise().sum(); // 1x3 x 3x6
			b.block<6,1>(6*i1, 0) += bi.transpose();
			// add diagonal blocks to Hessian
			Mat6f Hii = Ai1jTi1.transpose() * Ai1jTi1; // 6x3 x 3x6
			
			H.block<6,6>(6*i1, 0) += (1 - inv_Nj) * Hii;
		}
	}


	// update poses accordingly
	for (int i = 0; i < num_frames; ++i) {
		// solve Hi * delta = bi using LDLT decomposition
		Vec6f delta_pose = H.block<6, 6>(6*i, 0).ldlt().solve(b.segment<6>(6*i));
		if (delta_pose.hasNaN())
			continue;
        Vec3f delta_t = delta_pose.head(3);
        Vec3f omega = delta_pose.tail(3);
        poses_[i].topRightCorner(3, 1) = poses_[i].topRightCorner(3, 1) - delta_t;
        poses_[i].topLeftCorner(3, 3) = poses_[i].topLeftCorner(3, 3) * SO3::exp(-omega).matrix();
	}
}

bool PhotometricOptimizer::savePoses(std::string filename)
{
	std::ofstream posefile;
	posefile.open((save_path_ + filename + ".txt").c_str());
	if(!posefile.is_open())
		return false;

	for(size_t i= 0; i < poses_.size(); i++){
		Eigen::Quaternion<float> q(poses_[i].topLeftCorner<3,3>());
		Vec3f t = poses_[i].topRightCorner(3,1);
		posefile << key_stamps_[i] << " " << t[0] << " " << t[1] << " " << t[2] << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";

	}
	posefile.close();
	std::cout << "poses file is successfully saved!" << std::endl;
	return true;

}
//! optimization of all variables
bool PhotometricOptimizer::optimize()
{
	// save poses before optimization for comparison
	savePoses("selected_frame_poses_before_optimization");
	// // Coarse Bundle Adjustment
	Timer T;
    float c_E = getEnergy();
    float c_damping = 1.0;
    std::vector<float> c_energy;
    c_energy.push_back(c_E);
    std::cout << "Energy before BA: " << c_E << std::endl;
    int iter = 0;
    float crel_diff;
    while (iter < settings_.max_it)
    {
        T.tic();
        solvePose(1.0);
		// solvePoseFull(1.0);    //uncomment this line and comment last line to test coupled version of poses optimization.
        T.toc("Pose optimization");
        c_E = getEnergy();
        c_energy.push_back(c_E);
        std::cout << "Energy after " << iter << " iterations of coarse BA (pose): " << c_E << std::endl;
        
		T.tic();
		solveDist(c_damping);
		T.toc("Distance optimization");
		c_E = getEnergy();
		c_energy.push_back(c_E);
		
		std::cout << "Energy after " << iter << " iterations of coarse BA (dist): " << c_E << std::endl;
        
        crel_diff = abs(c_energy.end()[-2] - c_E) / c_energy.end()[-2];

        std::cout << "======> [" << iter << "]: rel_diff " << crel_diff << std::endl;
        if(crel_diff < 0.0005){
            std::cout << "======================================================> converge after " << iter << " iterations." << std::endl;
			savePoses("coarse_BA_poses_optimized");
            return true;
        }

        if(c_energy.end()[-2] < c_energy.end()[-1]){
            std::cout << "=====================================> DIVERGE after " << iter << "iterations." <<std::endl;
			savePoses("coarse_BA_poses_optimized");
			return false;
        }
        iter++;

    }
   
	savePoses("coarse_BA_poses_optimized");

    return false;
}