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


// standard includes
#include <iostream>
#include <fstream>
#include <vector>
// library includes
#include <cstdlib>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <CLI/CLI.hpp>
// class includes
#include "Timer.h"
#include "normals/NormalEstimator.h"
#include "sdf_tracker/MapGradPixelSdf.h"
#include "sdf_tracker/RigidPointOptimizer.h"
#include "img_loader/img_loader.h"
#include "ps_optimizer/PhotometricOptimizer.h"
#include "ps_optimizer/ColorUpsampler.h"
#include "ps_optimizer/SharpDetector.h"
// own includes
#include "mat.h"

void sampleKeyFrame(std::vector<int>& key_frames, std::vector<std::string>& key_stamps, std::vector<std::shared_ptr<cv::Mat>>& key_images, std::vector<Mat4f, Eigen::aligned_allocator<Mat4f>>& key_poses, int max_num);
/**
 * main function
 */
int main(int argc, char *argv[]) {

    Timer T;
    // Default input sequence in folder
    std::string input = "";
    std::string output = "../results/";
    std::string stype = "map-gp";
    std::string dtype = "";
    size_t first = 0;
    size_t last = 300;
    float voxel_size = 0.01; // Voxel size in m
    float z_max = 3.5; // maximal depth to take into account in m
    float truncation_factor = 5; // truncation in voxels
    float sharp_threshold = 0.0001;
    int num_frame = 30;

    CLI::App app{"Hash Table-Based 3D Scanning and Texture Optimization"};
    app.add_option("--input", input, "folder of input sequence");
    app.add_option("--results", output, "folder to store results in");
    app.add_option("--first", first, "number of first frame to be processed (default: 0)");
    app.add_option("--last", last, "number of last frame (default: all)");
    app.add_option("--data-type", dtype, "type of dataset");
    app.add_option("--voxel-size", voxel_size, "voxel size in meters (default: 0.01)");
    app.add_option("--trunc", truncation_factor, "truncation in multiples of voxels (default: 5)");
    app.add_option("--key-frame", num_frame, "number of the key frames to be sampled (default: 20)");
   

    // parse input arguments
    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    }

    // parse dataset type
    enum class DataType {
        TUM_RGBD,
        SYNTH,
        PRINTED_3D,
        REDWOOD,
    };
    DataType DT;
    if (dtype == "tum" || dtype == "tumrgbd") {
        DT = DataType::TUM_RGBD;
        sharp_threshold = 0.026;
    }
    else if (dtype == "synth") {
        DT = DataType::SYNTH;
    }
    else if (dtype == "printed") {
        DT = DataType::PRINTED_3D;
        sharp_threshold = 0.026;
    }
    else if (dtype == "rw" || dtype == "redwood") {
        DT = DataType::REDWOOD;
        sharp_threshold = 0.033;
    }
    else {
        std::cerr << "Your specified dataset type is not supported (yet)." << std::endl;
        return 1;
    }
    
    // create image loader
    ImageLoader* loader;
    switch (DT) {
    case DataType::TUM_RGBD :
        loader = new TumrgbdLoader(input);
        break;
    case DataType::SYNTH :
        loader = new SynthLoader(input);
        break;
    case DataType::PRINTED_3D :
        loader = new Printed3dLoader(input);
        break;
    case DataType::REDWOOD :
        loader = new RedwoodLoader(input);
        break;
    default:
        std::cerr << "Specified dataset type not recognized, return" << std::endl;
        return 1;
    }

    // Load camera intrinsics
    if (!loader->load_intrinsics("intrinsics.txt")) {
        std::cerr << "No intrinsics file found in " << input << "!" << std::endl;
        return 1;
    }
    const Mat3f K = loader->K();
    std::cout << "K: " << std::endl << K << std::endl;
    
    // create normal estimator
    T.tic();
    cv::NormalEstimator<float>* NEst;
    NEst = new cv::NormalEstimator<float>(640, 480, K, cv::Size(2*5+1, 2*5+1));
    T.toc("Init normal estimation");

    std::stringstream stream;
    stream << std::fixed << std::setprecision(3) << voxel_size;
    std::string voxel_size_str = stream.str();

    // Trunction distance for the tSDF
    const float truncation = truncation_factor * voxel_size;

    Sdf* tSDF;
    RigidPointOptimizer* pOpt;
    PhotometricOptimizer* cOpt;
    
    std::ofstream pose_file(output + stype + "_" + voxel_size_str + "m_T" + std::to_string(int(truncation_factor)) + "_poses.txt");

    std::vector<Mat4f, Eigen::aligned_allocator<Mat4f>> poses;
    // poses.push_back(Mat4f::Identity());
    std::vector<int> valid_frames;
    std::vector<int> invalid_frames;
    std::vector<int> keyframes;
    valid_frames.push_back(0);
    keyframes.push_back(0);
    // vector of sampled poses and frames
    std::vector<std::string> key_stamps;
    std::vector<std::shared_ptr<cv::Mat>> key_images;
    std::vector<Mat4f, Eigen::aligned_allocator<Mat4f>> key_poses;
    key_poses.push_back(Mat4f::Identity());


    // Frames to be processed
    cv::Mat color, depth;
    int dist_to_last_keyframe = 0;

    // Proceed until first frame
    for (size_t i = 0; i < first; ++i) {
        loader->load_next(color, depth);
    }
    
    // Actual scanning loop
    for (size_t i = first; i <= last; ++i) {
        std::cout << "Working on frame: " << i << std::endl;
        
        // Load data
        T.tic();
        if (!loader->load_next(color, depth)) {
            std::cerr << " -> Frame " << i << " could not be loaded!" << std::endl;
            T.toc("Load data");
            break;
        }
        T.toc("Load data");

        // Get initial volume pose from centroid of first depth map
        if (i == first) {
            // create SDF data
            T.tic();
            tSDF = new MapGradPixelSdf(voxel_size, truncation);
            T.toc("Create Sdf");
            // Initialize tSDF
            T.tic();
            tSDF->setup(color, depth, K, NEst);
            T.toc("Integrate depth data into Sdf");

			// Initialize optimizers
			T.tic();
			pOpt = new RigidPointOptimizer(tSDF);
			T.toc("Create RigidOptimizer");
            T.tic();
            cOpt = new PhotometricOptimizer(static_cast<MapGradPixelSdf*>(tSDF), voxel_size, K, output);
            T.toc("Create PhotometricOptimizer");
            key_stamps.push_back(loader->rgb_timestamp());
            cv::Mat new_color;
            color.copyTo(new_color);
            key_images.push_back(std::make_shared<cv::Mat>(new_color));
            
        }
		else {
            // Perform optimization
            T.tic();
            bool conv = pOpt->optimize(depth, K);
            T.toc("Point optimization");
            // Integrate data into model
            if (conv) {
                valid_frames.push_back(i-first);
                T.tic();
                tSDF->update(color, depth, K, pOpt->pose(), NEst);
                T.toc("Integrate depth data into Sdf");

                if (sharpDetector(color, sharp_threshold) || dist_to_last_keyframe > 5)
                {
                    dist_to_last_keyframe = 0;
                    keyframes.push_back(i-first);
                    key_stamps.push_back(loader->rgb_timestamp());
                    key_poses.push_back(pOpt->pose().matrix());
                    cv::Mat new_color;
                    color.copyTo(new_color);
                    key_images.push_back(std::make_shared<cv::Mat>(new_color));
                }
                else{

                    dist_to_last_keyframe++;
                }
            }
            else {
                invalid_frames.push_back(i-first);
            }
		}
		// write timestamp + pose in tx ty tz qx qy qz qw format
		std::cout << "Current pose:" << std::endl
		          << pOpt->pose().matrix() << std::endl;
        poses.push_back(pOpt->pose().matrix());
		Vec3f t(pOpt->pose().translation());
		Eigen::Quaternion<float> q(pOpt->pose().rotationMatrix());
		pose_file << loader->depth_timestamp() << " "
		          << t[0] << " " << t[1] << " " << t[2] << " "
		          << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
    }
    pose_file.close();
    
    // extract mesh and write to file
    T.tic();
    if (!tSDF->extract_mesh(output + "mesh_lr.ply")) { 
        std::cerr << "Could not save mesh!" << std::endl;
    }
    T.toc("Save mesh to disk");
    
    // extract point cloud and write to file
    T.tic();
    if (!tSDF->extract_pc(output + "cloud_lr.ply")) { 
        std::cerr << "Could not save point cloud!" << std::endl;
    }
    T.toc("Save point cloud to disk");

    sampleKeyFrame(keyframes, key_stamps, key_images, key_poses, num_frame);

    cOpt->setImages(key_images);
    cOpt->setKeyframes(keyframes);
    cOpt->setPoses(key_poses);
    cOpt->setKeytimestamps(key_stamps);
    cOpt->optimize();

    // up sampling
    ColorUpsampler tOpt((static_cast<MapGradPixelSdf*>(tSDF))->get_tsdf(),
                    (static_cast<MapGradPixelSdf*>(tSDF))->get_vis(),
                    key_images,
                    key_poses,
                    keyframes,
                    voxel_size,
                    K);
    
    std::cout << "Color optimizer constructed" << std::endl;
    tOpt.computeColor(); // just to be able to compare better
    tOpt.extractMesh(output + "coarse_BA_mesh_after_upsample");
    tOpt.extractCloud(output + "coarse_BA_cloud_after_upsample");
    

    return 0;
}


//! To selected limited number of frames if there are too many input as key frames.
void sampleKeyFrame(std::vector<int>& key_frames, std::vector<std::string>& key_stamps, std::vector<std::shared_ptr<cv::Mat>>& key_images, std::vector<Mat4f, Eigen::aligned_allocator<Mat4f>>& key_poses, int max_num){
    if (key_frames.size() < max_num ){
        return;
    }
    max_num -= 1;
    float step = static_cast<float>(key_frames.size()) / static_cast<float>(max_num);
    std::vector<int> frames;
    std::vector<std::string> stamps;
    std::vector<std::shared_ptr<cv::Mat>> images;
    std::vector<Mat4f, Eigen::aligned_allocator<Mat4f>> poses;
    float idx = 0;
    for(int count = 0; count < max_num; count++){
        int i = static_cast<int>(idx);
        frames.push_back(key_frames[i]);
        stamps.push_back(key_stamps[i]);
        images.push_back(key_images[i]);
        poses.push_back(key_poses[i]);
        idx+=step;
    }
    frames.push_back(key_frames.back()); //we need the last frame for resize the visibility vector
    stamps.push_back(key_stamps.back());
    images.push_back(key_images.back());
    poses.push_back(key_poses.back());
    
    key_frames = frames;
    key_stamps = stamps;
    key_poses = poses;
    key_images = images;
}