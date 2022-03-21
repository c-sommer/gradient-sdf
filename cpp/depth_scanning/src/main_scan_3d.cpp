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
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <CLI/CLI.hpp>
// class includes
#include "Timer.h"
#include "normals/NormalEstimator.h"
#include "sdf_tracker/MapGradPixelSdf.h"
#include "sdf_tracker/MapPixelSdf.h"
#include "sdf_tracker/RigidPointOptimizer.h"
#include "img_loader/img_loader.h"

// own includes
#include "mat.h"

/**
 * main function
 */

int main(int argc, char *argv[]) {

    Timer T;

    // Default input sequence in folder
    
    std::string input = "";
    std::string output = "../results/";
    std::string pose_file_name = "pose.txt";
    std::string stype = "map-gp";
    std::string dtype = "";
    size_t first = 0;
    size_t last = std::numeric_limits<size_t>::max();
    float voxel_size = 0.01; // Voxel size in m
    float z_max = 3.5; // maximal depth to take into account in m
    float truncation_factor = 5; // truncation in voxels
    bool save_sdf = false; // specify if final SDF (+gradients) shall be saved

    CLI::App app{"Hash Table-Based 3D Scanning"};
    app.add_option("--input", input, "folder of input sequence");
    app.add_option("--results", output, "folder to store results in");
    app.add_option("--pose-file", pose_file_name, "pose file name in input folder");
    app.add_option("--first", first, "number of first frame to be processed (default: 0)");
    app.add_option("--last", last, "number of last frame (default: all)");
    app.add_option("--scan-type", stype, "type of scanner used (default: grad-sdf)");
    app.add_option("--data-type", dtype, "type of dataset");
    app.add_option("--voxel-size", voxel_size, "voxel size in meters (default: 0.01)");
    app.add_option("--trunc", truncation_factor, "truncation in multiples of voxels");
    app.add_flag("--save-sdf", save_sdf, "flag: save sdf (+gradient) to txt file");

    // parse input arguments
    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    }

    // parse scan type
    enum class ScanType {
        GRAD_SDF,
        BASE
    };
    ScanType ST;
    if (stype == "grad-sdf") {
        ST = ScanType::GRAD_SDF;
    }
    else if (stype == "base-sdf") {
        ST = ScanType::BASE;
    }
    else {
        std::cerr << "Your specified scan type is not supported (yet)." << std::endl;
        return 1;
    }
    
    // parse dataset type
    enum class DataType {
        TUM_RGBD,
        SYNTH,
        PRINTED_3D,
        REDWOOD,
    };
    DataType DT;
    if (dtype == "tum") {
        DT = DataType::TUM_RGBD;
    }
    else if (dtype == "synth") {
        DT = DataType::SYNTH;
    }
    else if (dtype == "printed") {
        DT = DataType::PRINTED_3D;
    }
    else if (dtype == "rw" || dtype == "redwood") {
        DT = DataType::REDWOOD;
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

    // load GT poses if they are available
    std::vector<Mat4f, Eigen::aligned_allocator<Mat4f>> poses;
    bool GT_pose = false;
    if (!loader->load_pose(input + pose_file_name, poses)){
        std::cerr << "No GT poses are avaible!" << std::endl;
    }
    else{
        std::cout << poses.size() << " GT poses are loaded!" << std::endl;
        GT_pose = true;
    }
    
    // create normal estimator
    T.tic();
    cv::NormalEstimator<float>* NEst;
    NEst = new cv::NormalEstimator<float>(640, 480, K, cv::Size(2*5+1, 2*5+1));
    T.toc("Init normal estimation");

    // float voxel_size = 0.02; // Voxel size in m
    std::stringstream stream;
    stream << std::fixed << std::setprecision(3) << voxel_size;
    std::string voxel_size_str = stream.str();
    // Trunction distance for the tSDF
    const float truncation = truncation_factor * voxel_size;

    Sdf* tSDF;
    RigidPointOptimizer* pOpt;
    
    std::string filename = output + "_poses.txt";
    std::ofstream pose_file(filename);

    // Frames to be processed
    cv::Mat color, depth;

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

        if (i == first) {

            Mat4f Trans = Mat4f::Identity();
            if(GT_pose)Trans = poses[0];

            // create SDF data
            T.tic();
            switch (ST) {
                case ScanType::GRAD_SDF :
                    tSDF = new MapGradPixelSdf(voxel_size, truncation);
                    break;
                case ScanType::BASE :
                    tSDF = new MapPixelSdf(voxel_size, truncation);
                    break;
                default:
                    std::cerr << "Specified scan type not recognized, return" << std::endl;
                    return 1;
            }
            T.toc("Create Sdf");
            
            // Initialize tSDF
            T.tic();
            if(GT_pose){tSDF->update(color, depth, K, SE3(poses[0]), NEst);}
            else{tSDF->setup(color, depth, K, NEst);}
            T.toc("Integrate depth data into Sdf");
			// Initialize optimizer
			T.tic();
			pOpt = new RigidPointOptimizer(tSDF);
			T.toc("Create RigidOptimizer");
        }
        else if(GT_pose){
            T.tic();
            tSDF->update(color, depth, K, SE3(poses[i]), NEst);
            T.toc("Integrate depth data into Sdf");
        }
		else {
            // Perform optimization
            T.tic();
            bool conv = pOpt->optimize(depth, K);
            T.toc("Point optimization");
            // Integrate data into model
            if (conv) {
                T.tic();
                tSDF->update(color, depth, K, pOpt->pose(), NEst);
                T.toc("Integrate depth data into Sdf");
            }
		}
		// write timestamp + pose in tx ty tz qx qy qz qw format
		Mat4f p;
        if(GT_pose){p = poses[i];}
        else{p = pOpt->pose().matrix(); }
		std::cout << "Current pose:" << std::endl
		          << p << std::endl;

		Vec3f t(p.topRightCorner(3,1));
        Mat3f R = p.topLeftCorner(3,3);
		Eigen::Quaternion<float> q(R);

		pose_file << loader->depth_timestamp() << " "
		          << t[0] << " " << t[1] << " " << t[2] << " "
		          << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
    }
    
    pose_file.close();
    
    std::string file_prefix = "gradient_sdf";

    // extract mesh and write to file
    T.tic();
    filename = output + file_prefix + "_mesh_final.ply";
    if (!tSDF->extract_mesh(filename)) {
        std::cerr << "Could not save mesh to " << filename << "!" << std::endl;
    }
    T.toc("Save mesh to disk");
    
    // extract point cloud and write to file
    T.tic();
    filename = output + file_prefix + "_cloud_final.ply";
    if (!tSDF->extract_pc(filename)) {
        std::cerr << "Could not save point cloud to " << filename << "!" << std::endl;
    }
    T.toc("Save point cloud to disk");

    // write sdf to file to run gradient analysis
    if (save_sdf) {
        T.tic();
        filename = output + file_prefix;
        if(!tSDF->save_sdf(filename)){
            std::cerr << "could not save voxel grid info file " << filename << "!" << std::endl;
        }
        T.toc("Save sdf txt files to disk");
    }
    
    // tidy up
    delete tSDF;
    delete pOpt;
    delete loader;

    return 0;
}
