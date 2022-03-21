# Gradient-SDF

This repository contains the code accompanying the paper 

**Gradient-SDF: A Semi-Implicit Surface Representation for 3D Reconstruction**

by Christiane Sommer*, Lu Sang* (equal contribution), David Schubert, and Daniel Cremers, accepted for publication at CVPR 2022. A preprint version can be found on [arXiv](https://arxiv.org/pdf/2111.13652v1.pdf).

The **C++ code** in [cpp/](cpp/) contains two experiments:
- 3D scanning: our Gradient-SDF representation is used to do depth camera tracking and mapping. 
- Photometric bundle adjustment: our Gradient-SDF representation is used to do bundle adjustment to further improve camera poses and SDF distance.

The **Matlab code** in [matlab/](matlab/) allows to analyze SDF gradients for synthetic renderings of a set of random spheres.

## Instructions - C++ code

### Dependencies
- You need to have **OpenCV 4** installed on your computer.
- For other packages, see [`cpp/include/third/`](cpp/include/third). All these additional dependencies are included as submodules and will be automatically added when the repository is cloned with the `--recurse-submodules` flag. Only headers will be used from these.

We also provide a Dockerfile to create a configuration in which OpenCV 4 is already installed:
```
docker build . -t gradient_sdf
docker run -it gradient_sdf
```
Note that for this to work, you need to have all test data copied to the repository directory.
You can then continue inside the command line of the Docker container.

### Compile
1. Go to `cpp` folder and run `mkdir build`
2. Go to `cpp/build/` and run
    ```
    cmake ..
    make -j
    ```
Two binary files will be generated in `cpp/depth_scanning/bin/` and `cpp/photometric_opt/bin`.

### Run

We provide data loaders for different dataset formats. After downloading a sequence of one of the datasets mentioned below, all you need to add it a file `intrinsics.txt` to the sequence folder, containing the camera intrinsics in format
```
fx 0 cx
0 fy cy
0 0 1
```
- **3D scanning**: to obtain runtimes comparable to those stated in the paper, make sure to activate the OMP versions of the `.cpp` files in `CMakeLists.txt` (ll. 54-56).
The executable file `cpp/depth_scanning/bin/Scan3D` takes at least 3 input arguments. 
    - `--input`: folder containing rgb and depth images, as well as *intrinsic.txt*
    - `--output`: path the save output files
    - `--data-type`: supported datasets format *[tum_rgbd_benchmark](https://vision.in.tum.de/data/datasets/rgbd-dataset)*, *[redwood](http://redwood-data.org/3dscan/)*, *[printed3D](http://campar.in.tum.de/personal/slavcheva/3d-printed-dataset/index.html)*

    The scan type is an optional argument specifying if you want to run our proposed Gradient-SDF code or our baseline implementation of a direct SDF tracker with a voxel hash map to store the SDF.
    - `--scan-type`: Gradient-SDF (`grad-sdf`, default) or standard SDF (`base-sdf`), both use a hash table to store the SDF voxels.

    For other input argument you can use `./Scan3D -h` to check.

    Example command line:
    ```
    cd cpp/depth_scanning/bin
    ./Scan3D --input /dataset_path/ --output /savepath/ --scan-type voxel-gp --data-type tum
    ```

- **Photometric bundle adjustment**: The excutable file `cpp/photometric_opt/bin/PhotoBA` takes at least 3 input arguments.
Note that the non-OMP versions need to be activated in `CMakeLists.txt` (ll. 50-52) for this to work.
    - `--input`: folder containing rgb and depth images, as well as `intrinsics.txt`
    - `--output`: path the save output files
    - `--data-type`: supported datasets format *[tum_rgbd_benchmark](https://vision.in.tum.de/data/datasets/rgbd-dataset)*, *[redwood](http://redwood-data.org/3dscan/)*, *[printed3D](http://campar.in.tum.de/personal/slavcheva/3d-printed-dataset/index.html)*
    
    For other input argument you can use `./PhotoBA -h` to check.

    Example command line:
    ```
    cd cpp/photometric_opt/bin
    ./PhotoBA --input /dataset_path/ --output /savepath/ --data-type tum
    ```

## Instructions - Matlab code

### Data generation

Adapt the `out_path` variable in `RenderSpheres.m` and then run the script to generate the synthetic sphere data. A new random set of five spheres will be generated for each run. This can be disabled by commenting ll.10-18 and uncommenting l.21.

### SDF generation

To generate the SDF and the stored Gradient-SDF vectors, you need to run the C++ code with `--input <out_path>` and `--trunc 10`.

### Gradient analysis

Adapt the `path` variable in `GradientAnalysisSpheres.m` to your `--output` path from the SDF generation and adapt the `sz`, `dmin`, and `dmax` variables accordingly (see file). Then run the script to generate the plot according to Figure 3 in the paper. In addition to Gradient-SDF and central difference vectors, by default, also forward and backward finite difference curves will be plotted. As they are a lot worse than the central differences and clutter the plot, we decided to exclude them in the paper. This can be achieved by commenting the respective lines (see file).

## License and Publication

Our code is released under the BSD-3 license, for more details please see the license file. Also note the different licenses of the submodules in the folder [`cpp/third/`](cpp/third/).

Please cite our paper when using the code in a scientific project. You can copy-paste the following BibTex entry:

```
@inproceedings{sommersang2022,
    title   = {Gradient-SDF: A Semi-Implicit Surface Representation for 3D Reconstruction},
    author  = {Sommer, Christiane and Sang, Lu and Schubert, David and Cremers, Daniel},
    booktitle = {IEEE/CVF International Conference on Computer Vision and Pattern Recognition (CVPR)},
    year    = {2022}
}
```