CMAKE_MINIMUM_REQUIRED(VERSION 3.4)

SET(PROJECT_NAME Gradient-Sdf)
SET(CMAKE_BUILD_TYPE RelWithDebInfo)

########## INCLUDED LIBRARIES ##########

# Eigen
SET(EIGEN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third/eigen/")
# FIND_PACKAGE( Eigen REQUIRED )
MESSAGE(STATUS "Found Eigen headers in: ${EIGEN_INCLUDE_DIR}")
INCLUDE_DIRECTORIES( ${EIGEN_INCLUDE_DIR} )

# OpenCV
FIND_PACKAGE( OpenCV REQUIRED )
MESSAGE(STATUS "Found OpenCV headers in: ${OpenCV_INCLUDE_DIRS}")
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )

# Sophus
INCLUDE_DIRECTORIES( third/Sophus/ )

# CLI, command line parser
INCLUDE_DIRECTORIES( third/CLI11/include/ )

# Parallel Hashmap
INCLUDE_DIRECTORIES( third/parallel_hashmap/ )


########## CUSTOM LIBRARIES ##########

INCLUDE_DIRECTORIES( include/
                     third/
                    )

# marching cubes
SET( LIB_NAME mc_lib )
ADD_LIBRARY( ${LIB_NAME}
    include/mesh/HrLayeredMarchingCubes.cpp
    include/mesh/LayeredMarchingCubesNoColor.cpp
    )
TARGET_COMPILE_OPTIONS( ${LIB_NAME} PRIVATE -std=c++17 )



# SDF-Tracking stuff
SET( LIB_NAME sdf_tracker_lib )
ADD_LIBRARY( ${LIB_NAME}
    ## uncomment following lines to run non-OMP version
    include/sdf_tracker/MapGradPixelSdf.cpp
    include/sdf_tracker/MapPixelSdf.cpp
    include/sdf_tracker/RigidPointOptimizer.cpp
    ## uncomment following lines to run OMP version
    # include/sdf_tracker/MapGradPixelSdfOmp.cpp
    # include/sdf_tracker/MapPixelSdfOmp.cpp
    # include/sdf_tracker/RigidPointOptimizerOmp.cpp
    )

TARGET_COMPILE_OPTIONS( ${LIB_NAME} PRIVATE -std=c++17 )

SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp" )
                 

# Color Optimization
SET( LIB_NAME color_lib )
ADD_LIBRARY( ${LIB_NAME}
    include/ps_optimizer/PhotometricOptimizer.cpp
    include/ps_optimizer/ColorUpsampler.cpp
    )


########## DIRECTORIES ##########

ADD_SUBDIRECTORY(depth_scanning) # direct SDF tracking 
ADD_SUBDIRECTORY(photometric_opt) # photometric bundle adjustment