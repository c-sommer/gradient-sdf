cmake_minimum_required(VERSION 3.2)

SET(EXECUTABLE_OUTPUT_PATH ../../photometric_opt/bin)
SET(SUBPROJECT_NAME PhotoBA)

ADD_EXECUTABLE( ${SUBPROJECT_NAME}
    src/main_photo_ba.cpp
)

TARGET_LINK_LIBRARIES( ${SUBPROJECT_NAME}
    sdf_tracker_lib
    ${OpenCV_LIBS}
    color_lib
    mc_lib
)