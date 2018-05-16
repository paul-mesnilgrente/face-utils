cmake_minimum_required(VERSION 3.0)
project(face-utils)

find_package(OpenCV REQUIRED)
find_package(dlib REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)
find_package(Eigen3 3.3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})

set(LIB_SOURCES lib/face_detector.cpp
                lib/facial_landmark_detector.cpp
                lib/face_aligner.cpp)
set(LIB_HEADERS lib/face_detector.hpp
                lib/facial_landmark_detector.hpp
                lib/face_aligner.hpp)
add_library(FaceUtils_LIB ${LIB_SOURCES} ${LIB_HEADERS})

set(APP_SOURCES src/face_detection.cpp
                src/main.cpp)
foreach(source_file ${APP_SOURCES})
    get_filename_component(executable_name ${source_file} NAME_WE)
    add_executable(${executable_name} ${source_file})
    # Make sure YourLib is linked to each app
    target_link_libraries(${executable_name} ${OpenCV_LIBS}
                                             ${dlib}
                                             ${Boost_LIBRARIES}
                                             FaceUtils_LIB)
endforeach(source_file ${APP_SOURCES})
