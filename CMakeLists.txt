cmake_minimum_required(VERSION 2.8)
project(face-utils)

find_package(OpenCV REQUIRED)
find_package(dlib REQUIRED)

set(LIB_SOURCES lib/face_detector.cpp
                lib/facial_landmark_detector.cpp
                lib/face_aligner.cpp)
set(LIB_HEADERS lib/face_detector.hpp
                lib/facial_landmark_detector.hpp
                lib/face_aligner.hpp)
add_library(FaceUtils_LIB ${LIB_SOURCES} ${LIB_HEADERS})

set(APP_SOURCES src/face_detection.cpp)
foreach(source_file ${APP_SOURCES})
    get_filename_component(executable_name ${source_file} NAME_WE)
    add_executable(${executable_name} ${source_file})
    # Make sure YourLib is linked to each app
    target_link_libraries(${executable_name} ${OpenCV_LIBS}
                                             ${dlib}
                                             FaceUtils_LIB)
endforeach(source_file ${APP_SOURCES})
