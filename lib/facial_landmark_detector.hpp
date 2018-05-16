#ifndef FACIAL_LANDMARK_HPP
#define FACIAL_LANDMARK_HPP
#include "includes.ihh"
#include "face_detector.hpp"

struct landmarks {
    Eigen::MatrixXi marks;
};

struct facial_landmark_detector {

    facial_landmark_detector(std::string model);

    landmarks detect(const cv::Mat& image, const detected_faces& faces);

    cv::Mat draw(const cv::Mat& image, landmarks marks);
};

#endif
