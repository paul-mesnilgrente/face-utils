#ifndef FACE_DETECTION_HPP
#define FACE_DETECTION_HPP
#include "includes.ihh"

struct detected_faces {
    Eigen::MatrixXi faces_bounds;
    Eigen::VectorXf confidences;
};

struct face_detector {
    face_detector(std::string weights, std::string model);

    detected_faces detect(const cv::Mat& image, float min_confidence=.5f);

    cv::Mat draw(const cv::Mat& image,
                 const detected_faces& faces,
                 const float confidence);
private:
    detected_faces post_process(const cv::Mat& frame,
                                const cv::Mat& outs,
                                float min_confidence);
    
    cv::dnn::Net network;
};
#endif
