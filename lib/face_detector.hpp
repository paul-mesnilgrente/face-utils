#ifndef FACE_DETECTION_HPP
#define FACE_DETECTION_HPP

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/shape_utils.hpp>

struct detected_face {
    cv::Rect bounds;
    double confidence;
};

struct face_detector {
    face_detector(std::string weights, std::string model);

    std::vector<detected_face> detect(cv::Mat & image);

    cv::Mat draw(cv::Mat & image,
                 std::vector<detected_face> faces,
                 float confidence);
private:
    std::vector<detected_face> post_process(cv::Mat & frame,
                                            const cv::Mat& outs);
    
    cv::dnn::Net network;
};
#endif
