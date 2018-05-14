#include "../lib/face_detector.hpp"

int main(int argc, char** argv) {
    // initialize the face detector
    std::string weights = "../../models/deploy.prototxt";
    std::string model = "../../models/res10_300x300_ssd_iter_140000_fp16.caffemodel";
    face_detector detector(weights, model);
    std::vector<detected_face> faces;

    // initialize the camera reading
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        throw std::runtime_error("Camera could not be opened");
    }
    cv::Mat image;

    // detect and print detected faces
    while (cv::waitKey(1) != 27) {
        cap >> image;
        faces = detector.detect(image);
        cv::imshow("Camera 0", detector.draw(image, faces, 0.5));
    }
    
    return 0;
}
