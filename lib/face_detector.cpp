#include "face_detector.hpp"

face_detector::face_detector(std::string weights, std::string model)
{
    network = cv::dnn::readNetFromCaffe(weights, model);
}

detected_faces face_detector::post_process(const cv::Mat& image,
                                           const cv::Mat& outs,
                                           float min_confidence)
{
    detected_faces res;
    res.faces_bounds = Eigen::MatrixXi::Constant(outs.total(), 4, 0);
    res.confidences = Eigen::VectorXf::Constant(outs.total(), 0.f);
    int width = image.cols, height = image.rows;

    // Network produces output blob with a shape 1x1xNx7 where N is a number of
    // detections and every detection is a vector of values
    // [batchId, classId, confidence, left, top, right, bottom]
    float* data = (float*) outs.data;
    for (size_t i = 0; i < outs.total(); i += 7)
    {
        res.confidences(i) = data[i + 2];
        res.faces_bounds(i, 0) = (int) (data[i + 3] * width);
        res.faces_bounds(i, 1) = (int) (data[i + 4] * height);
        res.faces_bounds(i, 2) = (int) (data[i + 5] * width - res.faces_bounds(i, 0));
        res.faces_bounds(i, 3) = (int) (data[i + 6] * height - res.faces_bounds(i, 1));
    }
    return res;
}

detected_faces face_detector::detect(const cv::Mat& image, float min_confidence)
{
    detected_faces res;
    int height = image.rows, width = image.cols;
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(300, 300));
    cv::Mat blob = cv::dnn::blobFromImage(resized,
                                          1.0, cv::Size(300, 300),
                                          cv::Scalar(104., 177., 123.));

    network.setInput(blob);
    auto detections = network.forward();
    
    return post_process(image, detections, min_confidence);
}

cv::Mat face_detector::draw(const cv::Mat & image,
                            const detected_faces & faces,
                            const float min_confidence)
{
    cv::Mat res(image);
    cv::Scalar color(0, 0, 255);
    for (int i = 0; i < faces.faces_bounds.rows(); i++) {
        if (faces.confidences(i) >= min_confidence) {
            std::string text = std::to_string(faces.confidences(i)) + " %";
            int x = faces.faces_bounds(i, 0);
            int y = faces.faces_bounds(i, 1);
            int w = faces.faces_bounds(i, 2);
            int h = faces.faces_bounds(i, 3);
            int y_text = y - 15 < 15 ? y + 15 : y - 10;
            cv::putText(res, text, cv::Point(x, y_text), cv::FONT_HERSHEY_SIMPLEX,
                        0.45, color, 2);
            cv::rectangle(res, cv::Rect(x, y, w, h), color, 2);
        }
    }
    return res;
}
