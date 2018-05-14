#include "face_detector.hpp"

face_detector::face_detector(std::string weights, std::string model)
{
    network = cv::dnn::readNetFromCaffe(weights, model);
}

std::vector<detected_face> face_detector::post_process(cv::Mat & frame,
                                                        const cv::Mat& outs)
{
    std::vector<detected_face> res;
    int width = frame.cols, height = frame.rows;

    // Network produces output blob with a shape 1x1xNx7 where N is a number of
    // detections and an every detection is a vector of values
    // [batchId, classId, confidence, left, top, right, bottom]
    float* data = (float*)outs.data;
    for (size_t i = 0; i < outs.total(); i += 7)
    {
        detected_face current;
        current.confidence = data[i + 2];
        int left = (int)(data[i + 3] * width);
        int top = (int)(data[i + 4] * height);
        int right = (int)(data[i + 5] * width);
        int bottom = (int)(data[i + 6] * height);
        // int classId = (int)(data[i + 1]) - 1;  // Skip 0th background class id.
        current.bounds = cv::Rect(left, top, right - left, bottom - top);
        res.push_back(current);
    }
    return res;
}

std::vector<detected_face> face_detector::detect(cv::Mat & image) {
    std::vector<detected_face> res;
    int height = image.rows, width = image.cols;
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(300, 300));
    cv::Mat blob = cv::dnn::blobFromImage(resized,
                                          1.0, cv::Size(300, 300),
                                          cv::Scalar(104., 177., 123.));

    network.setInput(blob);
    auto detections = network.forward();
    
    return post_process(image, detections);
}

cv::Mat face_detector::draw(cv::Mat & image,
                            std::vector<detected_face> faces,
                            float confidence) {
    cv::Mat res(image);
    cv::Scalar color(0, 0, 255);
    for (auto face : faces) {
        if (face.confidence >= confidence) {
            std::string text = std::to_string(face.confidence) + " %";
            int y = face.bounds.y - 15 < 15 ? face.bounds.y + 15 : face.bounds.y - 10;
            cv::putText(res, text, cv::Size(face.bounds.x, y), cv::FONT_HERSHEY_SIMPLEX,
                        0.45, color, 2);
            cv::rectangle(res, face.bounds, color, 2);
        }
    }
    return image;
}
