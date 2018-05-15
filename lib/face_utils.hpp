#ifndef FACE_UTILS_HPP
#define FACE_UTILS_HPP
#include "includes.ihh"

namespace face_utils {
cv::Mat resize(const cv::Mat& image, int width=-1, int height=-1, int inter=cv::INTER_AREA)
{
    // initialize the dimensions of the image to be resized and
    // grab the image size
    cv::Size dim;
    int h = image.rows, w = image.cols;

    // if both the width and height are None, then return the
    // original image
    if (width == -1 && height == -1)
        return image;

    // check to see if the width is None
    if (width == -1) {
        // calculate the ratio of the height and construct the
        // dimensions
        float ratio = height / float(h);
        dim = cv::Size(int(w * ratio), height);
    }

    // otherwise, the height is None
    else {
        // calculate the ratio of the width and construct the
        // dimensions
        float ratio = width / float(w);
        dim = cv::Size(width, int(h * ratio));
    }
    // resize the image
    cv::Mat resized;
    cv::resize(image, resized, dim, 0., 0., inter);

    // return the resized image
    return resized;
}
}
#endif
