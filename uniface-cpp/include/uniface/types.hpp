#ifndef UNIFACE_TYPES_HPP_
#define UNIFACE_TYPES_HPP_

#include <array>
#include <vector>

#include <opencv2/core.hpp>

namespace uniface {

// Detected face with bbox, confidence, and 5-point landmarks
struct Face {
    cv::Rect2f bbox;
    float confidence;
    std::array<cv::Point2f, 5> landmarks;  // left_eye, right_eye, nose, left_mouth, right_mouth
};

// 512-dimensional face embedding
using Embedding = std::array<float, 512>;

// 106-point facial landmarks
struct Landmarks {
    std::array<cv::Point2f, 106> points;
};

// Configuration structs
struct DetectorConfig {
    float conf_thresh = 0.5f;
    float nms_thresh = 0.4f;
    cv::Size input_size = cv::Size(640, 640);
};

struct RecognizerConfig {
    float input_mean = 127.5f;
    float input_std = 127.5f;
    cv::Size input_size = cv::Size(112, 112);
};

struct LandmarkerConfig {
    cv::Size input_size = cv::Size(192, 192);
};

}  // namespace uniface

#endif  // UNIFACE_TYPES_HPP_
