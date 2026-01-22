#ifndef UNIFACE_LANDMARKER_HPP_
#define UNIFACE_LANDMARKER_HPP_

#include "uniface/types.hpp"

#include <string>

#include <opencv2/dnn.hpp>

namespace uniface {

// 106-point facial landmark detector
class Landmark106 {
public:
    explicit Landmark106(const std::string& model_path, const LandmarkerConfig& config = LandmarkerConfig{});

    // Detect 106 landmarks for a face, returns points in original image coordinates
    [[nodiscard]] Landmarks getLandmarks(const cv::Mat& image, const cv::Rect2f& bbox);

    [[nodiscard]] cv::Size getInputSize() const noexcept { return config_.input_size; }

private:
    cv::dnn::Net net_;
    LandmarkerConfig config_;

    [[nodiscard]] cv::Mat preprocess(const cv::Mat& image, const cv::Rect2f& bbox, cv::Mat& transform);
    [[nodiscard]] Landmarks postprocess(const cv::Mat& predictions, const cv::Mat& transform);
};

}  // namespace uniface

#endif  // UNIFACE_LANDMARKER_HPP_
