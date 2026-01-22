#ifndef UNIFACE_DETECTOR_HPP_
#define UNIFACE_DETECTOR_HPP_

#include "uniface/types.hpp"

#include <array>
#include <string>
#include <vector>

#include <opencv2/dnn.hpp>

namespace uniface {

// RetinaFace detector using OpenCV DNN backend
class RetinaFace {
public:
    explicit RetinaFace(
        const std::string& model_path,
        float conf_thresh = 0.5f,
        float nms_thresh = 0.4f,
        cv::Size input_size = cv::Size(640, 640)
    );

    // Detect faces in BGR image, returns bboxes + 5-point landmarks
    [[nodiscard]] std::vector<Face> detect(const cv::Mat& image);

    // Accessors
    [[nodiscard]] float getConfidenceThreshold() const noexcept { return confidence_threshold_; }
    [[nodiscard]] float getNmsThreshold() const noexcept { return nms_threshold_; }
    [[nodiscard]] cv::Size getInputSize() const noexcept { return input_size_; }

    void setConfidenceThreshold(float threshold) noexcept { confidence_threshold_ = threshold; }
    void setNmsThreshold(float threshold) noexcept { nms_threshold_ = threshold; }

private:
    cv::dnn::Net net_;
    float confidence_threshold_;
    float nms_threshold_;
    cv::Size input_size_;
    std::vector<std::array<float, 4>> anchors_;

    void generateAnchors();
};

}  // namespace uniface

#endif  // UNIFACE_DETECTOR_HPP_
