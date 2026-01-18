#ifndef UNIFACE_LANDMARKER_HPP_
#define UNIFACE_LANDMARKER_HPP_

#include "uniface/types.hpp"

#include <string>

#include <opencv2/dnn.hpp>

namespace uniface {

/// 106-point facial landmark detector
class Landmark106 {
public:
    /// Construct landmark detector from ONNX model
    /// @param model_path Path to ONNX model file
    /// @param config Landmark configuration (input size)
    explicit Landmark106(
        const std::string& model_path, const LandmarkerConfig& config = LandmarkerConfig{}
    );

    /// Detect 106 facial landmarks for a face
    /// @param image Source BGR image
    /// @param bbox Face bounding box [x, y, width, height]
    /// @return 106 facial landmark points in original image coordinates
    [[nodiscard]] Landmarks getLandmarks(const cv::Mat& image, const cv::Rect2f& bbox);

    // Accessors
    [[nodiscard]] cv::Size getInputSize() const noexcept { return config_.input_size; }

private:
    cv::dnn::Net net_;
    LandmarkerConfig config_;

    /// Preprocess face crop for inference
    /// @param image Source image
    /// @param bbox Face bounding box
    /// @param transform Output affine transform matrix
    /// @return Preprocessed blob for inference
    [[nodiscard]] cv::Mat preprocess(
        const cv::Mat& image, const cv::Rect2f& bbox, cv::Mat& transform
    );

    /// Postprocess model output to get landmarks in original coordinates
    /// @param predictions Raw model output
    /// @param transform Affine transform from preprocessing
    /// @return Landmarks in original image coordinates
    [[nodiscard]] Landmarks postprocess(const cv::Mat& predictions, const cv::Mat& transform);
};

}  // namespace uniface

#endif  // UNIFACE_LANDMARKER_HPP_
