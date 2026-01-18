#ifndef UNIFACE_RECOGNIZER_HPP_
#define UNIFACE_RECOGNIZER_HPP_

#include "uniface/types.hpp"

#include <string>

#include <opencv2/dnn.hpp>

namespace uniface {

/// ArcFace face recognition model
/// Supports both MobileNet and ResNet backbone variants
class ArcFace {
public:
    /// Construct ArcFace recognizer from ONNX model
    /// @param model_path Path to ONNX model file
    /// @param config Recognition configuration (input size, normalization)
    explicit ArcFace(
        const std::string& model_path, const RecognizerConfig& config = RecognizerConfig{}
    );

    /// Get face embedding from an already-aligned face image
    /// @param aligned_face Pre-aligned 112x112 face image (BGR)
    /// @return 512-dimensional embedding vector
    [[nodiscard]] Embedding getEmbedding(const cv::Mat& aligned_face);

    /// Get face embedding with automatic alignment
    /// @param image Source BGR image
    /// @param landmarks 5-point facial landmarks for alignment
    /// @return 512-dimensional embedding vector
    [[nodiscard]] Embedding getEmbedding(
        const cv::Mat& image, const std::array<cv::Point2f, 5>& landmarks
    );

    /// Get L2-normalized face embedding with automatic alignment
    /// @param image Source BGR image
    /// @param landmarks 5-point facial landmarks for alignment
    /// @return L2-normalized 512-dimensional embedding vector
    [[nodiscard]] Embedding getNormalizedEmbedding(
        const cv::Mat& image, const std::array<cv::Point2f, 5>& landmarks
    );

    // Accessors
    [[nodiscard]] cv::Size getInputSize() const noexcept { return config_.input_size; }

private:
    cv::dnn::Net net_;
    RecognizerConfig config_;

    /// Preprocess face image for inference
    [[nodiscard]] cv::Mat preprocess(const cv::Mat& face_image);
};

}  // namespace uniface

#endif  // UNIFACE_RECOGNIZER_HPP_
