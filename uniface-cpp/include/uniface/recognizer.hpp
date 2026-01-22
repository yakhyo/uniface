#ifndef UNIFACE_RECOGNIZER_HPP_
#define UNIFACE_RECOGNIZER_HPP_

#include "uniface/types.hpp"

#include <string>

#include <opencv2/dnn.hpp>

namespace uniface {

// ArcFace face recognition (MobileNet/ResNet backbones)
class ArcFace {
public:
    explicit ArcFace(const std::string& model_path, const RecognizerConfig& config = RecognizerConfig{});

    // Get 512-dim embedding from pre-aligned 112x112 face
    [[nodiscard]] Embedding getEmbedding(const cv::Mat& aligned_face);

    // Get 512-dim embedding with automatic alignment
    [[nodiscard]] Embedding getEmbedding(const cv::Mat& image, const std::array<cv::Point2f, 5>& landmarks);

    // Get L2-normalized embedding with automatic alignment
    [[nodiscard]] Embedding getNormalizedEmbedding(const cv::Mat& image, const std::array<cv::Point2f, 5>& landmarks);

    [[nodiscard]] cv::Size getInputSize() const noexcept { return config_.input_size; }

private:
    cv::dnn::Net net_;
    RecognizerConfig config_;

    [[nodiscard]] cv::Mat preprocess(const cv::Mat& face_image);
};

}  // namespace uniface

#endif  // UNIFACE_RECOGNIZER_HPP_
