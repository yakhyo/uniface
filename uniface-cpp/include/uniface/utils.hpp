#ifndef UNIFACE_UTILS_HPP_
#define UNIFACE_UTILS_HPP_

#include "uniface/types.hpp"

#include <array>
#include <cmath>

#include <opencv2/core.hpp>

namespace uniface {

// Reference 5-point landmarks for ArcFace alignment (112x112)
inline constexpr std::array<float, 10> kReferenceAlignment = {
    38.2946f, 51.6963f,  // left eye
    73.5318f, 51.5014f,  // right eye
    56.0252f, 71.7366f,  // nose
    41.5493f, 92.3655f,  // left mouth
    70.7299f, 92.2041f   // right mouth
};

// Align face using 5-point landmarks (default 112x112 for ArcFace)
[[nodiscard]] cv::Mat alignFace(
    const cv::Mat& image,
    const std::array<cv::Point2f, 5>& landmarks,
    cv::Size output_size = cv::Size(112, 112)
);

// Cosine similarity between embeddings, returns [-1, 1]
[[nodiscard]] float cosineSimilarity(const Embedding& a, const Embedding& b) noexcept;

// Apply 2x3 affine transform to points
template <size_t N>
[[nodiscard]] std::array<cv::Point2f, N> transformPoints2D(
    const std::array<cv::Point2f, N>& points, const cv::Mat& transform
) {
    std::array<cv::Point2f, N> result{};
    for (size_t i = 0; i < N; ++i) {
        const float x = points[i].x;
        const float y = points[i].y;
        result[i].x = static_cast<float>(
            transform.at<double>(0, 0) * x + transform.at<double>(0, 1) * y +
            transform.at<double>(0, 2)
        );
        result[i].y = static_cast<float>(
            transform.at<double>(1, 0) * x + transform.at<double>(1, 1) * y +
            transform.at<double>(1, 2)
        );
    }
    return result;
}

// Letterbox resize preserving aspect ratio, returns scale factor
[[nodiscard]] float letterboxResize(const cv::Mat& src, cv::Mat& dst, cv::Size target_size);

}  // namespace uniface

#endif  // UNIFACE_UTILS_HPP_
