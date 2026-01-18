#ifndef UNIFACE_UTILS_HPP_
#define UNIFACE_UTILS_HPP_

#include "uniface/types.hpp"

#include <array>
#include <cmath>

#include <opencv2/core.hpp>

namespace uniface {

// Standard 5-point facial landmark reference for ArcFace alignment (112x112)
inline constexpr std::array<float, 10> kReferenceAlignment = {
    38.2946f,
    51.6963f,  // left eye
    73.5318f,
    51.5014f,  // right eye
    56.0252f,
    71.7366f,  // nose
    41.5493f,
    92.3655f,  // left mouth
    70.7299f,
    92.2041f   // right mouth
};

/// Align face using 5-point landmarks for recognition
/// @param image Source BGR image
/// @param landmarks 5-point facial landmarks from detector
/// @param output_size Output aligned face size (default 112x112 for ArcFace)
/// @return Aligned face image
[[nodiscard]] cv::Mat alignFace(
    const cv::Mat& image,
    const std::array<cv::Point2f, 5>& landmarks,
    cv::Size output_size = cv::Size(112, 112)
);

/// Compute cosine similarity between two face embeddings
/// @param a First embedding vector
/// @param b Second embedding vector
/// @return Cosine similarity score in range [-1, 1]
[[nodiscard]] float cosineSimilarity(const Embedding& a, const Embedding& b) noexcept;

/// Apply 2D affine transformation to points
/// @param points Input points
/// @param transform 2x3 affine transformation matrix
/// @return Transformed points
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

/// Resize image with letterbox padding (aspect ratio preserved)
/// @param src Source image
/// @param dst Destination image
/// @param target_size Target size
/// @return Resize scale factor
[[nodiscard]] float letterboxResize(const cv::Mat& src, cv::Mat& dst, cv::Size target_size);

}  // namespace uniface

#endif  // UNIFACE_UTILS_HPP_
