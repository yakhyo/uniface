#include "uniface/landmarker.hpp"

#include <cmath>

#include <opencv2/imgproc.hpp>

namespace uniface {

namespace {

constexpr int kNumLandmarks = 106;

/// Compute center-based alignment transform for face crop
cv::Mat computeCenterTransform(const cv::Point2f& center, float scale, int output_size) {
    // Build 2x3 affine transform matrix:
    // [scale,  0,     -center.x * scale + output_size/2]
    // [0,      scale, -center.y * scale + output_size/2]
    cv::Mat transform = cv::Mat::zeros(2, 3, CV_64F);

    transform.at<double>(0, 0) = scale;
    transform.at<double>(1, 1) = scale;
    transform.at<double>(0, 2) = -center.x * scale + output_size / 2.0;
    transform.at<double>(1, 2) = -center.y * scale + output_size / 2.0;

    return transform;
}

}  // namespace

Landmark106::Landmark106(const std::string& model_path, const LandmarkerConfig& config)
    : net_(cv::dnn::readNetFromONNX(model_path))
    , config_(config) {}

cv::Mat Landmark106::preprocess(const cv::Mat& image, const cv::Rect2f& bbox, cv::Mat& transform) {
    // Calculate bbox center and scale
    const float width = bbox.width;
    const float height = bbox.height;
    const float center_x = bbox.x + width / 2.0f;
    const float center_y = bbox.y + height / 2.0f;

    // Scale factor: input_size / (max_dim * 1.5)
    const float max_dim = std::max(width, height);
    const float scale = static_cast<float>(config_.input_size.width) / (max_dim * 1.5f);

    // Compute affine transform
    transform =
        computeCenterTransform(cv::Point2f(center_x, center_y), scale, config_.input_size.width);

    // Apply transform to get aligned face crop
    cv::Mat aligned;
    cv::warpAffine(
        image, aligned, transform, config_.input_size, cv::INTER_LINEAR, cv::BORDER_CONSTANT
    );

    // Create blob (no normalization, just RGB conversion)
    cv::Mat blob = cv::dnn::blobFromImage(
        aligned,
        1.0,                  // No scaling
        config_.input_size,
        cv::Scalar(0, 0, 0),  // No mean subtraction
        true,                 // swapRB: BGR -> RGB
        false                 // crop
    );

    return blob;
}

Landmarks Landmark106::postprocess(const cv::Mat& predictions, const cv::Mat& transform) {
    Landmarks result{};

    // Get raw predictions (212 values = 106 landmarks * 2 coordinates)
    const auto* pred_data = reinterpret_cast<const float*>(predictions.data);

    // Compute inverse transform
    cv::Mat inverse_transform;
    cv::invertAffineTransform(transform, inverse_transform);

    const int input_size = config_.input_size.width;
    const float half_size = static_cast<float>(input_size) / 2.0f;

    for (int i = 0; i < kNumLandmarks; ++i) {
        // Denormalize from [-1, 1] to pixel coordinates
        float x = (pred_data[i * 2 + 0] + 1.0f) * half_size;
        float y = (pred_data[i * 2 + 1] + 1.0f) * half_size;

        // Apply inverse transform to get original image coordinates
        const float orig_x = static_cast<float>(
            inverse_transform.at<double>(0, 0) * x + inverse_transform.at<double>(0, 1) * y +
            inverse_transform.at<double>(0, 2)
        );
        const float orig_y = static_cast<float>(
            inverse_transform.at<double>(1, 0) * x + inverse_transform.at<double>(1, 1) * y +
            inverse_transform.at<double>(1, 2)
        );

        result.points[static_cast<size_t>(i)] = cv::Point2f(orig_x, orig_y);
    }

    return result;
}

Landmarks Landmark106::getLandmarks(const cv::Mat& image, const cv::Rect2f& bbox) {
    // Preprocess
    cv::Mat transform;
    cv::Mat blob = preprocess(image, bbox, transform);

    // Run inference
    net_.setInput(blob);
    cv::Mat output = net_.forward();

    // Postprocess
    return postprocess(output, transform);
}

}  // namespace uniface
