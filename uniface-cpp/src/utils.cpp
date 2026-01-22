#include "uniface/utils.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace uniface {

cv::Mat alignFace(const cv::Mat& image, const std::array<cv::Point2f, 5>& landmarks, cv::Size output_size) {
    const float ratio = static_cast<float>(output_size.width) / 112.0f;

    std::vector<cv::Point2f> dst_points(5);
    for (int i = 0; i < 5; ++i) {
        dst_points[i].x = kReferenceAlignment[static_cast<size_t>(i) * 2] * ratio;
        dst_points[i].y = kReferenceAlignment[static_cast<size_t>(i) * 2 + 1] * ratio;
    }

    std::vector<cv::Point2f> src_points(landmarks.begin(), landmarks.end());
    cv::Mat transform = cv::estimateAffinePartial2D(src_points, dst_points);

    if (transform.empty()) {
        cv::Mat resized;
        cv::resize(image, resized, output_size);
        return resized;
    }

    cv::Mat aligned;
    cv::warpAffine(image, aligned, transform, output_size, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

    return aligned;
}

float cosineSimilarity(const Embedding& a, const Embedding& b) noexcept {
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    const float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denom < 1e-8f) {
        return 0.0f;
    }

    return dot / denom;
}

float letterboxResize(const cv::Mat& src, cv::Mat& dst, cv::Size target_size) {
    const auto src_height = static_cast<float>(src.rows);
    const auto src_width = static_cast<float>(src.cols);
    const auto target_height = static_cast<float>(target_size.height);
    const auto target_width = static_cast<float>(target_size.width);

    const float im_ratio = src_height / src_width;
    const float model_ratio = target_height / target_width;

    int new_width = 0;
    int new_height = 0;

    if (im_ratio > model_ratio) {
        new_height = static_cast<int>(target_height);
        new_width = static_cast<int>(static_cast<float>(new_height) / im_ratio);
    } else {
        new_width = static_cast<int>(target_width);
        new_height = static_cast<int>(static_cast<float>(new_width) * im_ratio);
    }

    const float resize_factor = static_cast<float>(new_height) / src_height;

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_width, new_height));

    dst = cv::Mat::zeros(target_size, src.type());
    resized.copyTo(dst(cv::Rect(0, 0, new_width, new_height)));

    return resize_factor;
}

}  // namespace uniface
