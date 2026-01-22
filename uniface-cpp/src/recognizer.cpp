#include "uniface/recognizer.hpp"

#include "uniface/utils.hpp"

#include <cmath>

#include <opencv2/imgproc.hpp>

namespace uniface {

ArcFace::ArcFace(const std::string& model_path, const RecognizerConfig& config)
    : net_(cv::dnn::readNetFromONNX(model_path))
    , config_(config) {}

cv::Mat ArcFace::preprocess(const cv::Mat& face_image) {
    cv::Mat resized;
    if (face_image.size() != config_.input_size) {
        cv::resize(face_image, resized, config_.input_size);
    } else {
        resized = face_image;
    }

    // Normalize: (pixel - mean) / std, BGR -> RGB
    cv::Mat blob = cv::dnn::blobFromImage(
        resized, 1.0 / config_.input_std, config_.input_size,
        cv::Scalar(config_.input_mean, config_.input_mean, config_.input_mean), true, false
    );

    return blob;
}

Embedding ArcFace::getEmbedding(const cv::Mat& aligned_face) {
    cv::Mat blob = preprocess(aligned_face);

    net_.setInput(blob);
    cv::Mat output = net_.forward();

    Embedding embedding{};
    const auto* output_data = reinterpret_cast<const float*>(output.data);
    const size_t embedding_size = std::min(static_cast<size_t>(output.total()), embedding.size());

    for (size_t i = 0; i < embedding_size; ++i) {
        embedding[i] = output_data[i];
    }

    return embedding;
}

Embedding ArcFace::getEmbedding(const cv::Mat& image, const std::array<cv::Point2f, 5>& landmarks) {
    cv::Mat aligned = alignFace(image, landmarks, config_.input_size);
    return getEmbedding(aligned);
}

Embedding ArcFace::getNormalizedEmbedding(const cv::Mat& image, const std::array<cv::Point2f, 5>& landmarks) {
    Embedding embedding = getEmbedding(image, landmarks);

    // L2 normalize
    float norm = 0.0f;
    for (const float val : embedding) {
        norm += val * val;
    }
    norm = std::sqrt(norm);

    if (norm > 1e-8f) {
        for (float& val : embedding) {
            val /= norm;
        }
    }

    return embedding;
}

}  // namespace uniface
