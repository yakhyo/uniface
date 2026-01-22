#include "uniface/analyzer.hpp"

#include <stdexcept>

namespace uniface {

FaceAnalyzer& FaceAnalyzer::loadDetector(const std::string& path, const DetectorConfig& config) {
    detector_ = std::make_unique<RetinaFace>(
        path, config.conf_thresh, config.nms_thresh, config.input_size
    );
    return *this;
}

FaceAnalyzer& FaceAnalyzer::loadRecognizer(
    const std::string& path, const RecognizerConfig& config
) {
    recognizer_ = std::make_unique<ArcFace>(path, config);
    return *this;
}

FaceAnalyzer& FaceAnalyzer::loadLandmarker(
    const std::string& path, const LandmarkerConfig& config
) {
    landmarker_ = std::make_unique<Landmark106>(path, config);
    return *this;
}

std::vector<AnalyzedFace> FaceAnalyzer::analyze(const cv::Mat& image) {
    if (!detector_) {
        throw std::runtime_error("FaceAnalyzer: detector not loaded. Call loadDetector() first.");
    }

    auto faces = detector_->detect(image);

    std::vector<AnalyzedFace> results;
    results.reserve(faces.size());

    for (const auto& face : faces) {
        AnalyzedFace result;
        result.face = face;

        if (landmarker_) {
            result.landmarks = landmarker_->getLandmarks(image, face.bbox);
        }
        if (recognizer_) {
            result.embedding = recognizer_->getNormalizedEmbedding(image, face.landmarks);
        }

        results.push_back(std::move(result));
    }

    return results;
}

}  // namespace uniface
