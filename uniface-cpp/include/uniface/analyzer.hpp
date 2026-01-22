#ifndef UNIFACE_ANALYZER_HPP_
#define UNIFACE_ANALYZER_HPP_

#include "uniface/detector.hpp"
#include "uniface/landmarker.hpp"
#include "uniface/recognizer.hpp"
#include "uniface/types.hpp"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace uniface {

// Result of face analysis
struct AnalyzedFace {
    Face face;                           // detection result (bbox, confidence, 5-point landmarks)
    std::optional<Landmarks> landmarks;  // 106-point landmarks (if landmarker loaded)
    std::optional<Embedding> embedding;  // face embedding (if recognizer loaded)
};

// Unified face analysis combining detection, recognition, and landmarks
class FaceAnalyzer {
public:
    FaceAnalyzer() = default;
    ~FaceAnalyzer() = default;

    FaceAnalyzer(const FaceAnalyzer&) = delete;
    FaceAnalyzer& operator=(const FaceAnalyzer&) = delete;
    FaceAnalyzer(FaceAnalyzer&&) = default;
    FaceAnalyzer& operator=(FaceAnalyzer&&) = default;

    // Load components (returns *this for chaining)
    FaceAnalyzer& loadDetector(const std::string& path, const DetectorConfig& config = DetectorConfig{});
    FaceAnalyzer& loadRecognizer(const std::string& path, const RecognizerConfig& config = RecognizerConfig{});
    FaceAnalyzer& loadLandmarker(const std::string& path, const LandmarkerConfig& config = LandmarkerConfig{});

    // Analyze faces in BGR image (throws if detector not loaded)
    [[nodiscard]] std::vector<AnalyzedFace> analyze(const cv::Mat& image);

    // Component checks
    [[nodiscard]] bool hasDetector() const noexcept { return detector_ != nullptr; }
    [[nodiscard]] bool hasRecognizer() const noexcept { return recognizer_ != nullptr; }
    [[nodiscard]] bool hasLandmarker() const noexcept { return landmarker_ != nullptr; }

    // Direct component access
    [[nodiscard]] RetinaFace* detector() noexcept { return detector_.get(); }
    [[nodiscard]] ArcFace* recognizer() noexcept { return recognizer_.get(); }
    [[nodiscard]] Landmark106* landmarker() noexcept { return landmarker_.get(); }
    [[nodiscard]] const RetinaFace* detector() const noexcept { return detector_.get(); }
    [[nodiscard]] const ArcFace* recognizer() const noexcept { return recognizer_.get(); }
    [[nodiscard]] const Landmark106* landmarker() const noexcept { return landmarker_.get(); }

private:
    std::unique_ptr<RetinaFace> detector_;
    std::unique_ptr<ArcFace> recognizer_;
    std::unique_ptr<Landmark106> landmarker_;
};

}  // namespace uniface

#endif  // UNIFACE_ANALYZER_HPP_
