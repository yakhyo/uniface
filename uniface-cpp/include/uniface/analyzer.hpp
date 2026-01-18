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

/// Result of face analysis containing detection, optional landmarks, and optional embedding
struct AnalyzedFace {
    Face face;                           ///< Detection result (bbox, confidence, 5-point landmarks)
    std::optional<Landmarks> landmarks;  ///< 106-point landmarks (if landmarker loaded)
    std::optional<Embedding> embedding;  ///< Face embedding (if recognizer loaded)
};

/// Unified face analysis combining detection, recognition, and landmarks
/// Allows loading only the components needed for your use case
class FaceAnalyzer {
public:
    FaceAnalyzer() = default;
    ~FaceAnalyzer() = default;

    // Non-copyable, movable
    FaceAnalyzer(const FaceAnalyzer&) = delete;
    FaceAnalyzer& operator=(const FaceAnalyzer&) = delete;
    FaceAnalyzer(FaceAnalyzer&&) = default;
    FaceAnalyzer& operator=(FaceAnalyzer&&) = default;

    /// Load face detector (required for analyze())
    /// @param path Path to detector ONNX model
    /// @param config Detector configuration
    /// @return Reference to this for method chaining
    FaceAnalyzer& loadDetector(
        const std::string& path, const DetectorConfig& config = DetectorConfig{}
    );

    /// Load face recognizer (optional)
    /// @param path Path to recognizer ONNX model
    /// @param config Recognizer configuration
    /// @return Reference to this for method chaining
    FaceAnalyzer& loadRecognizer(
        const std::string& path, const RecognizerConfig& config = RecognizerConfig{}
    );

    /// Load 106-point landmark detector (optional)
    /// @param path Path to landmark ONNX model
    /// @param config Landmarker configuration
    /// @return Reference to this for method chaining
    FaceAnalyzer& loadLandmarker(
        const std::string& path, const LandmarkerConfig& config = LandmarkerConfig{}
    );

    /// Analyze faces in an image
    /// @param image Input BGR image
    /// @return Vector of analyzed faces with available information
    /// @throws std::runtime_error if detector is not loaded
    [[nodiscard]] std::vector<AnalyzedFace> analyze(const cv::Mat& image);

    // Component availability checks
    [[nodiscard]] bool hasDetector() const noexcept { return detector_ != nullptr; }

    [[nodiscard]] bool hasRecognizer() const noexcept { return recognizer_ != nullptr; }

    [[nodiscard]] bool hasLandmarker() const noexcept { return landmarker_ != nullptr; }

    // Direct component access (for power users)
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
