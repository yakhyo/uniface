#include <iomanip>
#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <uniface/uniface.hpp>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0]
                  << " <detector_model> <image_path> [recognizer_model] [landmark_model]"
                  << std::endl;
        std::cout << "\nAnalyzes faces in an image using available models." << std::endl;
        std::cout << "  - detector_model: Required. Path to face detector ONNX model." << std::endl;
        std::cout << "  - recognizer_model: Optional. Path to face recognizer ONNX model."
                  << std::endl;
        std::cout << "  - landmark_model: Optional. Path to 106-point landmark ONNX model."
                  << std::endl;
        return 1;
    }

    const std::string detector_path = argv[1];
    const std::string image_path = argv[2];
    const std::string recognizer_path = (argc > 3) ? argv[3] : "";
    const std::string landmark_path = (argc > 4) ? argv[4] : "";

    try {
        // Create analyzer and load components
        uniface::FaceAnalyzer analyzer;

        std::cout << "Loading detector: " << detector_path << std::endl;
        analyzer.loadDetector(detector_path);

        if (!recognizer_path.empty()) {
            std::cout << "Loading recognizer: " << recognizer_path << std::endl;
            analyzer.loadRecognizer(recognizer_path);
        }

        if (!landmark_path.empty()) {
            std::cout << "Loading landmarker: " << landmark_path << std::endl;
            analyzer.loadLandmarker(landmark_path);
        }

        // Load image
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Failed to load image: " << image_path << std::endl;
            return 1;
        }

        std::cout << "\nAnalyzing image..." << std::endl;

        // Analyze faces
        auto results = analyzer.analyze(image);

        std::cout << "Found " << results.size() << " face(s)\n" << std::endl;

        // Process each face
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& result = results[i];

            std::cout << "Face " << (i + 1) << ":" << std::endl;
            std::cout << "  BBox: [" << result.face.bbox.x << ", " << result.face.bbox.y << ", "
                      << result.face.bbox.width << ", " << result.face.bbox.height << "]"
                      << std::endl;
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "  Confidence: " << result.face.confidence << std::endl;

            // Draw bounding box
            cv::rectangle(image, result.face.bbox, cv::Scalar(0, 255, 0), 2);

            // Draw 5-point landmarks from detector
            for (const auto& pt : result.face.landmarks) {
                cv::circle(image, pt, 3, cv::Scalar(0, 0, 255), -1);
            }

            // If 106-point landmarks available
            if (result.landmarks) {
                std::cout << "  Landmarks: 106 points detected" << std::endl;
                for (const auto& pt : result.landmarks->points) {
                    cv::circle(image, pt, 1, cv::Scalar(0, 255, 255), -1);
                }
            }

            // If embedding available
            if (result.embedding) {
                // Show first few values of embedding
                std::cout << "  Embedding: [";
                for (size_t j = 0; j < 5; ++j) {
                    std::cout << (*result.embedding)[j];
                    if (j < 4)
                        std::cout << ", ";
                }
                std::cout << ", ... ] (512-dim)" << std::endl;
            }

            std::cout << std::endl;
        }

        // Save result
        cv::imwrite("analyzer_result.jpg", image);
        std::cout << "Saved result to analyzer_result.jpg" << std::endl;

    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV Error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
