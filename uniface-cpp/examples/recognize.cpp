#include <iomanip>
#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <uniface/uniface.hpp>

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cout << "Usage: " << argv[0]
                  << " <detector_model> <recognizer_model> <image1> <image2>" << std::endl;
        std::cout << "\nCompares faces from two images and outputs similarity score." << std::endl;
        return 1;
    }

    const std::string detector_path = argv[1];
    const std::string recognizer_path = argv[2];
    const std::string image1_path = argv[3];
    const std::string image2_path = argv[4];

    try {
        // Load models
        uniface::RetinaFace detector(detector_path);
        uniface::ArcFace recognizer(recognizer_path);

        // Load images
        cv::Mat image1 = cv::imread(image1_path);
        cv::Mat image2 = cv::imread(image2_path);

        if (image1.empty()) {
            std::cerr << "Failed to load image: " << image1_path << std::endl;
            return 1;
        }
        if (image2.empty()) {
            std::cerr << "Failed to load image: " << image2_path << std::endl;
            return 1;
        }

        // Detect faces
        auto faces1 = detector.detect(image1);
        auto faces2 = detector.detect(image2);

        if (faces1.empty()) {
            std::cerr << "No face detected in image1" << std::endl;
            return 1;
        }
        if (faces2.empty()) {
            std::cerr << "No face detected in image2" << std::endl;
            return 1;
        }

        std::cout << "Detected " << faces1.size() << " face(s) in image1" << std::endl;
        std::cout << "Detected " << faces2.size() << " face(s) in image2" << std::endl;

        // Get embeddings for first face in each image
        auto embedding1 = recognizer.getNormalizedEmbedding(image1, faces1[0].landmarks);
        auto embedding2 = recognizer.getNormalizedEmbedding(image2, faces2[0].landmarks);

        // Compute similarity
        float similarity = uniface::cosineSimilarity(embedding1, embedding2);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "\nCosine Similarity: " << similarity << std::endl;

        // Interpretation
        if (similarity > 0.4f) {
            std::cout << "Result: Same person (similarity > 0.4)" << std::endl;
        } else {
            std::cout << "Result: Different persons (similarity <= 0.4)" << std::endl;
        }

    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV Error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
