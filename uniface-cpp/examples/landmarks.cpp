#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <uniface/uniface.hpp>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <detector_model> <landmark_model> <image_path>"
                  << std::endl;
        std::cout << "\nDetects 106-point facial landmarks and saves visualization." << std::endl;
        return 1;
    }

    const std::string detector_path = argv[1];
    const std::string landmark_path = argv[2];
    const std::string image_path = argv[3];

    try {
        // Load models
        uniface::RetinaFace detector(detector_path);
        uniface::Landmark106 landmarker(landmark_path);

        // Load image
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Failed to load image: " << image_path << std::endl;
            return 1;
        }

        // Detect faces
        auto faces = detector.detect(image);
        std::cout << "Detected " << faces.size() << " face(s)" << std::endl;

        // Process each face
        for (size_t i = 0; i < faces.size(); ++i) {
            const auto& face = faces[i];

            // Draw bounding box
            cv::rectangle(image, face.bbox, cv::Scalar(0, 255, 0), 2);

            // Get 106-point landmarks
            auto landmarks = landmarker.getLandmarks(image, face.bbox);

            // Draw all 106 points
            for (const auto& pt : landmarks.points) {
                cv::circle(image, pt, 1, cv::Scalar(0, 255, 255), -1);
            }

            std::cout << "Face " << (i + 1) << ": 106 landmarks detected" << std::endl;
        }

        // Save result
        cv::imwrite("landmarks_result.jpg", image);
        std::cout << "Saved result to landmarks_result.jpg" << std::endl;

    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV Error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
