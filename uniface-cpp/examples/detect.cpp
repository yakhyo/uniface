#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <uniface/uniface.hpp>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <model_path> <image_path>" << std::endl;
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string image_path = argv[2];

    try {
        uniface::RetinaFace detector(model_path);

        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Failed to load image: " << image_path << std::endl;
            return 1;
        }

        const auto faces = detector.detect(image);
        std::cout << "Detected " << faces.size() << " faces." << std::endl;

        // Draw results
        for (const auto& face : faces) {
            cv::rectangle(image, face.bbox, cv::Scalar(0, 255, 0), 2);
            for (const auto& pt : face.landmarks) {
                cv::circle(image, pt, 2, cv::Scalar(0, 0, 255), -1);
            }
        }

        cv::imwrite("result.jpg", image);
        std::cout << "Saved result to result.jpg" << std::endl;

    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV Error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
