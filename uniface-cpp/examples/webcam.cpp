#include <chrono>
#include <iostream>
#include <memory>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <uniface/uniface.hpp>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <detector_model> [landmark_model] [camera_id]"
                  << std::endl;
        std::cout << "\nArguments:" << std::endl;
        std::cout << "  detector_model : Path to face detector ONNX model (required)" << std::endl;
        std::cout << "  landmark_model : Path to 106-point landmark ONNX model (optional)"
                  << std::endl;
        std::cout << "  camera_id      : Camera device ID, default 0 (optional)" << std::endl;
        std::cout << "\nExamples:" << std::endl;
        std::cout << "  " << argv[0] << " detector.onnx" << std::endl;
        std::cout << "  " << argv[0] << " detector.onnx landmark.onnx" << std::endl;
        std::cout << "  " << argv[0] << " detector.onnx landmark.onnx 1" << std::endl;
        return 1;
    }

    const std::string detector_path = argv[1];
    std::string landmark_path;
    int camera_id = 0;

    // Parse arguments - landmark_model is optional
    if (argc >= 3) {
        // Check if argv[2] is a number (camera_id) or a path (landmark_model)
        if (std::isdigit(argv[2][0]) && strlen(argv[2]) <= 2) {
            camera_id = std::atoi(argv[2]);
        } else {
            landmark_path = argv[2];
            if (argc >= 4) {
                camera_id = std::atoi(argv[3]);
            }
        }
    }

    try {
        // Load detector
        std::cout << "Loading detector: " << detector_path << std::endl;
        uniface::RetinaFace detector(detector_path);
        std::cout << "Detector loaded!" << std::endl;

        // Load landmark model if provided
        std::unique_ptr<uniface::Landmark106> landmarker;
        if (!landmark_path.empty()) {
            std::cout << "Loading landmarker: " << landmark_path << std::endl;
            landmarker = std::make_unique<uniface::Landmark106>(landmark_path);
            std::cout << "Landmarker loaded!" << std::endl;
        }

        // Open camera
        cv::VideoCapture cap(camera_id);
        if (!cap.isOpened()) {
            std::cerr << "Error: Cannot open camera " << camera_id << std::endl;
            return 1;
        }

        const int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        const int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        std::cout << "\nCamera opened: " << frame_width << "x" << frame_height << std::endl;
        std::cout << "Press 'q' to quit, 's' to save screenshot, 'l' to toggle landmarks"
                  << std::endl;

        cv::Mat frame;
        int frame_count = 0;
        double total_time = 0.0;
        bool show_landmarks = true;  // Toggle for 106-point landmarks

        while (true) {
            cap >> frame;
            if (frame.empty()) {
                std::cerr << "Error: Empty frame captured" << std::endl;
                break;
            }

            const auto start = std::chrono::high_resolution_clock::now();

            // Detect faces
            const auto faces = detector.detect(frame);

            // Get 106-point landmarks if available
            std::vector<uniface::Landmarks> all_landmarks;
            if (landmarker && show_landmarks) {
                all_landmarks.reserve(faces.size());
                for (const auto& face : faces) {
                    all_landmarks.push_back(landmarker->getLandmarks(frame, face.bbox));
                }
            }

            const auto end = std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double, std::milli> elapsed = end - start;
            const double inference_time = elapsed.count();

            ++frame_count;
            total_time += inference_time;
            const double avg_time = total_time / static_cast<double>(frame_count);
            const double fps = 1000.0 / avg_time;

            // Draw results
            for (size_t i = 0; i < faces.size(); ++i) {
                const auto& face = faces[i];

                // Draw bounding box
                cv::rectangle(frame, face.bbox, cv::Scalar(0, 255, 0), 2);

                // Draw 5-point landmarks from detector
                for (size_t j = 0; j < face.landmarks.size(); ++j) {
                    cv::Scalar color;
                    if (j < 2) {
                        color = cv::Scalar(255, 0, 0);  // Eyes - Blue
                    } else if (j == 2) {
                        color = cv::Scalar(0, 255, 0);  // Nose - Green
                    } else {
                        color = cv::Scalar(0, 0, 255);  // Mouth - Red
                    }
                    cv::circle(frame, face.landmarks[j], 3, color, -1);
                }

                // Draw 106-point landmarks if available
                if (i < all_landmarks.size()) {
                    const auto& lm = all_landmarks[i];

                    // Draw all 106 points
                    for (const auto& pt : lm.points) {
                        cv::circle(frame, pt, 1, cv::Scalar(0, 255, 255), -1);
                    }
                }

                // Draw confidence
                const std::string conf_text = cv::format("%.2f", face.confidence);
                const cv::Point text_org(
                    static_cast<int>(face.bbox.x), static_cast<int>(face.bbox.y) - 5
                );
                cv::putText(
                    frame,
                    conf_text,
                    text_org,
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar(0, 255, 0),
                    1
                );
            }

            // Draw info overlay
            std::string mode = landmarker
                                 ? (show_landmarks ? "Detection + 106 Landmarks" : "Detection Only")
                                 : "Detection Only";
            const std::string info_text = cv::format(
                "FPS: %.1f | Faces: %zu | Time: %.1fms", fps, faces.size(), inference_time
            );
            cv::putText(
                frame,
                info_text,
                cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX,
                0.7,
                cv::Scalar(0, 255, 0),
                2
            );
            cv::putText(
                frame,
                mode,
                cv::Point(10, 60),
                cv::FONT_HERSHEY_SIMPLEX,
                0.6,
                cv::Scalar(255, 255, 0),
                2
            );

            cv::imshow("Uniface - Face Detection & Landmarks", frame);

            const char key = static_cast<char>(cv::waitKey(1));
            if (key == 'q' || key == 27) {
                break;
            } else if (key == 's') {
                const std::string filename = cv::format("screenshot_%d.jpg", frame_count);
                cv::imwrite(filename, frame);
                std::cout << "Screenshot saved: " << filename << std::endl;
            } else if (key == 'l' && landmarker) {
                show_landmarks = !show_landmarks;
                std::cout << "106-point landmarks: " << (show_landmarks ? "ON" : "OFF")
                          << std::endl;
            }
        }

        cap.release();
        cv::destroyAllWindows();

        std::cout << "\n=== Statistics ===" << std::endl;
        std::cout << "Total frames: " << frame_count << std::endl;
        std::cout << "Average inference time: " << (total_time / frame_count) << " ms" << std::endl;

    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV Error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
