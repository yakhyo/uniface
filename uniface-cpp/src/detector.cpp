/**
 * @file detector.cpp
 * @brief RetinaFace detector implementation
 */

#include "uniface/detector.hpp"

#include <cmath>
#include <iostream>

#include <opencv2/imgproc.hpp>

namespace uniface {

namespace {

// Model configuration constants
constexpr std::array<int, 3> kFeatureStrides = {8, 16, 32};
constexpr std::array<float, 2> kVariance = {0.1f, 0.2f};
constexpr int kNumLandmarks = 5;

// BGR mean values for image normalization
constexpr float kMeanB = 104.0f;
constexpr float kMeanG = 117.0f;
constexpr float kMeanR = 123.0f;

// Anchor min sizes for each feature map level
const std::vector<std::vector<int>> kMinSizes = {
    { 16,  32},
    { 64, 128},
    {256, 512}
};

/**
 * @brief Resize image while preserving aspect ratio with letterbox padding
 */
float letterboxResize(const cv::Mat& src, cv::Mat& dst, const cv::Size& target_size) {
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

    dst = cv::Mat::zeros(target_size, CV_8UC3);
    resized.copyTo(dst(cv::Rect(0, 0, new_width, new_height)));

    return resize_factor;
}

}  // namespace

RetinaFace::RetinaFace(
    const std::string& model_path, float conf_thresh, float nms_thresh, cv::Size input_size
)
    : net_(cv::dnn::readNetFromONNX(model_path))
    , confidence_threshold_(conf_thresh)
    , nms_threshold_(nms_thresh)
    , input_size_(input_size) {
    generateAnchors();
}

void RetinaFace::generateAnchors() {
    anchors_.clear();

    // Pre-calculate approximate anchor count for reservation
    size_t estimated_anchors = 0;
    for (size_t k = 0; k < kFeatureStrides.size(); ++k) {
        const int step = kFeatureStrides[k];
        const auto feature_h = static_cast<size_t>(
            std::ceil(static_cast<float>(input_size_.height) / static_cast<float>(step))
        );
        const auto feature_w = static_cast<size_t>(
            std::ceil(static_cast<float>(input_size_.width) / static_cast<float>(step))
        );
        estimated_anchors += feature_h * feature_w * kMinSizes[k].size();
    }
    anchors_.reserve(estimated_anchors);

    // Generate anchors for each feature map level
    for (size_t k = 0; k < kFeatureStrides.size(); ++k) {
        const int step = kFeatureStrides[k];
        const int feature_h = static_cast<int>(
            std::ceil(static_cast<float>(input_size_.height) / static_cast<float>(step))
        );
        const int feature_w = static_cast<int>(
            std::ceil(static_cast<float>(input_size_.width) / static_cast<float>(step))
        );

        for (int i = 0; i < feature_h; ++i) {
            for (int j = 0; j < feature_w; ++j) {
                for (const int min_size : kMinSizes[k]) {
                    const float s_kx = static_cast<float>(min_size) /
                                       static_cast<float>(input_size_.height);
                    const float s_ky = static_cast<float>(min_size) /
                                       static_cast<float>(input_size_.width);
                    const float cx = (static_cast<float>(j) + 0.5f) * static_cast<float>(step) /
                                     static_cast<float>(input_size_.height);
                    const float cy = (static_cast<float>(i) + 0.5f) * static_cast<float>(step) /
                                     static_cast<float>(input_size_.width);

                    anchors_.push_back({cx, cy, s_kx, s_ky});
                }
            }
        }
    }
}

std::vector<Face> RetinaFace::detect(const cv::Mat& image) {
    // Preprocess image
    cv::Mat input_blob;
    const float resize_factor = letterboxResize(image, input_blob, input_size_);

    // Create blob with mean subtraction
    const cv::Mat blob = cv::dnn::blobFromImage(
        input_blob, 1.0, cv::Size(), cv::Scalar(kMeanB, kMeanG, kMeanR), false, false
    );

    // Run inference
    net_.setInput(blob);
    const auto output_names = net_.getUnconnectedOutLayersNames();
    std::vector<cv::Mat> outputs;
    net_.forward(outputs, output_names);

    // Validate output count
    if (outputs.size() < 3) {
        std::cerr << "Error: Model output count mismatch. Expected at least 3, got "
                  << outputs.size() << std::endl;
        return {};
    }

    // Identify outputs by shape: loc(N,4), conf(N,2), landmarks(N,10)
    cv::Mat loc_output;
    cv::Mat conf_output;
    cv::Mat land_output;

    for (const auto& output : outputs) {
        switch (output.size[2]) {
            case 4:
                loc_output = output;
                break;
            case 2:
                conf_output = output;
                break;
            case 10:
                land_output = output;
                break;
            default:
                break;
        }
    }

    // Fallback to positional outputs if shape matching failed
    if (loc_output.empty())
        loc_output = outputs[0];
    if (conf_output.empty())
        conf_output = outputs[1];
    if (land_output.empty())
        land_output = outputs[2];

    // Get raw data pointers
    const auto* loc_data = reinterpret_cast<const float*>(loc_output.data);
    const auto* conf_data = reinterpret_cast<const float*>(conf_output.data);
    const auto* land_data = reinterpret_cast<const float*>(land_output.data);

    const auto num_priors = static_cast<size_t>(loc_output.size[1]);

    // Validate anchor count
    if (num_priors != anchors_.size()) {
        std::cerr << "Error: Anchor count mismatch! Expected " << anchors_.size()
                  << " anchors but model output has " << num_priors << " priors.\n"
                  << "This usually means the input size doesn't match the model's "
                  << "expected size." << std::endl;
        return {};
    }

    // Decode detections
    std::vector<cv::Rect2f> decoded_boxes;
    std::vector<float> scores;
    std::vector<std::vector<cv::Point2f>> decoded_landmarks;

    decoded_boxes.reserve(num_priors);
    scores.reserve(num_priors);
    decoded_landmarks.reserve(num_priors);

    const auto scale_w = static_cast<float>(input_size_.width);
    const auto scale_h = static_cast<float>(input_size_.height);

    for (size_t i = 0; i < num_priors; ++i) {
        const float score = conf_data[i * 2 + 1];
        if (score < confidence_threshold_) {
            continue;
        }

        // Get anchor parameters
        const float px = anchors_[i][0];
        const float py = anchors_[i][1];
        const float pw = anchors_[i][2];
        const float ph = anchors_[i][3];

        // Decode bounding box
        const float dx = loc_data[i * 4 + 0];
        const float dy = loc_data[i * 4 + 1];
        const float dw = loc_data[i * 4 + 2];
        const float dh = loc_data[i * 4 + 3];

        const float cx = px + dx * kVariance[0] * pw;
        const float cy = py + dy * kVariance[0] * ph;
        const float w = pw * std::exp(dw * kVariance[1]);
        const float h = ph * std::exp(dh * kVariance[1]);

        // Convert center format to corner format and scale to original image
        const float x1 = (cx - w / 2.0f) * scale_w / resize_factor;
        const float y1 = (cy - h / 2.0f) * scale_h / resize_factor;
        const float x2 = (cx + w / 2.0f) * scale_w / resize_factor;
        const float y2 = (cy + h / 2.0f) * scale_h / resize_factor;

        decoded_boxes.emplace_back(x1, y1, x2 - x1, y2 - y1);
        scores.push_back(score);

        // Decode landmarks
        std::vector<cv::Point2f> landmarks;
        landmarks.reserve(kNumLandmarks);

        for (int k = 0; k < kNumLandmarks; ++k) {
            const float ldx = land_data[i * 10 + static_cast<size_t>(k) * 2 + 0];
            const float ldy = land_data[i * 10 + static_cast<size_t>(k) * 2 + 1];
            const float lx = (px + ldx * kVariance[0] * pw) * scale_w / resize_factor;
            const float ly = (py + ldy * kVariance[0] * ph) * scale_h / resize_factor;
            landmarks.emplace_back(lx, ly);
        }
        decoded_landmarks.push_back(std::move(landmarks));
    }

    // Apply Non-Maximum Suppression
    std::vector<cv::Rect2d> boxes_for_nms;
    boxes_for_nms.reserve(decoded_boxes.size());

    for (const auto& box : decoded_boxes) {
        boxes_for_nms.emplace_back(box.x, box.y, box.width, box.height);
    }

    std::vector<int> nms_indices;
    cv::dnn::NMSBoxes(boxes_for_nms, scores, confidence_threshold_, nms_threshold_, nms_indices);

    // Build final results
    std::vector<Face> results;
    results.reserve(nms_indices.size());

    for (const int idx : nms_indices) {
        const auto uidx = static_cast<size_t>(idx);
        results.push_back({decoded_boxes[uidx], scores[uidx], decoded_landmarks[uidx]});
    }

    return results;
}

}  // namespace uniface
