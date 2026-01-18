/**
 * @file types.hpp
 * @brief Common data types for uniface
 */

#ifndef UNIFACE_TYPES_HPP_
#define UNIFACE_TYPES_HPP_

#include <vector>

#include <opencv2/core.hpp>

namespace uniface {

/**
 * @brief Detected face data structure
 */
struct Face {
    cv::Rect2f bbox;   ///< Bounding box [x, y, width, height] in original image coordinates
    float confidence;  ///< Detection confidence score [0.0, 1.0]
    std::vector<cv::Point2f> landmarks;  ///< 5 facial landmarks:
                                         ///< [left_eye, right_eye, nose, left_mouth, right_mouth]
};

}  // namespace uniface

#endif  // UNIFACE_TYPES_HPP_
