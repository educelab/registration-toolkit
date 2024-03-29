#pragma once

/** @file */

#include <vector>

#include <opencv2/core.hpp>

#include "rt/LandmarkRegistrationBase.hpp"

namespace rt
{
/** @brief Pair of matched positions
 *
 * The first element in the pair is a position in the fixed image. The
 * second element is from the moving image.
 */
using LandmarkPair = std::pair<cv::Point2f, cv::Point2f>;

/**
 * @class LandmarkDetector
 * @brief Automatically generate landmark pairs between a two images
 * @author Ali Bertelsman
 *
 * Uses AKAZE feature descriptors to generate pairs of matching key points
 * between two images. To create key points bounded by a region of interest,
 * set the mask for either the static or moving image.
 *
 */
class LandmarkDetector
{
public:
    /** @brief Set the fixed image */
    void setFixedImage(const cv::Mat& img);
    /** @brief Set the fixed image mask */
    void setFixedMask(const cv::Mat& img);
    /** @brief Set the moving image */
    void setMovingImage(const cv::Mat& img);
    /** @brief Set the fixed image mask */
    void setMovingMask(const cv::Mat& img);
    /** @brief The nearest-neighbor matching ratio */
    void setMatchRatio(float r);
    /** @copydoc setMatchRatio(float) */
    [[nodiscard]] auto matchRatio() const -> float;
    /**
     * @brief Maximum image dimension
     *
     * Images with any dimension larger than this size will be
     * downscaled before landmark detection.
     */
    void setMaxImageDim(int s);
    /** @copydoc setMaxImageDim(int) */
    [[nodiscard]] auto maxImageDim() const -> int;

    /** @brief Compute key point matches between the fixed and moving images
     *
     * Returns a list of matches, sorted by strength of match and filtered for
     * outliers.
     */
    auto compute() -> std::vector<LandmarkPair>;

    /**
     * @brief Get the computed matches
     *
     * Returns a list of matches, sorted by strength of match and filtered for
     * outliers.
     */
    auto getLandmarkPairs() -> std::vector<LandmarkPair>;

    /** @brief Get the detected landmarks for the fixed image */
    [[nodiscard]] auto getFixedLandmarks() const -> LandmarkContainer;
    /** @brief Get the detected landmarks for the moving image */
    [[nodiscard]] auto getMovingLandmarks() const -> LandmarkContainer;

private:
    /** Fixed image */
    cv::Mat fixedImg_;
    /** Fixed image mask */
    cv::Mat fixedMask_;
    /** Moving image */
    cv::Mat movingImg_;
    /** Moving image mask */
    cv::Mat movingMask_;
    /** Matched pairs */
    std::vector<LandmarkPair> output_;
    /** Nearest-neighbor matching ratio */
    float nnMatchRatio_{0.7F};
    /** Maximum image size for feature detection */
    int maxImageDim_{4096};
};
}  // namespace rt
