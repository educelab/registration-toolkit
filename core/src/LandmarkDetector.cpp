#include "rt/LandmarkDetector.hpp"

#include <algorithm>
#include <exception>

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#include "rt/util/ImageConversion.hpp"

using namespace rt;

void LandmarkDetector::setFixedImage(const cv::Mat& img) { fixedImg_ = img; }
void LandmarkDetector::setFixedMask(const cv::Mat& img) { fixedMask_ = img; }
void LandmarkDetector::setMovingImage(const cv::Mat& img) { movingImg_ = img; }
void LandmarkDetector::setMovingMask(const cv::Mat& img) { movingMask_ = img; }
void LandmarkDetector::setMatchRatio(const float r) { nnMatchRatio_ = r; }
void LandmarkDetector::setMaxImageDim(const int s) { maxImageDim_ = s; }

namespace
{
auto NeedsResize(const cv::Mat& img, const int dimLimit, float& scale) -> bool
{
    const auto maxDim = std::max(img.rows, img.cols);
    const auto res = maxDim > dimLimit;
    if (res) {
        scale = static_cast<float>(dimLimit) / static_cast<float>(maxDim);
    }
    return res;
}

auto RatioTest(
    const std::vector<std::vector<cv::DMatch>>& matches,
    const float ratio) -> std::vector<cv::DMatch>
{
    std::vector<cv::DMatch> goodMatches;
    for (const auto& m : matches) {
        if (m[0].distance < ratio * m[1].distance) {
            goodMatches.push_back(m[0]);
        }
    }
    return goodMatches;
}

auto DetectAndMatch(
    const cv::Mat& fixedImg,
    const cv::Mat& movingImg,
    const cv::Mat& fixedMask,
    const cv::Mat& movingMask,
    const float ratio)
{
    const auto featureDetector = cv::SIFT::create();
    const auto matcher =
        cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector<cv::KeyPoint> fixedKeys;
    std::vector<cv::KeyPoint> movingKeys;
    cv::Mat fixedDesc;
    cv::Mat movingDesc;
    std::vector<std::vector<cv::DMatch>> matches;

    // detect features
    featureDetector->detectAndCompute(
        fixedImg, fixedMask, fixedKeys, fixedDesc);
    featureDetector->detectAndCompute(
        movingImg, movingMask, movingKeys, movingDesc);

    // Match keypoints
    matcher->knnMatch(fixedDesc, movingDesc, matches, 2);

    // Apply ratio test
    const auto filtered = RatioTest(matches, ratio);

    // collect the actual points
    std::vector<cv::Point2f> fixed;
    std::vector<cv::Point2f> moving;
    for (const auto& m : filtered) {
        fixed.push_back(fixedKeys[m.queryIdx].pt);
        moving.emplace_back(movingKeys[m.trainIdx].pt);
    }

    // result struct
    struct result {
        std::vector<cv::DMatch> matches;
        std::vector<cv::Point2f> fixed;
        std::vector<cv::Point2f> moving;
    };
    return result{filtered, fixed, moving};
}
}  // namespace

// Compute the matches
auto LandmarkDetector::compute() -> std::vector<LandmarkPair>
{
    // Make sure we have the images
    if (fixedImg_.empty() or movingImg_.empty()) {
        throw std::runtime_error("Missing image(s)");
    }

    // Clear the output vector
    output_.clear();

    // Resize inputs
    cv::Mat fixedImg = QuantizeImage(ColorConvertImage(fixedImg_), CV_8U);
    cv::Mat movingImg = QuantizeImage(ColorConvertImage(movingImg_), CV_8U);
    cv::Mat fixedMask = QuantizeImage(ColorConvertImage(fixedMask_), CV_8U);
    cv::Mat movingMask = QuantizeImage(ColorConvertImage(movingMask_), CV_8U);
    float fs{1.};
    float ms{1.};
    if (NeedsResize(fixedImg, maxImageDim_, fs)) {
        cv::resize(fixedImg, fixedImg, cv::Size(), fs, fs, cv::INTER_AREA);
        std::cerr << "Resized fixed image: ";
        std::cerr << fixedImg.cols << "x" << fixedImg.rows << std::endl;
        if (not fixedMask.empty()) {
            cv::resize(
                fixedMask, fixedMask, cv::Size(), fs, fs, cv::INTER_AREA);
        }
    }
    if (NeedsResize(movingImg, maxImageDim_, ms)) {
        cv::resize(movingImg, movingImg, cv::Size(), ms, ms, cv::INTER_AREA);
        std::cerr << "Resized moving image: ";
        std::cerr << movingImg.cols << "x" << movingImg.rows << std::endl;
        if (not movingMask.empty()) {
            cv::resize(
                movingMask, movingMask, cv::Size(), ms, ms, cv::INTER_AREA);
        }
    }

    // Set up detection and matching
    std::vector<cv::DMatch> goodMatches;
    std::vector<cv::Point2f> fixed;
    std::vector<cv::Point2f> moving;

    // Detect + match in original images
    if (enhanceMode_ == None or enhanceMode_ == OriginalWithCLAHE) {
        auto [matches, fixedPts, movingPts] = DetectAndMatch(
            fixedImg, movingImg, fixedMask, movingMask, nnMatchRatio_);
        goodMatches.reserve(goodMatches.size() + matches.size());
        goodMatches.insert(goodMatches.end(), matches.begin(), matches.end());

        fixed.reserve(fixed.size() + fixedPts.size());
        fixed.insert(fixed.end(), fixedPts.begin(), fixedPts.end());

        moving.reserve(moving.size() + movingPts.size());
        moving.insert(moving.end(), movingPts.begin(), movingPts.end());
    }

    // Detect + match in equalized images
    if (enhanceMode_ == CLAHE or enhanceMode_ == OriginalWithCLAHE) {
        auto s = static_cast<int>(8.F * ms / fs);
        auto clahe = cv::createCLAHE(40., {s, s});
        clahe->apply(fixedImg, fixedImg);

        s = static_cast<int>(8.F * fs / ms);
        clahe->setTilesGridSize({s, s});
        clahe->apply(movingImg, movingImg);
        auto [matches, fixedPts, movingPts] = DetectAndMatch(
            fixedImg, movingImg, fixedMask, movingMask, nnMatchRatio_);
        goodMatches.reserve(goodMatches.size() + matches.size());
        goodMatches.insert(goodMatches.end(), matches.begin(), matches.end());

        fixed.reserve(fixed.size() + fixedPts.size());
        fixed.insert(fixed.end(), fixedPts.begin(), fixedPts.end());

        moving.reserve(moving.size() + movingPts.size());
        moving.insert(moving.end(), movingPts.begin(), movingPts.end());
    }

    // Use RANSAC to filter matches further
    cv::Mat mask;
    cv::estimateAffinePartial2D(moving, fixed, mask, cv::RANSAC, 1.);

    // Convert good matches to landmark pairs
    // query = fixed, train = moving
    fs = 1.F / fs;
    ms = 1.F / ms;
    for (int idx = 0; idx < static_cast<int>(goodMatches.size()); idx++) {
        // Get match
        const auto& m = goodMatches[idx];

        // Filter by mask
        if (mask.at<std::uint8_t>(idx, 0) == 0) {
            continue;
        }

        // From fixed -> moving
        if (m.imgIdx == 0) {
            auto fixPt = fixed[idx] * fs;
            auto movPt = moving[idx] * ms;
            output_.emplace_back(fixPt, movPt);
        }

        // From moving -> fixed
        else if (m.imgIdx != 0) {
            std::cerr << "Warning: Unexpected image match index: ";
            std::cerr << m.imgIdx << std::endl;
        }
    }

    return output_;
}

// Return previously computed matches
auto LandmarkDetector::getLandmarkPairs() -> std::vector<LandmarkPair>
{
    return output_;
}

auto LandmarkDetector::getFixedLandmarks() const -> LandmarkContainer
{
    LandmarkContainer res;
    Landmark l;
    for (const auto& [fixed, _] : output_) {
        l[0] = fixed.x;
        l[1] = fixed.y;
        res.emplace_back(l);
    }
    return res;
}

auto LandmarkDetector::getMovingLandmarks() const -> LandmarkContainer
{
    LandmarkContainer res;
    Landmark l;
    for (const auto& [_, moving] : output_) {
        l[0] = moving.x;
        l[1] = moving.y;
        res.push_back(l);
    }
    return res;
}

auto LandmarkDetector::matchRatio() const -> float { return nnMatchRatio_; }

auto LandmarkDetector::maxImageDim() const -> int { return maxImageDim_; }

void LandmarkDetector::setEnhancementMode(const EnhancementMode m)
{
    enhanceMode_ = m;
}

auto LandmarkDetector::enhancementMode() const -> EnhancementMode
{
    return enhanceMode_;
}
