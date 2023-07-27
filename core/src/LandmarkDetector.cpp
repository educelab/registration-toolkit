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
void LandmarkDetector::setMatchRatio(float r) { nnMatchRatio_ = r; }
void LandmarkDetector::setMaxImageDim(int s) { maxImageDim_ = s; }

namespace
{
auto NeedsResize(const cv::Mat& img, int dimLimit, float& scale) -> bool
{
    auto maxDim = std::max(img.rows, img.cols);
    auto res = maxDim > dimLimit;
    if (res) {
        scale = static_cast<float>(dimLimit) / static_cast<float>(maxDim);
    }
    return res;
}
}  // namespace

// Compute the matches
auto LandmarkDetector::compute() -> std::vector<rt::LandmarkPair>
{
    // Make sure we have the images
    if (fixedImg_.empty() or movingImg_.empty()) {
        throw std::runtime_error("Missing image(s)");
    }

    // Clear the output vector
    output_.clear();

    // Resize inputs
    cv::Mat fixedImg = QuantizeImage(fixedImg_, CV_8U);
    cv::Mat movingImg = QuantizeImage(movingImg_, CV_8U);
    cv::Mat fixedMask = QuantizeImage(ColorConvertImage(fixedMask_), CV_8U);
    cv::Mat movingMask = QuantizeImage(ColorConvertImage(movingMask_), CV_8U);
    float fs{1.};
    float ms{1.};
    if (::NeedsResize(fixedImg, maxImageDim_, fs)) {
        cv::resize(fixedImg, fixedImg, cv::Size(), fs, fs, cv::INTER_AREA);
        std::cerr << "Resized fixed image: ";
        std::cerr << fixedImg.cols << "x" << fixedImg.rows << std::endl;
        if (not fixedMask.empty()) {
            cv::resize(
                fixedMask, fixedMask, cv::Size(), fs, fs, cv::INTER_AREA);
        }
    }
    if (::NeedsResize(movingImg, maxImageDim_, ms)) {
        cv::resize(movingImg, movingImg, cv::Size(), ms, ms, cv::INTER_AREA);
        std::cerr << "Resized moving image: ";
        std::cerr << movingImg.cols << "x" << movingImg.rows << std::endl;
        if (not movingMask.empty()) {
            cv::resize(
                movingMask, movingMask, cv::Size(), ms, ms, cv::INTER_AREA);
        }
    }

    // Detect key points and compute their descriptors
    auto featureDetector = cv::SIFT::create();
    std::vector<cv::KeyPoint> fixedKeys;
    std::vector<cv::KeyPoint> movingKeys;
    cv::Mat fixedDesc;
    cv::Mat movingDesc;
    featureDetector->detectAndCompute(
        fixedImg, fixedMask, fixedKeys, fixedDesc);
    featureDetector->detectAndCompute(
        movingImg, movingMask, movingKeys, movingDesc);

    // Match keypoints
    auto matcher =
        cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector<std::vector<cv::DMatch>> matches;
    matcher->knnMatch(fixedDesc, movingDesc, matches, 2);

    // Filter matches
    std::vector<cv::DMatch> goodMatches;
    for (const auto& m : matches) {
        if (m[0].distance < nnMatchRatio_ * m[1].distance) {
            goodMatches.push_back(m[0]);
        }
    }

    // Use RANSAC to filter matches further
    // TODO: It's a shame that we can't keep and use this homography...
    std::vector<cv::Point2f> fixed;
    std::vector<cv::Point2f> moving;
    cv::Mat mask;
    for (const auto& m : goodMatches) {
        fixed.push_back(fixedKeys[m.queryIdx].pt);
        moving.emplace_back(movingKeys[m.trainIdx].pt);
    }
    cv::findHomography(moving, fixed, cv::RANSAC, 3., mask);

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
            auto fixPt = fixedKeys[m.queryIdx].pt * fs;
            auto movPt = movingKeys[m.trainIdx].pt * ms;
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
auto LandmarkDetector::getLandmarkPairs() -> std::vector<rt::LandmarkPair>
{
    return output_;
}

auto LandmarkDetector::getFixedLandmarks() const -> LandmarkContainer
{
    LandmarkContainer res;
    Landmark l;
    for (const auto& p : output_) {
        l[0] = p.first.x;
        l[1] = p.first.y;
        res.push_back(l);
    }
    return res;
}

auto LandmarkDetector::getMovingLandmarks() const -> LandmarkContainer
{
    LandmarkContainer res;
    Landmark l;
    for (const auto& p : output_) {
        l[0] = p.second.x;
        l[1] = p.second.y;
        res.push_back(l);
    }
    return res;
}

auto LandmarkDetector::matchRatio() const -> float { return nnMatchRatio_; }

auto LandmarkDetector::maxImageDim() const -> int { return maxImageDim_; }
