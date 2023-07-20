#include "rt/LandmarkDetector.hpp"

#include <algorithm>
#include <exception>

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

using namespace rt;

void LandmarkDetector::setFixedImage(const cv::Mat& img) { fixedImg_ = img; }
void LandmarkDetector::setFixedMask(const cv::Mat& img) { fixedMask_ = img; }
void LandmarkDetector::setMovingImage(const cv::Mat& img) { movingImg_ = img; }
void LandmarkDetector::setMovingMask(const cv::Mat& img) { movingMask_ = img; }
void LandmarkDetector::setMatchRatio(float r) { nnMatchRatio_ = r; }
void LandmarkDetector::setMaxImageDim(int s) { maxImageDim_ = s; }

namespace
{
auto NeedsResize(const cv::Mat& img, int dimLimit, float& scale)
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
    cv::Mat fixedImg = fixedImg_;
    cv::Mat movingImg = movingImg_;
    cv::Mat fixedMask = fixedMask_;
    cv::Mat movingMask = movingMask_;
    float fs{1.};
    float ms{1.};
    if (::NeedsResize(fixedImg, maxImageDim_, fs)) {
        cv::resize(fixedImg_, fixedImg, cv::Size(), fs, fs, cv::INTER_AREA);
        std::cerr << "Resized fixed image: ";
        std::cerr << fixedImg.cols << "x" << fixedImg.rows << std::endl;
        if (not fixedMask_.empty()) {
            cv::resize(
                fixedMask_, fixedMask, cv::Size(), fs, fs, cv::INTER_AREA);
        }
    }
    if (::NeedsResize(movingImg, maxImageDim_, ms)) {
        cv::resize(movingImg_, movingImg, cv::Size(), ms, ms, cv::INTER_AREA);
        std::cerr << "Resized moving image: ";
        std::cerr << movingImg.cols << "x" << movingImg.rows << std::endl;
        if (not movingMask_.empty()) {
            cv::resize(
                movingMask_, movingMask, cv::Size(), ms, ms, cv::INTER_AREA);
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

    // Convert good matches to landmark pairs
    // query = fixed, train = moving
    fs = 1.F / fs;
    ms = 1.F / ms;
    for (const auto& m : goodMatches) {
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
