#include <cmath>
#include <limits>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include "rt/graph/ImageOps.hpp"

using Node = rt::graph::PositionMapTransformNode;
using Mode = Node::Mode;

namespace
{
constexpr auto kNaN = std::numeric_limits<float>::quiet_NaN();

// A 2x2 CV_32FC3 position map. Per-axis finite ranges are:
//   ch0: [10, 14] (range 4), ch1: [20, 24] (range 4), ch2: [30, 34] (range 4).
// One pixel is all-NaN to stand in for a no-intersection sample.
auto MakePositionMap() -> cv::Mat
{
    cv::Mat m(2, 2, CV_32FC3);
    m.at<cv::Vec3f>(0, 0) = {10.F, 20.F, 30.F};
    m.at<cv::Vec3f>(0, 1) = {14.F, 24.F, 34.F};
    m.at<cv::Vec3f>(1, 0) = {kNaN, kNaN, kNaN};
    m.at<cv::Vec3f>(1, 1) = {12.F, 22.F, 32.F};
    return m;
}

// Run the input map through a PositionMapTransformNode in the given mode.
auto Transform(const cv::Mat& in, Mode mode) -> cv::Mat
{
    Node node;
    node.imageIn.post(in);
    node.mode.post(mode);
    node.update();
    return node.imageOut();
}

void ExpectVec3fNear(const cv::Vec3f& got, const cv::Vec3f& want)
{
    for (int c = 0; c < 3; ++c) {
        EXPECT_NEAR(got[c], want[c], 1e-5F) << "channel " << c;
    }
}
}  // namespace

TEST(PositionMapTransform, RawPassesThrough)
{
    const auto in = MakePositionMap();
    const auto out = Transform(in, Mode::Raw);

    ExpectVec3fNear(out.at<cv::Vec3f>(0, 0), {10.F, 20.F, 30.F});
    ExpectVec3fNear(out.at<cv::Vec3f>(0, 1), {14.F, 24.F, 34.F});
    ExpectVec3fNear(out.at<cv::Vec3f>(1, 1), {12.F, 22.F, 32.F});
}

TEST(PositionMapTransform, ShiftedSubtractsPerAxisMinimum)
{
    const auto out = Transform(MakePositionMap(), Mode::Shifted);

    ExpectVec3fNear(out.at<cv::Vec3f>(0, 0), {0.F, 0.F, 0.F});
    ExpectVec3fNear(out.at<cv::Vec3f>(0, 1), {4.F, 4.F, 4.F});
    ExpectVec3fNear(out.at<cv::Vec3f>(1, 1), {2.F, 2.F, 2.F});
}

TEST(PositionMapTransform, NormalizedRescalesPerAxisToUnitRange)
{
    const auto out = Transform(MakePositionMap(), Mode::Normalized);

    ExpectVec3fNear(out.at<cv::Vec3f>(0, 0), {0.F, 0.F, 0.F});
    ExpectVec3fNear(out.at<cv::Vec3f>(0, 1), {1.F, 1.F, 1.F});
    ExpectVec3fNear(out.at<cv::Vec3f>(1, 1), {0.5F, 0.5F, 0.5F});
}

TEST(PositionMapTransform, PreservesNaNPixels)
{
    for (const auto mode : {Mode::Shifted, Mode::Normalized}) {
        const auto out = Transform(MakePositionMap(), mode);
        const auto px = out.at<cv::Vec3f>(1, 0);
        EXPECT_TRUE(std::isnan(px[0]));
        EXPECT_TRUE(std::isnan(px[1]));
        EXPECT_TRUE(std::isnan(px[2]));
    }
}

TEST(PositionMapTransform, AllNaNAxisIsLeftUnchanged)
{
    // Every pixel NaN: no finite samples to rescale. Guarded so minMaxLoc is
    // never called with an empty mask; output stays all-NaN without crashing.
    cv::Mat m(2, 2, CV_32FC3, cv::Scalar(kNaN, kNaN, kNaN));
    const auto out = Transform(m, Mode::Normalized);
    for (int r = 0; r < out.rows; ++r) {
        for (int c = 0; c < out.cols; ++c) {
            const auto px = out.at<cv::Vec3f>(r, c);
            EXPECT_TRUE(std::isnan(px[0]));
            EXPECT_TRUE(std::isnan(px[1]));
            EXPECT_TRUE(std::isnan(px[2]));
        }
    }
}

TEST(PositionMapTransform, EmptyInputIsHandled)
{
    const auto out = Transform(cv::Mat{}, Mode::Shifted);
    EXPECT_TRUE(out.empty());
}
