#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <set>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include "rt/ReorderUnorganizedTexture.hpp"
#include "rt/types/Mesh.hpp"
#include "rt/types/UVMap.hpp"

using ProjectionParams = rt::ReorderUnorganizedTexture::ProjectionParams;

// A minimal, valid pinhole camera: identity pose, unit-ish intrinsics.
static auto ValidCamera() -> ProjectionParams
{
    ProjectionParams p;
    p.fx = 800.0;
    p.fy = 800.0;
    p.cx = 320.0;
    p.cy = 240.0;
    p.width = 640;
    p.height = 480;
    p.extrinsics = cv::Matx44d::eye();
    return p;
}

TEST(ValidateProjectionParams, AcceptsValidCamera)
{
    EXPECT_FALSE(rt::ValidateProjectionParams(ValidCamera()).has_value());
}

TEST(ValidateProjectionParams, AcceptsRotatedPose)
{
    // 90 deg about Z is still a proper rotation
    auto p = ValidCamera();
    p.extrinsics = cv::Matx44d::eye();
    p.extrinsics(0, 0) = 0.0;
    p.extrinsics(0, 1) = -1.0;
    p.extrinsics(1, 0) = 1.0;
    p.extrinsics(1, 1) = 0.0;
    EXPECT_FALSE(rt::ValidateProjectionParams(p).has_value());
}

TEST(ValidateProjectionParams, RejectsNonPositiveSize)
{
    auto p = ValidCamera();
    p.width = 0;
    EXPECT_TRUE(rt::ValidateProjectionParams(p).has_value());

    p = ValidCamera();
    p.height = -1;
    EXPECT_TRUE(rt::ValidateProjectionParams(p).has_value());
}

TEST(ValidateProjectionParams, RejectsNonPositiveFocalLength)
{
    auto p = ValidCamera();
    p.fx = 0.0;
    EXPECT_TRUE(rt::ValidateProjectionParams(p).has_value());

    p = ValidCamera();
    p.fy = -10.0;
    EXPECT_TRUE(rt::ValidateProjectionParams(p).has_value());
}

TEST(ValidateProjectionParams, RejectsNonFiniteIntrinsics)
{
    const auto inf = std::numeric_limits<double>::infinity();
    const auto nan = std::numeric_limits<double>::quiet_NaN();

    auto p = ValidCamera();
    p.fx = inf;
    EXPECT_TRUE(rt::ValidateProjectionParams(p).has_value());

    p = ValidCamera();
    p.cx = nan;
    EXPECT_TRUE(rt::ValidateProjectionParams(p).has_value());
}

TEST(ValidateProjectionParams, RejectsNonOrthonormalRotation)
{
    // Scale the rotation block: orthogonal but not orthonormal
    auto p = ValidCamera();
    p.extrinsics(0, 0) = 2.0;
    EXPECT_TRUE(rt::ValidateProjectionParams(p).has_value());
}

TEST(ValidateProjectionParams, RejectsReflection)
{
    // Flip one axis: orthonormal but left-handed (det = -1)
    auto p = ValidCamera();
    p.extrinsics(0, 0) = -1.0;
    EXPECT_TRUE(rt::ValidateProjectionParams(p).has_value());
}

TEST(ValidateProjectionParams, RejectsNonFiniteDistortion)
{
    const auto nan = std::numeric_limits<double>::quiet_NaN();
    auto p = ValidCamera();
    p.k2 = nan;
    EXPECT_TRUE(rt::ValidateProjectionParams(p).has_value());
}

TEST(RadialDistortion, IdentityWhenNoCoefficients)
{
    auto p = ValidCamera();  // k1 = k2 = k3 = 0
    const cv::Vec2d pt{0.3, -0.2};
    const auto d = rt::DistortNormalized(p, pt);
    EXPECT_NEAR(d[0], pt[0], 1e-12);
    EXPECT_NEAR(d[1], pt[1], 1e-12);

    const auto u = rt::UndistortNormalized(p, pt);
    EXPECT_NEAR(u[0], pt[0], 1e-12);
    EXPECT_NEAR(u[1], pt[1], 1e-12);
}

TEST(RadialDistortion, DistortPushesPointsOutward)
{
    // Positive k1 (barrel) moves an off-center point away from the center
    auto p = ValidCamera();
    p.k1 = 0.1;
    const cv::Vec2d pt{0.4, 0.3};
    const auto d = rt::DistortNormalized(p, pt);
    EXPECT_GT(cv::norm(d), cv::norm(pt));
}

TEST(RadialDistortion, UndistortInvertsDistort)
{
    auto p = ValidCamera();
    p.k1 = -0.12;
    p.k2 = 0.05;
    p.k3 = -0.01;
    // Sweep a range of normalized coordinates and confirm round-trip recovery
    for (double y = -0.4; y <= 0.4; y += 0.2) {
        for (double x = -0.4; x <= 0.4; x += 0.2) {
            const cv::Vec2d ideal{x, y};
            const auto round =
                rt::UndistortNormalized(p, rt::DistortNormalized(p, ideal));
            EXPECT_NEAR(round[0], ideal[0], 1e-6) << "x=" << x << " y=" << y;
            EXPECT_NEAR(round[1], ideal[1], 1e-6) << "x=" << x << " y=" << y;
        }
    }
}

namespace
{
// A flat 2-quad sheet (x: 0..2, y: 0..1) in the z=0 plane. Faces 0,1 (left
// quad) are UV chart 0; faces 2,3 (right quad) are chart 1. UVs are all
// (0.5, 0.5): for a solid-color source image the exact UV is irrelevant, only
// the chart index matters. Sampled via an explicit top-down camera (see
// TopDownCamera) so the test does not depend on the OBB estimation used by the
// orthographic path, which produces an invalid output size for a small
// degenerate mesh like this one (see issue #20).
struct TwoChartMesh {
    rt::Mesh::Pointer mesh;
    rt::UVMap uv;
};

auto MakeTwoChartMesh() -> TwoChartMesh
{
    auto mesh = rt::Mesh::New();
    mesh->insert_vertex(0.0, 0.0, 0.0);  // 0
    mesh->insert_vertex(1.0, 0.0, 0.0);  // 1
    mesh->insert_vertex(1.0, 1.0, 0.0);  // 2
    mesh->insert_vertex(0.0, 1.0, 0.0);  // 3
    mesh->insert_vertex(2.0, 0.0, 0.0);  // 4
    mesh->insert_vertex(2.0, 1.0, 0.0);  // 5

    mesh->insert_face(0, 1, 2);  // face 0, left quad
    mesh->insert_face(0, 2, 3);  // face 1, left quad
    mesh->insert_face(1, 4, 5);  // face 2, right quad
    mesh->insert_face(1, 5, 2);  // face 3, right quad

    const std::array<std::size_t, 4> faceChart{0, 0, 1, 1};
    rt::UVMap uv;
    for (std::size_t fi = 0; fi < faceChart.size(); ++fi) {
        for (std::size_t corner = 0; corner < 3; ++corner) {
            const auto idx = uv.insert(0.5F, 0.5F);
            uv.at(idx).chart = faceChart[fi];
            uv.map(fi, corner, idx);
        }
    }
    return {mesh, uv};
}

// A pinhole camera at world (1, 0.5, 1) looking straight down onto the z=0
// sheet (OpenCV convention: +Z_cam forward into the scene, +Y_cam down). The
// sheet fills the 40x20 output. Bypasses the OBB-based orthographic path.
auto TopDownCamera() -> rt::ReorderUnorganizedTexture::ProjectionParams
{
    rt::ReorderUnorganizedTexture::ProjectionParams p;
    p.fx = 20.0;
    p.fy = 20.0;
    p.cx = 20.0;
    p.cy = 10.0;
    p.width = 40;
    p.height = 20;
    // world->camera: R rows are the camera axes in world; t = -R * C.
    // R = [[1,0,0],[0,-1,0],[0,0,-1]], C = (1, 0.5, 1) -> t = (-1, 0.5, 1).
    p.extrinsics = cv::Matx44d::eye();
    p.extrinsics(1, 1) = -1.0;
    p.extrinsics(2, 2) = -1.0;
    p.extrinsics(0, 3) = -1.0;
    p.extrinsics(1, 3) = 0.5;
    p.extrinsics(2, 3) = 1.0;
    return p;
}

// Distinct non-black (non-background) colors present in a BGR image.
auto DistinctColors(const cv::Mat& img) -> std::set<std::array<int, 3>>
{
    std::set<std::array<int, 3>> colors;
    for (int r = 0; r < img.rows; ++r) {
        for (int c = 0; c < img.cols; ++c) {
            const auto px = img.at<cv::Vec3b>(r, c);
            if (px[0] == 0 and px[1] == 0 and px[2] == 0) {
                continue;
            }
            colors.insert({px[0], px[1], px[2]});
        }
    }
    return colors;
}

auto HasBackground(const cv::Mat& img) -> bool
{
    for (int r = 0; r < img.rows; ++r) {
        for (int c = 0; c < img.cols; ++c) {
            const auto px = img.at<cv::Vec3b>(r, c);
            if (px[0] == 0 and px[1] == 0 and px[2] == 0) {
                return true;
            }
        }
    }
    return false;
}
}  // namespace

TEST(ReorderMultiTexture, SamplesEachChartFromItsImage)
{
    const auto data = MakeTwoChartMesh();
    const cv::Mat redImg(16, 16, CV_8UC3, cv::Scalar(0, 0, 255));   // BGR red
    const cv::Mat blueImg(16, 16, CV_8UC3, cv::Scalar(255, 0, 0));  // BGR blue

    rt::ReorderUnorganizedTexture reorder;
    reorder.setMesh(data.mesh);
    reorder.setUVMap(data.uv);
    reorder.setTextureMats({redImg, blueImg});
    reorder.setProjectionMode(
        rt::ReorderUnorganizedTexture::ProjectionMode::Camera);
    reorder.setProjectionParams(TopDownCamera());
    const auto out = reorder.compute();

    ASSERT_FALSE(out.empty());
    const auto colors = DistinctColors(out);
    // Every colored pixel is exactly RED or BLUE, and both charts contributed.
    EXPECT_EQ(colors.size(), 2U);
    EXPECT_EQ(colors.count({0, 0, 255}), 1U);
    EXPECT_EQ(colors.count({255, 0, 0}), 1U);
}

TEST(ReorderMultiTexture, SkipsFacesWhoseChartHasNoImage)
{
    const auto data = MakeTwoChartMesh();
    const cv::Mat redImg(16, 16, CV_8UC3, cv::Scalar(0, 0, 255));  // BGR red

    rt::ReorderUnorganizedTexture reorder;
    reorder.setMesh(data.mesh);
    reorder.setUVMap(data.uv);
    reorder.setTextureMats({redImg});  // no image supplied for chart 1
    reorder.setProjectionMode(
        rt::ReorderUnorganizedTexture::ProjectionMode::Camera);
    reorder.setProjectionParams(TopDownCamera());
    const auto out = reorder.compute();

    ASSERT_FALSE(out.empty());
    const auto colors = DistinctColors(out);
    // Only chart 0 (RED) is colored; chart-1 faces are left as background.
    EXPECT_EQ(colors.size(), 1U);
    EXPECT_EQ(colors.count({0, 0, 255}), 1U);
    EXPECT_TRUE(HasBackground(out));
}
