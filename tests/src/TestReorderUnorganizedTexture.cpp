#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include "rt/ReorderUnorganizedTexture.hpp"

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
