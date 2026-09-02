#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
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

namespace
{
// A canonically oriented sheet: mesh right along +X, mesh up along +Y, surface
// normal along +Z. Spans x in [-2, 2], y in [-1.5, 1.5], bowed slightly in z
// so the bounding box has a genuine (if thin) third dimension and a
// well-defined handedness.
//
// Each of the four world-space quadrants is its own UV chart, so with one
// solid-color image per chart the output image's quadrant colors report the
// orientation of the sampling frame directly. Four distinct colors detect a
// mirror as well as a rotation, which matters: a mirrored texture cannot be
// undone by rotating the image, and a rotation-only check passes while the
// papyrus still reads backwards.
//
//   chart 0 = RED   : +x, +y      chart 1 = GREEN : -x, +y
//   chart 2 = BLUE  : -x, -y      chart 3 = WHITE : +x, -y
struct QuadrantSheet {
    rt::Mesh::Pointer mesh;
    rt::UVMap uv;
};

auto MakeQuadrantSheet(double xHalf = 2.0, double yHalf = 1.5)
    -> QuadrantSheet
{
    constexpr int kNx = 41;
    constexpr int kNy = 31;
    const auto kXMin = -xHalf;
    const auto kXMax = xHalf;
    const auto kYMin = -yHalf;
    const auto kYMax = yHalf;

    auto mesh = rt::Mesh::New();
    for (int j = 0; j < kNy; ++j) {
        for (int i = 0; i < kNx; ++i) {
            const auto x = kXMin + (kXMax - kXMin) * i / (kNx - 1);
            const auto y = kYMin + (kYMax - kYMin) * j / (kNy - 1);
            const auto z = 0.2 * (1.0 - (x / kXMax) * (x / kXMax));
            mesh->insert_vertex(x, y, z);
        }
    }

    // Chart index for a face, from the sign of its cell center
    const auto chartOf = [](double cx, double cy) -> std::size_t {
        if (cy > 0.0) {
            return (cx > 0.0) ? 0U : 1U;
        }
        return (cx > 0.0) ? 3U : 2U;
    };

    rt::UVMap uv;
    std::size_t faceIdx{0};
    for (int j = 0; j + 1 < kNy; ++j) {
        for (int i = 0; i + 1 < kNx; ++i) {
            const auto v00 = j * kNx + i;
            const auto v10 = j * kNx + i + 1;
            const auto v01 = (j + 1) * kNx + i;
            const auto v11 = (j + 1) * kNx + i + 1;

            const auto cx = kXMin + (kXMax - kXMin) * (i + 0.5) / (kNx - 1);
            const auto cy = kYMin + (kYMax - kYMin) * (j + 0.5) / (kNy - 1);
            const auto chart = chartOf(cx, cy);

            // Wind CCW seen from +Z so the surface normal points along +Z
            mesh->insert_face(v00, v10, v11);
            mesh->insert_face(v00, v11, v01);
            for (int f = 0; f < 2; ++f) {
                for (std::size_t corner = 0; corner < 3; ++corner) {
                    const auto idx = uv.insert(0.5F, 0.5F);
                    uv.at(idx).chart = chart;
                    uv.map(faceIdx, corner, idx);
                }
                ++faceIdx;
            }
        }
    }
    return {mesh, uv};
}

// One solid-color image per quadrant chart, in chart order
auto QuadrantImages() -> std::vector<cv::Mat>
{
    return {
        cv::Mat(8, 8, CV_8UC3, cv::Scalar(0, 0, 255)),      // 0 RED
        cv::Mat(8, 8, CV_8UC3, cv::Scalar(0, 255, 0)),      // 1 GREEN
        cv::Mat(8, 8, CV_8UC3, cv::Scalar(255, 0, 0)),      // 2 BLUE
        cv::Mat(8, 8, CV_8UC3, cv::Scalar(255, 255, 255)),  // 3 WHITE
    };
}

// The most common non-background color in the quadrant of @p img selected by
// (@p right, @p bottom). Sampling a whole quadrant rather than one pixel keeps
// the assertion clear of the chart seams that cross the image center.
auto QuadrantColor(const cv::Mat& img, bool right, bool bottom)
    -> std::array<int, 3>
{
    const auto c0 = right ? img.cols / 2 : 0;
    const auto r0 = bottom ? img.rows / 2 : 0;
    std::map<std::array<int, 3>, int> counts;
    for (int r = r0 + img.rows / 8; r < r0 + 3 * img.rows / 8; ++r) {
        for (int c = c0 + img.cols / 8; c < c0 + 3 * img.cols / 8; ++c) {
            const auto px = img.at<cv::Vec3b>(r, c);
            if (px[0] == 0 and px[1] == 0 and px[2] == 0) {
                continue;
            }
            counts[{px[0], px[1], px[2]}]++;
        }
    }
    std::array<int, 3> best{0, 0, 0};
    int bestCount{0};
    for (const auto& [color, n] : counts) {
        if (n > bestCount) {
            bestCount = n;
            best = color;
        }
    }
    return best;
}

constexpr std::array<int, 3> kRed{0, 0, 255};
constexpr std::array<int, 3> kGreen{0, 255, 0};
constexpr std::array<int, 3> kBlue{255, 0, 0};
constexpr std::array<int, 3> kWhite{255, 255, 255};

auto ReorderQuadrantSheet(
    rt::ReorderUnorganizedTexture::OrientationMode mode,
    double xHalf = 2.0,
    double yHalf = 1.5) -> cv::Mat
{
    const auto data = MakeQuadrantSheet(xHalf, yHalf);
    rt::ReorderUnorganizedTexture reorder;
    reorder.setMesh(data.mesh);
    reorder.setUVMap(data.uv);
    reorder.setTextureMats(QuadrantImages());
    reorder.setSamplingMode(
        rt::ReorderUnorganizedTexture::SamplingMode::Rate);
    reorder.setSampleRate(0.05);
    reorder.setOrientationMode(mode);
    return reorder.compute();
}
}  // namespace

// The orientation the pipeline actually depends on: for a canonically oriented
// input, image +u runs along world +X and image +v along world -Y, so each
// world quadrant lands in the corresponding image quadrant. This is the
// regression that catches a mirrored deliverable.
TEST(ReorderOrientation, CanonicalMatchesWorldAxes)
{
    using OrientationMode = rt::ReorderUnorganizedTexture::OrientationMode;
    const auto out = ReorderQuadrantSheet(OrientationMode::Canonical);
    ASSERT_FALSE(out.empty());

    // image top-left = (-x, +y) = GREEN, top-right = (+x, +y) = RED,
    // bottom-left = (-x, -y) = BLUE, bottom-right = (+x, -y) = WHITE
    EXPECT_EQ(QuadrantColor(out, false, false), kGreen) << "top-left";
    EXPECT_EQ(QuadrantColor(out, true, false), kRed) << "top-right";
    EXPECT_EQ(QuadrantColor(out, false, true), kBlue) << "bottom-left";
    EXPECT_EQ(QuadrantColor(out, true, true), kWhite) << "bottom-right";
}

// With the axis assignment unchanged (a clearly non-square sheet), redirecting
// the axes cannot change the output size.
//
// This does NOT generalise: when canonicalization reassigns which OBB axis
// becomes the image width, the extent that width is measured along changes
// with it, and under SamplingMode::OutputWidth/OutputHeight that changes the
// pixel scale. See CanonicalMayResizeWhenAxesPermute.
TEST(ReorderOrientation, CanonicalPreservesOutputSizeWhenAxesDoNotPermute)
{
    using OrientationMode = rt::ReorderUnorganizedTexture::OrientationMode;
    const auto obb = ReorderQuadrantSheet(OrientationMode::OBB);
    const auto canonical = ReorderQuadrantSheet(OrientationMode::Canonical);
    ASSERT_FALSE(obb.empty());
    ASSERT_FALSE(canonical.empty());
    EXPECT_EQ(obb.size(), canonical.size());
}

TEST(ReorderOrientation, DefaultsToOBB)
{
    rt::ReorderUnorganizedTexture reorder;
    EXPECT_EQ(
        reorder.orientationMode(),
        rt::ReorderUnorganizedTexture::OrientationMode::OBB);
}

// vtkOBBTree orders its axes by extent, so on a near-square fragment the
// in-plane axes can arrive on the opposite world axes -- a 90-degree error
// that correcting signs alone cannot undo. Canonicalization assigns axes by
// world-axis agreement, so the result holds even when the longer side is y.
TEST(ReorderOrientation, CanonicalHandlesNearSquareSheet)
{
    using OrientationMode = rt::ReorderUnorganizedTexture::OrientationMode;
    const auto out =
        ReorderQuadrantSheet(OrientationMode::Canonical, 1.5, 1.52);
    ASSERT_FALSE(out.empty());
    EXPECT_EQ(QuadrantColor(out, false, false), kGreen) << "top-left";
    EXPECT_EQ(QuadrantColor(out, true, false), kRed) << "top-right";
    EXPECT_EQ(QuadrantColor(out, false, true), kBlue) << "bottom-left";
    EXPECT_EQ(QuadrantColor(out, true, true), kWhite) << "bottom-right";
}

// Canonicalization reassigns axes by world agreement, so on a near-square sheet
// the image width can end up measured along the other extent. Pinned here
// because it is a real consequence of the assignment rule, not a bug: the
// sampling *plane* is unchanged, but the derived pixel scale is not.
TEST(ReorderOrientation, CanonicalMayResizeWhenAxesPermute)
{
    using OrientationMode = rt::ReorderUnorganizedTexture::OrientationMode;
    using SamplingMode = rt::ReorderUnorganizedTexture::SamplingMode;

    const auto run = [](OrientationMode mode) {
        const auto data = MakeQuadrantSheet(1.50, 1.52);  // 3.00 x 3.04
        rt::ReorderUnorganizedTexture reorder;
        reorder.setMesh(data.mesh);
        reorder.setUVMap(data.uv);
        reorder.setTextureMats(QuadrantImages());
        reorder.setSamplingMode(SamplingMode::OutputWidth);
        reorder.setSampleDim(800);
        reorder.setOrientationMode(mode);
        return reorder.compute();
    };

    const auto obb = run(OrientationMode::OBB);
    const auto canonical = run(OrientationMode::Canonical);
    ASSERT_FALSE(obb.empty());
    ASSERT_FALSE(canonical.empty());
    EXPECT_EQ(obb.cols, canonical.cols) << "width is pinned by sampleDim";
    EXPECT_NE(obb.rows, canonical.rows)
        << "a permuted assignment should change the height";
}

namespace
{
// Two stacked canonically oriented sheets: chart 0 (RED) is the world
// +Z-facing surface at z = +0.5, chart 1 (BLUE) sits below it at z = -0.5.
//
// A single-layer sheet cannot test which face gets sampled -- it reads
// identically from either side -- so a realignment that reflects instead of
// rotating still passes every quadrant-colour and UV check while sampling the
// wrong surface and inverting the depth map. Two layers make the face choice
// observable.
auto MakeTwoLayerSheet(double tiltDeg = 0.0) -> QuadrantSheet
{
    constexpr int kN = 21;
    constexpr double kPi = 3.14159265358979323846;
    const auto t = tiltDeg * kPi / 180.0;
    auto mesh = rt::Mesh::New();
    for (const double z : {0.5, -0.5}) {
        for (int j = 0; j < kN; ++j) {
            for (int i = 0; i < kN; ++i) {
                const auto x = -2.0 + 4.0 * i / (kN - 1);
                const auto y = -1.5 + 3.0 * j / (kN - 1);
                mesh->insert_vertex(
                    x * std::cos(t) - y * std::sin(t),
                    x * std::sin(t) + y * std::cos(t), z);
            }
        }
    }

    rt::UVMap uv;
    std::size_t faceIdx{0};
    for (int layer = 0; layer < 2; ++layer) {
        const auto base = layer * kN * kN;
        for (int j = 0; j + 1 < kN; ++j) {
            for (int i = 0; i + 1 < kN; ++i) {
                const auto a = base + j * kN + i;
                mesh->insert_face(a, a + 1, a + kN + 1);
                mesh->insert_face(a, a + kN + 1, a + kN);
                for (int f = 0; f < 2; ++f) {
                    for (std::size_t corner = 0; corner < 3; ++corner) {
                        const auto idx = uv.insert(0.5F, 0.5F);
                        uv.at(idx).chart = static_cast<std::size_t>(layer);
                        uv.map(faceIdx, corner, idx);
                    }
                    ++faceIdx;
                }
            }
        }
    }
    return {mesh, uv};
}
}  // namespace

// Canonical mode must sample the surface facing world +Z, not the one behind
// it. This is the assertion that catches a realignment which reflects (det=-1)
// rather than rotates: the image orientation still looks right, but the far
// surface is what got sampled.
TEST(ReorderOrientation, CanonicalSamplesTheZPlusFacingSurface)
{
    using OrientationMode = rt::ReorderUnorganizedTexture::OrientationMode;

    // Swept across in-plane tilt because the reflection this guards against is
    // discontinuous: the anti-parallel branch it came from only triggers within
    // ~2.56 degrees of axis alignment. A single untilted case would sit on one
    // side of that threshold and miss the other. Under OrientationMode::OBB the
    // sampled face flips between tilts, which is the arbitrariness this mode
    // exists to remove; under Canonical it must be +Z at every one.
    for (const double tilt : {0.0, 1.0, 2.0, 2.5, 2.6, 3.0, 5.0, 10.0}) {
        const auto data = MakeTwoLayerSheet(tilt);

        rt::ReorderUnorganizedTexture reorder;
        reorder.setMesh(data.mesh);
        reorder.setUVMap(data.uv);
        reorder.setTextureMats(
            {cv::Mat(8, 8, CV_8UC3, cv::Scalar(0, 0, 255)),      // chart 0 RED
             cv::Mat(8, 8, CV_8UC3, cv::Scalar(255, 0, 0))});    // chart 1 BLUE
        reorder.setSamplingMode(
            rt::ReorderUnorganizedTexture::SamplingMode::Rate);
        reorder.setSampleRate(0.05);
        reorder.setOrientationMode(OrientationMode::Canonical);
        const auto out = reorder.compute();

        ASSERT_FALSE(out.empty()) << "tilt " << tilt;
        for (const bool right : {false, true}) {
            for (const bool bottom : {false, true}) {
                EXPECT_EQ(QuadrantColor(out, right, bottom), kRed)
                    << "tilt " << tilt << " quadrant right=" << right
                    << " bottom=" << bottom << " sampled the -Z surface";
            }
        }
    }
}

// The same parity, read off the depth map: the bowed sheet crests at x = 0, so
// viewed from world +Z the crest must be nearer the sampling plane than the
// edges. A reflected realignment inverts this, which would silently corrupt
// the --depth-map and --position-map deliverables.
TEST(ReorderOrientation, CanonicalDepthMapIsMeasuredFromZPlus)
{
    using OrientationMode = rt::ReorderUnorganizedTexture::OrientationMode;
    const auto data = MakeQuadrantSheet();

    rt::ReorderUnorganizedTexture reorder;
    reorder.setMesh(data.mesh);
    reorder.setUVMap(data.uv);
    reorder.setTextureMats(QuadrantImages());
    reorder.setSamplingMode(
        rt::ReorderUnorganizedTexture::SamplingMode::Rate);
    reorder.setSampleRate(0.05);
    reorder.setOrientationMode(OrientationMode::Canonical);
    reorder.compute();

    const auto depth = reorder.getDepthMap();
    ASSERT_FALSE(depth.empty());

    // Median finite depth over a small window centred on (colFrac, 0.5)
    const auto sample = [&depth](double colFrac) {
        std::vector<float> vals;
        const auto c0 = static_cast<int>(colFrac * depth.cols) - 4;
        const auto r0 = depth.rows / 2 - 4;
        for (int r = r0; r < r0 + 8; ++r) {
            for (int c = c0; c < c0 + 8; ++c) {
                const auto d = depth.at<float>(r, c);
                if (std::isfinite(d)) {
                    vals.push_back(d);
                }
            }
        }
        std::sort(vals.begin(), vals.end());
        return vals.empty() ? std::numeric_limits<float>::quiet_NaN()
                            : vals[vals.size() / 2];
    };

    const auto crest = sample(0.50);   // x ~ 0, the high point of the bow
    const auto flank = sample(0.12);   // x ~ -1.5, closer to the sheet edge
    ASSERT_TRUE(std::isfinite(crest));
    ASSERT_TRUE(std::isfinite(flank));
    EXPECT_LT(crest, flank)
        << "crest depth " << crest << " should be less than flank " << flank
        << " when sampling from world +Z";
}

// The auto-derived camera inherits the same three arbitrary choices from the
// raw OBB that the orthographic frame does: which side of the surface it sits
// on (a mirror), which way is up (a 180-degree turn), and which in-plane axis
// becomes the image width (a 90-degree turn). Canonical resolves all three, so
// the camera image lands in the same frame the orthographic one does.
TEST(ReorderOrientation, CanonicalAutoCameraMatchesWorldAxes)
{
    using OrientationMode = rt::ReorderUnorganizedTexture::OrientationMode;
    using ProjectionMode = rt::ReorderUnorganizedTexture::ProjectionMode;

    const auto data = MakeQuadrantSheet();
    rt::ReorderUnorganizedTexture reorder;
    reorder.setMesh(data.mesh);
    reorder.setUVMap(data.uv);
    reorder.setTextureMats(QuadrantImages());
    reorder.setProjectionMode(ProjectionMode::Camera);
    reorder.setOrientationMode(OrientationMode::Canonical);
    const auto out = reorder.compute();

    ASSERT_FALSE(out.empty());
    EXPECT_EQ(QuadrantColor(out, false, false), kGreen) << "top-left";
    EXPECT_EQ(QuadrantColor(out, true, false), kRed) << "top-right";
    EXPECT_EQ(QuadrantColor(out, false, true), kBlue) << "bottom-left";
    EXPECT_EQ(QuadrantColor(out, true, true), kWhite) << "bottom-right";
}

// The camera must also be placed on the +Z side, not behind the surface. Swept
// across tilt for the same reason the orthographic face test is.
TEST(ReorderOrientation, CanonicalAutoCameraViewsTheZPlusSide)
{
    using OrientationMode = rt::ReorderUnorganizedTexture::OrientationMode;
    using ProjectionMode = rt::ReorderUnorganizedTexture::ProjectionMode;

    // Fewer tilts than the orthographic sweep: each camera render is a full
    // 2048-square ray cast. 0 and 2.5 straddle the anti-parallel threshold.
    for (const double tilt : {0.0, 2.5, 10.0}) {
        const auto data = MakeTwoLayerSheet(tilt);
        rt::ReorderUnorganizedTexture reorder;
        reorder.setMesh(data.mesh);
        reorder.setUVMap(data.uv);
        reorder.setTextureMats(
            {cv::Mat(8, 8, CV_8UC3, cv::Scalar(0, 0, 255)),      // chart 0 RED
             cv::Mat(8, 8, CV_8UC3, cv::Scalar(255, 0, 0))});    // chart 1 BLUE
        reorder.setProjectionMode(ProjectionMode::Camera);
        reorder.setOrientationMode(OrientationMode::Canonical);
        const auto out = reorder.compute();

        ASSERT_FALSE(out.empty()) << "tilt " << tilt;
        EXPECT_EQ(QuadrantColor(out, false, false), kRed)
            << "tilt " << tilt << ": camera viewed the -Z side";
    }
}

// An explicit camera is used verbatim; the orientation mode must not touch it.
TEST(ReorderOrientation, ExplicitCameraIgnoresOrientationMode)
{
    using OrientationMode = rt::ReorderUnorganizedTexture::OrientationMode;
    using ProjectionMode = rt::ReorderUnorganizedTexture::ProjectionMode;

    const auto run = [](OrientationMode mode) {
        const auto data = MakeTwoChartMesh();
        rt::ReorderUnorganizedTexture reorder;
        reorder.setMesh(data.mesh);
        reorder.setUVMap(data.uv);
        reorder.setTextureMats(
            {cv::Mat(16, 16, CV_8UC3, cv::Scalar(0, 0, 255)),
             cv::Mat(16, 16, CV_8UC3, cv::Scalar(255, 0, 0))});
        reorder.setProjectionMode(ProjectionMode::Camera);
        reorder.setProjectionParams(TopDownCamera());
        reorder.setOrientationMode(mode);
        return reorder.compute();
    };

    const auto obb = run(OrientationMode::OBB);
    const auto canonical = run(OrientationMode::Canonical);
    ASSERT_FALSE(obb.empty());
    ASSERT_EQ(obb.size(), canonical.size());
    EXPECT_EQ(cv::norm(obb, canonical, cv::NORM_INF), 0.0)
        << "an explicit camera must be used verbatim";
}
