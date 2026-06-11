#include "rt/graph/LandmarkRegistration.hpp"

#include "rt/io/LandmarkIO.hpp"
#include "rt/Logging.hpp"

namespace rtg = rt::graph;
namespace fs = rt::filesystem;

// Enum conversions
namespace rt
{
// clang-format off
using EnhancementNode = rtg::LandmarkDetectorNode::EnhancementMode;
NLOHMANN_JSON_SERIALIZE_ENUM(EnhancementNode, {
    {EnhancementNode::None, "none"},
    {EnhancementNode::CLAHE, "clahe"},
    {EnhancementNode::OriginalWithCLAHE, "original-clahe"}
})
// clang-format on
}  // namespace rt

rtg::LandmarkDetectorNode::LandmarkDetectorNode() : Node{true}
{
    registerInputPort("fixedImage", fixedImage);
    registerInputPort("fixedMask", fixedMask);
    registerInputPort("movingImage", movingImage);
    registerInputPort("movingMask", movingMask);
    registerInputPort("matchRatio", matchRatio);
    registerInputPort("maxImageDim", maxImageDim);
    registerInputPort("enhancementMode", enhancementMode);
    registerOutputPort("fixedLandmarks", fixedLandmarks);
    registerOutputPort("movingLandmarks", movingLandmarks);
    compute = [this]() {
        rt::logger()->info("Detecting landmarks");
        detector_.setFixedImage(fixedImg_);
        detector_.setFixedMask(fixedMask_);
        detector_.setMovingImage(movingImg_);
        detector_.setMovingMask(movingMask_);
        detector_.compute();
        fixedLdm_ = detector_.getFixedLandmarks();
        movingLdm_ = detector_.getMovingLandmarks();
    };
}

smgl::Metadata rtg::LandmarkDetectorNode::serialize_(
    const bool useCache, const fs::path& cacheDir)
{
    smgl::Metadata m{
        {"matchRatio", detector_.matchRatio()},
        {"maxImageDim", detector_.maxImageDim()},
        {"enhancementMode", detector_.enhancementMode()}};
    if (useCache) {
        LandmarkWriter writer;
        writer.setPath(cacheDir / "landmarks.ldm");
        writer.setFixedLandmarks(fixedLdm_);
        writer.setMovingLandmarks(movingLdm_);
        writer.write();
        m["landmarks"] = "landmarks.ldm";
    }

    return m;
}

void rtg::LandmarkDetectorNode::deserialize_(
    const smgl::Metadata& meta, const fs::path& cacheDir)
{
    detector_.setMatchRatio(meta["matchRatio"].get<float>());
    detector_.setMaxImageDim(meta["maxImageDim"].get<int>());
    auto mode = EnhancementMode::OriginalWithCLAHE;
    if (meta.contains("enhancementMode")) {
        mode = meta["enhancementMode"].get<EnhancementMode>();
    }
    detector_.setEnhancementMode(mode);
    if (meta.contains("landmarks")) {
        const auto file = meta["landmarks"].get<std::string>();
        LandmarkReader reader;
        reader.setLandmarksPath(cacheDir / file);
        reader.read();
        fixedLdm_ = reader.getFixedLandmarks();
        movingLdm_ = reader.getMovingLandmarks();
    }
}

rtg::AffineLandmarkRegistrationNode::AffineLandmarkRegistrationNode()
    : Node{true}
    , fixedLandmarks{&fixed_}
    , movingLandmarks{&moving_}
    , reportMetrics{&reg_, &AffineLandmarkRegistration::setReportMetrics}
    , transform{&tfm_}
{
    registerInputPort("fixedLandmarks", fixedLandmarks);
    registerInputPort("movingLandmarks", movingLandmarks);
    registerInputPort("reportMetrics", reportMetrics);
    registerOutputPort("transform", transform);

    compute = [this]() {
        rt::logger()->info("Running affine registration");
        reg_.setFixedLandmarks(fixed_);
        reg_.setMovingLandmarks(moving_);
        tfm_ = reg_.compute();
    };
}

auto rtg::AffineLandmarkRegistrationNode::serialize_(
    const bool useCache, const fs::path& cacheDir) -> smgl::Metadata
{
    smgl::Metadata m;
    m["reportMetrics"] = reg_.getReportMetrics();
    if (useCache and tfm_) {
        WriteTransform(cacheDir / "affine.tfm", tfm_);
        m["transform"] = "affine.tfm";
    }

    return m;
}

void rtg::AffineLandmarkRegistrationNode::deserialize_(
    const smgl::Metadata& meta, const fs::path& cacheDir)
{
    reg_.setReportMetrics(meta["reportMetrics"].get<bool>());
    if (meta.contains("transform")) {
        const auto file = meta["transform"].get<std::string>();
        tfm_ = ReadTransform(cacheDir / file);
    }
}

rtg::BSplineLandmarkWarpingNode::BSplineLandmarkWarpingNode() : Node{true}
{
    registerInputPort("fixedLandmarks", fixedLandmarks);
    registerInputPort("fixedImage", fixedImage);
    registerInputPort("movingLandmarks", movingLandmarks);
    registerOutputPort("transform", transform);

    compute = [this]() {
        rt::logger()->info("Running B-spline landmark registration");
        reg_.setFixedLandmarks(fixed_);
        reg_.setFixedImage(fixedImg_);
        reg_.setMovingLandmarks(moving_);
        tfm_ = reg_.compute();
    };
}

smgl::Metadata rtg::BSplineLandmarkWarpingNode::serialize_(
    const bool useCache, const fs::path& cacheDir)
{
    smgl::Metadata m;
    if (useCache and tfm_) {
        WriteTransform(cacheDir / "bspline.tfm", tfm_);
        m["transform"] = "bspline.tfm";
    }

    return m;
}

void rtg::BSplineLandmarkWarpingNode::deserialize_(
    const smgl::Metadata& meta, const fs::path& cacheDir)
{
    if (meta.contains("transform")) {
        const auto file = meta["transform"].get<std::string>();
        tfm_ = ReadTransform(cacheDir / file);
    }
}