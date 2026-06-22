#include "rt/graph/MeshOps.hpp"

#include <algorithm>
#include <array>

#include "rt/io/ImageIO.hpp"
#include "rt/io/UVMapIO.hpp"
#include "rt/Logging.hpp"

using namespace rt;

namespace fs = rt::filesystem;
namespace rtg = rt::graph;

// Enum conversions
namespace rt
{
// clang-format off
using SamplingOrigin = rtg::ReorderTextureNode::SamplingOrigin;
NLOHMANN_JSON_SERIALIZE_ENUM(SamplingOrigin, {
    {SamplingOrigin::TopLeft, "top-left"},
    {SamplingOrigin::TopRight, "top-right"},
    {SamplingOrigin::BottomLeft, "bottom-left"},
    {SamplingOrigin::BottomRight, "bottom-right"}
})

using SamplingMode = rtg::ReorderTextureNode::SamplingMode;
NLOHMANN_JSON_SERIALIZE_ENUM(SamplingMode, {
    {SamplingMode::Rate, "rate"},
    {SamplingMode::OutputWidth, "width"},
    {SamplingMode::OutputHeight, "height"},
    {SamplingMode::AutoUV, "auto"},
})

using ProjectionMode = rtg::ReorderTextureNode::ProjectionMode;
NLOHMANN_JSON_SERIALIZE_ENUM(ProjectionMode, {
    {ProjectionMode::Orthographic, "orthographic"},
    {ProjectionMode::Camera, "camera"},
})
// clang-format on

// Pinhole camera intrinsics + extrinsics
using ProjectionParams = rtg::ReorderTextureNode::ProjectionParams;
template <typename Json>
void to_json(Json& j, const ProjectionParams& p)
{
    j = Json{{"fx", p.fx},     {"fy", p.fy},         {"cx", p.cx},
             {"cy", p.cy},     {"width", p.width},   {"height", p.height}};
    std::array<double, 16> ext{};
    std::copy(p.extrinsics.val, p.extrinsics.val + 16, ext.begin());
    j["extrinsics"] = ext;
}

template <typename Json>
void from_json(const Json& j, ProjectionParams& p)
{
    j.at("fx").get_to(p.fx);
    j.at("fy").get_to(p.fy);
    j.at("cx").get_to(p.cx);
    j.at("cy").get_to(p.cy);
    j.at("width").get_to(p.width);
    j.at("height").get_to(p.height);
    const auto ext = j.at("extrinsics").template get<std::array<double, 16>>();
    std::copy(ext.begin(), ext.end(), p.extrinsics.val);
}
}  // namespace rt

rtg::ReorderTextureNode::ReorderTextureNode()
    : Node{true}
    , meshIn{&reorder_, &ReorderUnorganizedTexture::setMesh}
    , imageIn{&reorder_, &ReorderUnorganizedTexture::setTextureMat}
    , uvMapIn{&reorder_, &ReorderUnorganizedTexture::setUVMap}
    , samplingOrigin{&reorder_, &ReorderUnorganizedTexture::setSamplingOrigin}
    , samplingMode{&reorder_, &ReorderUnorganizedTexture::setSamplingMode}
    , sampleRate{&reorder_, &ReorderUnorganizedTexture::setSampleRate}
    , sampleDim{&reorder_, &ReorderUnorganizedTexture::setSampleDim}
    , useFirstIntersection{&reorder_, &ReorderUnorganizedTexture::setUseFirstIntersection}
    , projectionMode{&reorder_, &ReorderUnorganizedTexture::setProjectionMode}
    , projectionParams{[this](const ProjectionParams& p) {
        reorder_.setProjectionParams(p);
        haveProjParams_ = true;
    }}
    , imageOut{&outImg_}
    , uvMapOut{&outUV_}
    , depthMapOut{&reorder_, &ReorderUnorganizedTexture::getDepthMap}
    , positionMapOut{&reorder_, &ReorderUnorganizedTexture::getPositionMap}
{
    registerInputPort("mesh", meshIn);
    registerInputPort("imageIn", imageIn);
    registerInputPort("uvMapIn", uvMapIn);
    registerInputPort("samplingOrigin", samplingOrigin);
    registerInputPort("samplingMode", samplingMode);
    registerInputPort("sampleRate", sampleRate);
    registerInputPort("sampleDim", sampleDim);
    registerInputPort("useFirstIntersection", useFirstIntersection);
    registerInputPort("projectionMode", projectionMode);
    registerInputPort("projectionParams", projectionParams);
    registerOutputPort("imageOut", imageOut);
    registerOutputPort("uvMapOut", uvMapOut);
    registerOutputPort("depthMapOut", depthMapOut);
    registerOutputPort("positionMapOut", positionMapOut);

    compute = [this]() {
        rt::logger()->info("Reordering texture image");
        outImg_ = reorder_.compute();
        outUV_ = reorder_.getUVMap();
    };
}

auto rtg::ReorderTextureNode::serialize_(
    const bool useCache, const fs::path& cacheDir) -> smgl::Metadata
{
    smgl::Metadata m{
        {"samplingOrigin", reorder_.samplingOrigin()},
        {"samplingMode", reorder_.samplingMode()},
        {"sampleRate", reorder_.sampleRate()},
        {"sampleDim", reorder_.sampleDim()},
        {"useFirstIntersection", reorder_.useFirstIntersection()},
        {"projectionMode", reorder_.projectionMode()},
    };
    if (haveProjParams_) {
        m["projectionParams"] = reorder_.projectionParams();
    }
    if (useCache) {
        if (not outUV_.empty()) {
            WriteUVMap(cacheDir / "reordered_uv.uvm", outUV_);
            m["uvMap"] = "reordered_uv.uvm";
        }
        if (not outImg_.empty()) {
            WriteImage(cacheDir / "reordered_img.tif", outImg_);
            m["image"] = "reordered_img.tif";
            WriteImage(cacheDir / "depth_map.tif", reorder_.getDepthMap());
            m["depth-map"] = "depth_map.tif";
            if (not reorder_.getPositionMap().empty()) {
                WriteImage(
                    cacheDir / "position_map.tif", reorder_.getPositionMap());
                m["position-map"] = "position_map.tif";
            }
        }
    }
    return m;
}

void rtg::ReorderTextureNode::deserialize_(
    const smgl::Metadata& meta, const fs::path& cacheDir)
{
    reorder_.setSamplingOrigin(meta["samplingOrigin"].get<SamplingOrigin>());
    reorder_.setSamplingMode(meta["samplingMode"].get<SamplingMode>());
    reorder_.setSampleRate(meta["sampleRate"].get<double>());
    reorder_.setSampleDim(meta["sampleDim"].get<std::size_t>());
    reorder_.setUseFirstIntersection(meta["useFirstIntersection"].get<bool>());
    if (meta.contains("projectionMode")) {
        reorder_.setProjectionMode(meta["projectionMode"].get<ProjectionMode>());
    }
    if (meta.contains("projectionParams")) {
        reorder_.setProjectionParams(
            meta["projectionParams"].get<ProjectionParams>());
        haveProjParams_ = true;
    }
    if (meta.contains("uvMap")) {
        const auto file = meta["uvMap"].get<std::string>();
        outUV_ = ReadUVMap(cacheDir / file);
    }
    if (meta.contains("image")) {
        const auto file = meta["image"].get<std::string>();
        outImg_ = ReadImage(cacheDir / file);
    }
}
