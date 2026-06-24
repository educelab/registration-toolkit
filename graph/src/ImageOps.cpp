#include "rt/graph/ImageOps.hpp"

#include <vector>

#include "rt/io/ImageIO.hpp"
#include "rt/util/ImageConversion.hpp"

using namespace rt;

namespace fs = rt::filesystem;
namespace rtg = rt::graph;

// Enum conversions
namespace rt
{
// clang-format off
using PositionMapMode = rtg::PositionMapTransformNode::Mode;
NLOHMANN_JSON_SERIALIZE_ENUM(PositionMapMode, {
    {PositionMapMode::Raw, "raw"},
    {PositionMapMode::Shifted, "shifted"},
    {PositionMapMode::Normalized, "normalized"},
})
// clang-format on
}  // namespace rt

// Adjust each axis of a CV_32FC3 position map independently over its finite
// (intersected) pixels. NaN pixels (no surface intersection) are preserved;
// NaN != NaN, so (ch == ch) masks in only the finite pixels.
static auto TransformPositionMap(
    const cv::Mat& map, rtg::PositionMapTransformNode::Mode mode) -> cv::Mat
{
    using Mode = rtg::PositionMapTransformNode::Mode;
    if (mode == Mode::Raw or map.empty()) {
        return map;
    }

    std::vector<cv::Mat> channels;
    cv::split(map, channels);
    for (auto& ch : channels) {
        const cv::Mat finite = (ch == ch);
        double mn{0.0}, mx{0.0};
        cv::minMaxLoc(ch, &mn, &mx, nullptr, nullptr, finite);
        if (mode == Mode::Shifted) {
            // -> [0, extent), preserving scale. Scalar ops leave NaN untouched.
            ch = ch - mn;
        } else {
            // Normalized -> [0, 1]
            const double range = mx - mn;
            if (range > 0.0) {
                ch = (ch - mn) / range;
            } else {
                ch.setTo(0.0, finite);
            }
        }
    }
    cv::Mat out;
    cv::merge(channels, out);
    return out;
}

rtg::ColorConvertNode::ColorConvertNode()
    : Node{true}, imageIn{&input_}, channels{&cns_}, imageOut{&output_}
{
    registerInputPort("imageIn", imageIn);
    registerInputPort("channels", channels);
    registerOutputPort("imageOut", imageOut);

    compute = [this]() {
        output_ = ColorConvertImage(input_, static_cast<int>(cns_));
    };
}

auto rtg::ColorConvertNode::serialize_(
    const bool useCache, const fs::path& cacheDir) -> smgl::Metadata
{
    smgl::Metadata m{{"channels", cns_}};
    if (useCache and not output_.empty()) {
        WriteImage(cacheDir / "converted.tif", output_);
        m["converted"] = "converted.tif";
    }
    return m;
}

void rtg::ColorConvertNode::deserialize_(
    const smgl::Metadata& meta, const fs::path& cacheDir)
{
    cns_ = meta["channels"].get<std::size_t>();
    if (meta.contains("converted")) {
        const auto file = meta["converted"].get<std::string>();
        output_ = ReadImage(cacheDir / file);
    }
}

rtg::PositionMapTransformNode::PositionMapTransformNode()
    : Node{true}, imageIn{&input_}, mode{&mode_}, imageOut{&output_}
{
    registerInputPort("imageIn", imageIn);
    registerInputPort("mode", mode);
    registerOutputPort("imageOut", imageOut);

    compute = [this]() { output_ = TransformPositionMap(input_, mode_); };
}

auto rtg::PositionMapTransformNode::serialize_(
    const bool useCache, const fs::path& cacheDir) -> smgl::Metadata
{
    smgl::Metadata m{{"mode", mode_}};
    if (useCache and not output_.empty()) {
        WriteImage(cacheDir / "position_map.tif", output_);
        m["positionMap"] = "position_map.tif";
    }
    return m;
}

void rtg::PositionMapTransformNode::deserialize_(
    const smgl::Metadata& meta, const fs::path& cacheDir)
{
    mode_ = meta["mode"].get<Mode>();
    if (meta.contains("positionMap")) {
        const auto file = meta["positionMap"].get<std::string>();
        output_ = ReadImage(cacheDir / file);
    }
}