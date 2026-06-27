#include "rt/graph/ImageIO.hpp"

#include <educelab/core/utils/Iteration.hpp>
#include <educelab/core/utils/String.hpp>

#include "rt/io/ImageIO.hpp"

using namespace educelab;

namespace rtg = rt::graph;
namespace fs = std::filesystem;

rtg::ReadImageNode::ReadImageNode()
{
    registerInputPort("path", path);
    registerOutputPort("image", image);
    compute = [this]() { img_ = ReadImage(path_); };
}

smgl::Metadata rtg::ReadImageNode::serialize_(bool, const fs::path&)
{
    return {{"path", path_.string()}};
}

void rtg::ReadImageNode::deserialize_(
    const smgl::Metadata& meta, const fs::path&)
{
    path_ = meta["path"].get<std::string>();
    img_ = rt::ReadImage(path_);
}

rtg::WriteImageNode::WriteImageNode()
{
    registerInputPort("path", path);
    registerInputPort("image", image);
    compute = [this]() { WriteImage(path_, img_); };
}

smgl::Metadata rtg::WriteImageNode::serialize_(bool, const fs::path&)
{
    return {{"path", path_.string()}};
}

void rtg::WriteImageNode::deserialize_(
    const smgl::Metadata& meta, const fs::path&)
{
    path_ = meta["path"].get<std::string>();
}

rtg::WriteImageSeriesNode::WriteImageSeriesNode()
{
    registerInputPort("path", path);
    registerInputPort("images", images);
    compute = [this]() {
        // components
        fs::path parent;
        std::string prefix;
        std::string suffix;
        fs::path ext;

        // If directory, default to dir/###.tif
        if (fs::is_directory(path_)) {
            parent = path_;
            ext = ".tif";
        }

        // If path, decompose to replace {} with a number
        else {
            parent = path_.parent_path();
            ext = path_.extension();

            // Split into a prefix and suffix
            auto stem = path_.stem().string();
            std::tie(prefix, std::ignore, suffix) = partition(stem, "{}");
        }

        // Setup padding
        auto pad = static_cast<int>(std::to_string(imgs_.size()).size());

        // Write images
        for (const auto [i, image] : enumerate(imgs_)) {
            const auto name = prefix + to_padded_string(i, pad) + suffix;
            auto filepath = (parent / name).replace_extension(ext);
            WriteImage(filepath, image);
        }
    };
}

smgl::Metadata rtg::WriteImageSeriesNode::serialize_(bool, const fs::path&)
{
    return {{"path", path_.string()}};
}

void rtg::WriteImageSeriesNode::deserialize_(
    const smgl::Metadata& meta, const fs::path&)
{
    path_ = meta["path"].get<std::string>();
}