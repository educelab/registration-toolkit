#include "rt/graph/MeshIO.hpp"

#include <utility>

#include "rt/Logging.hpp"
#include "rt/io/MeshIO.hpp"

using namespace rt;

namespace fs = rt::filesystem;
namespace rtg = rt::graph;

rtg::MeshReadNode::MeshReadNode()
{
    registerInputPort("path", path);
    registerOutputPort("mesh", mesh);
    registerOutputPort("image", image);
    registerOutputPort("imagePath", imagePath);
    registerOutputPort("uvMap", uvMap);
    compute = [this]() {
        rt::logger()->info("Reading mesh: {}", path_.string());
        auto result = io::ReadMesh(path_);
        mesh_ = result.mesh;
        img_ = result.texture;
        imgPath_ = result.texturePath;
        uv_ = result.uvMap;
    };
}

smgl::Metadata rtg::MeshReadNode::serialize_(bool, const fs::path&)
{
    return {{"path", path_.string()}};
}

void rtg::MeshReadNode::deserialize_(
    const smgl::Metadata& meta, const fs::path&)
{
    path_ = meta["path"].get<std::string>();
    compute();
}

rtg::MeshWriteNode::MeshWriteNode()
    // image and imageSource share one logical "texture" input: each setter
    // records itself as the most recent, so the last assignment wins (smgl
    // ports cannot be un-set, so a fixed precedence would pin the first one).
    : image{[this](cv::Mat m) {
        img_ = std::move(m);
        lastTexture_ = TextureInput::Image;
    }}
    , imageSource{[this](filesystem::path p) {
        imgSource_ = std::move(p);
        lastTexture_ = TextureInput::Source;
    }}
{
    registerInputPort("path", path);
    registerInputPort("mesh", mesh);
    registerInputPort("image", image);
    registerInputPort("imageSource", imageSource);
    registerInputPort("uvMap", uvMap);
    compute = [this]() {
        rt::logger()->info("Writing mesh: {}", path_.string());
        if (not mesh_) {
            rt::logger()->warn("No mesh provided; skipping mesh write");
            return;
        }

        switch (lastTexture_) {
            case TextureInput::Image:
                io::WriteMesh(path_, *mesh_, uv_, img_);
                break;
            case TextureInput::Source:
                io::WriteMesh(path_, *mesh_, uv_, imgSource_);
                break;
            case TextureInput::None:
                io::WriteMesh(path_, *mesh_, uv_);
                break;
        }
    };
}

smgl::Metadata rtg::MeshWriteNode::serialize_(bool, const fs::path&)
{
    return {{"path", path_.string()}};
}

void rtg::MeshWriteNode::deserialize_(
    const smgl::Metadata& meta, const fs::path&)
{
    path_ = meta["path"].get<std::string>();
}
