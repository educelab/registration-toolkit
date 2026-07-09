#include "rt/graph/MeshIO.hpp"

#include <utility>

#include "rt/Logging.hpp"
#include "rt/io/ImageIO.hpp"
#include "rt/io/MeshIO.hpp"

using namespace rt;

namespace fs = std::filesystem;
namespace rtg = rt::graph;

rtg::MeshReadNode::MeshReadNode()
{
    registerInputPort("path", path);
    registerOutputPort("mesh", mesh);
    registerOutputPort("image", image);
    registerOutputPort("imagePath", imagePath);
    registerOutputPort("images", images);
    registerOutputPort("uvMap", uvMap);
    compute = [this]() {
        rt::logger()->info("Reading mesh: {}", path_.string());
        auto result = ReadMesh(path_);
        mesh_ = result.mesh;
        imgs_ = result.textures;
        uv_ = result.uvMap;
        // The single image/imagePath ports expose the first texture (chart 0)
        // for downstream single-image consumers (e.g. registration).
        img_ = imgs_.empty() ? cv::Mat() : imgs_.front();
        imgPath_ = result.texturePaths.empty() ? std::filesystem::path()
                                               : result.texturePaths.front();
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
    , imageSource{[this](std::filesystem::path p) {
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
                WriteMesh(path_, *mesh_, uv_, img_);
                break;
            case TextureInput::Source:
                WriteMesh(path_, *mesh_, uv_, imgSource_);
                break;
            case TextureInput::None:
                WriteMesh(path_, *mesh_, uv_);
                break;
        }
    };
}

smgl::Metadata rtg::MeshWriteNode::serialize_(
    const bool useCache, const fs::path& cacheDir)
{
    smgl::Metadata m{{"path", path_.string()}};
    switch (lastTexture_) {
        case TextureInput::Image:
            m["texture"] = "image";
            if (useCache and not img_.empty()) {
                WriteImage(cacheDir / "texture.tif", img_);
                m["image"] = "texture.tif";
            }
            break;
        case TextureInput::Source:
            m["texture"] = "source";
            m["imageSource"] = imgSource_.string();
            break;
        case TextureInput::None:
            m["texture"] = "none";
            break;
    }
    return m;
}

void rtg::MeshWriteNode::deserialize_(
    const smgl::Metadata& meta, const fs::path& cacheDir)
{
    path_ = meta["path"].get<std::string>();
    // Graphs written before texture tracking have no "texture" key; treat them
    // as having written an untextured mesh.
    const auto texture =
        meta.contains("texture") ? meta["texture"].get<std::string>() : "none";
    if (texture == "image") {
        lastTexture_ = TextureInput::Image;
        if (meta.contains("image")) {
            img_ = ReadImage(cacheDir / meta["image"].get<std::string>());
        }
    } else if (texture == "source") {
        lastTexture_ = TextureInput::Source;
        imgSource_ = meta["imageSource"].get<std::string>();
    } else {
        lastTexture_ = TextureInput::None;
    }
}
