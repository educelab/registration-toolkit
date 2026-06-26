#include "rt/graph/MeshIO.hpp"

#include "rt/io/MeshIO.hpp"
#include "rt/Logging.hpp"

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
{
    registerInputPort("path", path);
    registerInputPort("mesh", mesh);
    registerInputPort("image", image);
    registerInputPort("imageSource", imageSource);
    registerInputPort("uvMap", uvMap);
    compute = [this]() {
        rt::logger()->info("Writing mesh: {}", path_.string());
        if (mesh_) {
            io::WriteMesh(path_, *mesh_, uv_, img_, imgSource_);
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
