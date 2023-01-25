#include "rt/graph/MeshIO.hpp"

#include "rt/io/OBJReader.hpp"
#include "rt/io/OBJWriter.hpp"

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
        std::cout << "Reading mesh..." << std::endl;
        io::OBJReader r;
        r.setPath(path_);
        mesh_ = r.read();
        img_ = r.getTextureMat();
        imgPath_ = r.getTexturePath();
        uv_ = r.getUVMap();
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
    : path{&path_}
    , mesh{&writer_, &io::OBJWriter::setMesh}
    , image{&writer_, &io::OBJWriter::setTexture}
    , imageSource{&writer_, &io::OBJWriter::setTextureSource}
    , uvMap{&writer_, &io::OBJWriter::setUVMap}
{
    registerInputPort("path", path);
    registerInputPort("mesh", mesh);
    registerInputPort("image", image);
    registerInputPort("imageSource", imageSource);
    registerInputPort("uvMap", uvMap);
    compute = [this]() {
        std::cout << "Writing mesh..." << std::endl;
        writer_.setPath(path_);
        writer_.write();
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