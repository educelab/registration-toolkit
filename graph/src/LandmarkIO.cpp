#include "rt/graph/LandmarkIO.hpp"

#include "rt/Logging.hpp"

namespace rtg = rt::graph;
namespace fs = std::filesystem;

rtg::LandmarkReaderNode::LandmarkReaderNode()
{
    registerInputPort("path", path);
    registerOutputPort("fixedLandmarks", fixedLandmarks);
    registerOutputPort("movingLandmarks", movingLandmarks);
    compute = [this]() {
        rt::logger()->info("Loading landmarks from file: {}", path_.string());
        reader_.setLandmarksPath(path_);
        reader_.read();
        fixed_ = reader_.getFixedLandmarks();
        moving_ = reader_.getMovingLandmarks();
    };
}

smgl::Metadata rtg::LandmarkReaderNode::serialize_(bool, const fs::path&)
{
    return {{"path", path_.string()}};
}

void rtg::LandmarkReaderNode::deserialize_(
    const smgl::Metadata& meta, const fs::path&)
{
    path_ = meta["path"].get<std::string>();
    compute();
}

rtg::LandmarkWriterNode::LandmarkWriterNode()
{
    registerInputPort("path", path);
    registerInputPort("fixed", fixed);
    registerInputPort("moving", moving);
    compute = [this]() {
        rt::logger()->info("Writing landmarks to file: {}", path_.string());
        writer_.setPath(path_);
        writer_.setFixedLandmarks(fixed_);
        writer_.setMovingLandmarks(moving_);
        writer_.write();
    };
}

smgl::Metadata rtg::LandmarkWriterNode::serialize_(bool, const fs::path&)
{
    return {{"path", path_.string()}};
}

void rtg::LandmarkWriterNode::deserialize_(
    const smgl::Metadata& meta, const fs::path&)
{
    path_ = meta["path"].get<std::string>();
}
