#include "rt/graph.hpp"

#include <smgl/Node.hpp>

#include "rt/Version.hpp"

using namespace rt;
using namespace rt::graph;

static auto RegisterNodesImpl() -> bool
{
    bool registered{true};

    // clang-format off
    // ImageIO
    registered &= smgl::RegisterNodes(
        SMGL_NODE(rt::graph::ReadImageNode),
        SMGL_NODE(rt::graph::WriteImageNode),
        SMGL_NODE(rt::graph::WriteImageSeriesNode));

    // ImageOps
    registered &= smgl::RegisterNodes(
        SMGL_NODE(rt::graph::ColorConvertNode),
        SMGL_NODE(rt::graph::PositionMapTransformNode));

    // Landmark Registration
    registered &= smgl::RegisterNodes(
        SMGL_NODE(rt::graph::CompositeTransformNode),
        SMGL_NODE(rt::graph::LandmarkReaderNode),
        SMGL_NODE(rt::graph::LandmarkDetectorNode),
        SMGL_NODE(rt::graph::LandmarkWriterNode),
        SMGL_NODE(rt::graph::AffineLandmarkRegistrationNode),
        SMGL_NODE(rt::graph::BSplineLandmarkWarpingNode));

    // Transforms
    registered &= smgl::RegisterNodes(
        SMGL_NODE(rt::graph::ImageResampleNode),
        SMGL_NODE(rt::graph::TransformSeriesResampleNode),
        SMGL_NODE(rt::graph::TransformLandmarksNode),
        SMGL_NODE(rt::graph::WriteTransformNode),
        SMGL_NODE(rt::graph::TransformUVMapNode));

    // Deformable Registration
    registered &= smgl::RegisterNodes(
        SMGL_NODE(rt::graph::DeformableRegistrationNode));

    // MeshIO
    registered &= smgl::RegisterNodes(
        SMGL_NODE(rt::graph::MeshReadNode),
        SMGL_NODE(rt::graph::MeshWriteNode));

    // MeshOps
    registered &= smgl::RegisterNodes(SMGL_NODE(rt::graph::ReorderTextureNode));
    // clang-format on

    return registered;
}

auto rt::graph::RegisterNodes() -> bool
{
    static auto registered = RegisterNodesImpl();
    return registered;
}

auto rt::graph::ProjectMetadata() -> smgl::Metadata
{
    // clang-format off
    return
    {
        {"version", ProjectInfo::VersionString()},
        {"git-url", ProjectInfo::RepositoryURL()},
        {"git-hash", ProjectInfo::RepositoryHash()},
    };
    // clang-format on
}
