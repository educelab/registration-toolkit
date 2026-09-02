#include <gtest/gtest.h>
#include <smgl/Metadata.hpp>

#include "rt/graph.hpp"
#include "rt/graph/MeshOps.hpp"

using Node = rt::graph::ReorderTextureNode;
using OrientationMode = Node::OrientationMode;

namespace
{
// Node (and port) type names are resolved through smgl's factory, so the
// graph's types have to be registered before anything can be serialized.
const bool kNodesRegistered = rt::graph::RegisterNodes();

// Serialize a node whose orientation mode has been set, restore a fresh node
// from that metadata, and hand back what the restored node serializes.
//
// A setting that serialize_() writes but deserialize_() forgets to read
// reverts silently to its default, so re-serializing the restored node is what
// exposes the omission: the two metadata blobs disagree.
auto RoundTripOrientation(OrientationMode mode) -> smgl::Metadata
{
    EXPECT_TRUE(kNodesRegistered);
    Node node;
    node.orientationMode.post(mode, true);
    const auto meta = node.serialize(false, ".");

    Node restored;
    restored.deserialize(meta, ".");
    return restored.serialize(false, ".");
}
}  // namespace

TEST(ReorderTextureNodeSerialization, OrientationModeSerializesByName)
{
    Node node;

    node.orientationMode.post(OrientationMode::Canonical, true);
    EXPECT_EQ(node.serialize(false, ".")["data"]["orientationMode"],
              "canonical");

    node.orientationMode.post(OrientationMode::OBB, true);
    EXPECT_EQ(node.serialize(false, ".")["data"]["orientationMode"], "obb");
}

TEST(ReorderTextureNodeSerialization, OrientationModeSurvivesRoundTrip)
{
    EXPECT_EQ(
        RoundTripOrientation(OrientationMode::Canonical)["data"]
                                                        ["orientationMode"],
        "canonical");
    EXPECT_EQ(
        RoundTripOrientation(OrientationMode::OBB)["data"]["orientationMode"],
        "obb");
}

// Graph caches written before the orientation mode existed have no entry for
// it. Loading one must not throw, and must leave the default in place.
TEST(ReorderTextureNodeSerialization, MissingOrientationModeKeepsDefault)
{
    Node node;
    auto meta = node.serialize(false, ".");
    meta["data"].erase("orientationMode");

    Node restored;
    ASSERT_NO_THROW(restored.deserialize(meta, "."));
    EXPECT_EQ(restored.serialize(false, ".")["data"]["orientationMode"], "obb");
}
