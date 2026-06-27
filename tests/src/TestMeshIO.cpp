#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>

#include "rt/io/MeshIO.hpp"
#include "rt/types/Mesh.hpp"
#include "rt/types/UVMap.hpp"

using namespace rt;

namespace
{
// A single triangle with a per-wedge UV map in the in-memory (top-left) origin.
auto MakeTexturedTriangle() -> std::pair<Mesh::Pointer, UVMap>
{
    auto mesh = Mesh::New();
    mesh->insert_vertex(0.0, 0.0, 0.0);
    mesh->insert_vertex(1.0, 0.0, 0.0);
    mesh->insert_vertex(0.0, 1.0, 0.0);
    mesh->insert_face(0, 1, 2);

    UVMap uv;
    uv.aspect = 2.0F;
    uv.map(0, 0, uv.insert(0.10F, 0.20F));
    uv.map(0, 1, uv.insert(0.30F, 0.40F));
    uv.map(0, 2, uv.insert(0.50F, 0.60F));
    return {mesh, uv};
}
}  // namespace

// In-memory top-left UVs must survive a write -> read round-trip unchanged: the
// bottom-left v-flip applied at the write boundary is exactly undone on read.
TEST(MeshIO, UVRoundTripTopLeftInvariant)
{
    auto [mesh, uv] = MakeTexturedTriangle();

    const std::string path = "TestMeshIO_roundtrip.obj";
    EXPECT_NO_THROW(WriteMesh(path, *mesh, uv));

    MeshReadResult result;
    EXPECT_NO_THROW(result = ReadMesh(path));

    // Geometry survives
    ASSERT_EQ(result.mesh->num_vertices(), mesh->num_vertices());
    ASSERT_EQ(result.mesh->num_faces(), mesh->num_faces());

    // Per-wedge UVs round-trip exactly (compared by wedge, so the assertion is
    // robust to any pool reordering libcore may do on read).
    ASSERT_EQ(result.uvMap.face_corner_count(0), 3U);
    for (std::size_t c = 0; c < 3; ++c) {
        ASSERT_TRUE(result.uvMap.has(0, c));
        const auto& in = uv.get_coordinate(0, c);
        const auto& out = result.uvMap.get_coordinate(0, c);
        EXPECT_FLOAT_EQ(out[0], in[0]);
        EXPECT_FLOAT_EQ(out[1], in[1]);
    }
}

// Guards against the flip silently becoming a no-op on both sides (which a pure
// round-trip would not catch): the coordinates *on disk* must be bottom-left,
// i.e. v stored as (1 - in-memory v).
TEST(MeshIO, WritesBottomLeftVToDisk)
{
    auto [mesh, uv] = MakeTexturedTriangle();

    const std::string path = "TestMeshIO_flip.obj";
    ASSERT_NO_THROW(WriteMesh(path, *mesh, uv));

    // The wedge with in-memory UV (0.10, 0.20) must appear on disk as
    // (0.10, 0.80).
    std::ifstream ifs(path);
    std::string line;
    bool foundFlipped = false;
    while (std::getline(ifs, line)) {
        if (line.rfind("vt ", 0) != 0) {
            continue;
        }
        std::istringstream ss(line.substr(3));
        float u{0.F};
        float v{0.F};
        ss >> u >> v;
        if (std::abs(u - 0.10F) < 1e-4F and std::abs(v - 0.80F) < 1e-4F) {
            foundFlipped = true;
        }
    }
    EXPECT_TRUE(foundFlipped)
        << "on-disk vt should be bottom-left origin (v flipped to 1 - v)";
}
