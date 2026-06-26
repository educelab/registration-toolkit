#include <gtest/gtest.h>

#include <cstddef>
#include <fstream>
#include <random>

#include "rt/io/UVMapIO.hpp"
#include "rt/types/UVMap.hpp"

using namespace rt;

static auto RandomUVMap(std::size_t numUVs, std::size_t numFaces) -> UVMap
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> randReal(
        0.0F, std::nextafter(1.0F, std::numeric_limits<float>::max()));
    std::uniform_int_distribution<std::size_t> randInt(0, numUVs - 1);
    std::uniform_int_distribution<std::size_t> randChart(0, 3);

    // UV Map
    UVMap uv;
    uv.aspect = randReal(gen) + 0.5F;

    // Random UV coordinates (pool), each with a chart index
    for (std::size_t i = 0; i < numUVs; i++) {
        const auto idx = uv.insert(randReal(gen), randReal(gen));
        uv.at(idx).chart = randChart(gen);
    }

    // Random per-wedge faces (3 corners each, distinct pool indices)
    for (std::size_t f = 0; f < numFaces; f++) {
        auto a = randInt(gen);
        auto b = randInt(gen);
        while (b == a) {
            b = randInt(gen);
        }
        auto c = randInt(gen);
        while (c == a or c == b) {
            c = randInt(gen);
        }
        uv.map(f, 0, a);
        uv.map(f, 1, b);
        uv.map(f, 2, c);
    }

    return uv;
}

TEST(UVMapIO, RoundTrip)
{
    // Get random UV Map
    auto orig = RandomUVMap(100, 75);

    // Round trip
    EXPECT_NO_THROW(WriteUVMap("TestUVMapIO_RoundTrip.uvm", orig));

    UVMap result;
    EXPECT_NO_THROW(result = ReadUVMap("TestUVMapIO_RoundTrip.uvm"));

    // Compare sizes and per-map metadata
    EXPECT_EQ(result.size(), orig.size());
    EXPECT_EQ(result.num_faces(), orig.num_faces());
    EXPECT_FLOAT_EQ(result.aspect, orig.aspect);

    // Compare the coordinate pool (position + chart)
    for (std::size_t i = 0; i < orig.size(); ++i) {
        const auto& o = orig.at(i);
        const auto& r = result.at(i);
        EXPECT_FLOAT_EQ(r[0], o[0]);
        EXPECT_FLOAT_EQ(r[1], o[1]);
        EXPECT_EQ(r.chart, o.chart);
    }

    // Compare the per-wedge mapping
    for (std::size_t f = 0; f < orig.num_faces(); ++f) {
        EXPECT_EQ(result.face_corner_count(f), orig.face_corner_count(f));
        for (std::size_t c = 0; c < orig.face_corner_count(f); ++c) {
            EXPECT_EQ(result.has(f, c), orig.has(f, c));
            if (orig.has(f, c)) {
                EXPECT_EQ(result.get(f, c), orig.get(f, c));
            }
        }
    }
}

// Legacy v1 (per-face) caches must still load: the reader converts them to the
// current per-wedge representation (top-left coords, default chart 0, aspect
// recovered from width/height).
TEST(UVMapIO, ReadsLegacyV1)
{
    const std::string path = "TestUVMapIO_v1.uvm";
    {
        std::ofstream ofs{path, std::ios::binary};
        ofs << "filetype: uvmap\n"
            << "version: 1\n"
            << "type: per-face\n"
            << "size: 3\n"
            << "width: 800\n"
            << "height: 400\n"
            << "origin: 0\n"
            << "faces: 1\n"
            << "<>\n";
        double uvs[3][2] = {{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}};
        for (auto& uv : uvs) {
            ofs.write(reinterpret_cast<char*>(uv), 2 * sizeof(double));
        }
        std::size_t idx = 7;
        std::size_t f[3] = {0, 1, 2};
        ofs.write(reinterpret_cast<char*>(&idx), sizeof(std::size_t));
        ofs.write(reinterpret_cast<char*>(f), 3 * sizeof(std::size_t));
    }

    UVMap m;
    EXPECT_NO_THROW(m = ReadUVMap(path));

    // Pool + aspect
    EXPECT_EQ(m.size(), 3U);
    EXPECT_FLOAT_EQ(m.aspect, 2.0F);  // 800 / 400
    EXPECT_FLOAT_EQ(m.at(0)[0], 0.1F);
    EXPECT_FLOAT_EQ(m.at(0)[1], 0.2F);
    EXPECT_FLOAT_EQ(m.at(2)[0], 0.5F);
    EXPECT_FLOAT_EQ(m.at(2)[1], 0.6F);
    EXPECT_EQ(m.at(0).chart, 0U);

    // Face 7's three corners map to pool indices 0, 1, 2
    ASSERT_TRUE(m.has(7, 0));
    ASSERT_TRUE(m.has(7, 1));
    ASSERT_TRUE(m.has(7, 2));
    EXPECT_EQ(m.get(7, 0), 0U);
    EXPECT_EQ(m.get(7, 1), 1U);
    EXPECT_EQ(m.get(7, 2), 2U);
}
