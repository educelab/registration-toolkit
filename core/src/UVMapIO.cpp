#include "rt/io/UVMapIO.hpp"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <regex>
#include <sstream>
#include <string_view>

#include <educelab/core/utils/String.hpp>

#include "rt/types/Exceptions.hpp"

using namespace educelab;
namespace fs = rt::filesystem;

namespace
{
/** Current .uvm format version (v2: per-wedge pool + chart + aspect) */
constexpr int kUVMapVersion{2};
}  // namespace

void rt::WriteUVMap(const fs::path& path, const UVMap& uvMap)
{
    std::ofstream ofs{path.string(), std::ios::binary};
    if (!ofs.is_open()) {
        auto msg = "could not open file '" + path.string() + "'";
        throw IOException(msg);
    }

    // Header
    std::stringstream ss;
    ss << "filetype: uvmap\n";
    ss << "version: " << kUVMapVersion << "\n";
    ss << "type: per-wedge\n";
    ss << "aspect: "
       << std::setprecision(std::numeric_limits<float>::max_digits10)
       << uvMap.aspect << "\n";
    ss << "uvs: " << uvMap.size() << "\n";
    ss << "faces: " << uvMap.num_faces() << "\n";
    ss << "<>\n";
    ofs << ss.rdbuf();

    // Write the UV coordinate pool: (u, v) as float, then chart index
    for (std::size_t i = 0; i < uvMap.size(); ++i) {
        const auto& uv = uvMap.at(i);
        const float u = uv[0];
        const float v = uv[1];
        const std::size_t chart = uv.chart;
        ofs.write(reinterpret_cast<const char*>(&u), sizeof(float));
        ofs.write(reinterpret_cast<const char*>(&v), sizeof(float));
        ofs.write(reinterpret_cast<const char*>(&chart), sizeof(std::size_t));
    }

    // Write the per-wedge mapping: for each face, a corner count followed by
    // one (mapped flag, pool index) entry per corner.
    for (std::size_t f = 0; f < uvMap.num_faces(); ++f) {
        const std::size_t corners = uvMap.face_corner_count(f);
        ofs.write(
            reinterpret_cast<const char*>(&corners), sizeof(std::size_t));
        for (std::size_t c = 0; c < corners; ++c) {
            const std::uint8_t mapped = uvMap.has(f, c) ? 1 : 0;
            ofs.write(reinterpret_cast<const char*>(&mapped), sizeof(mapped));
            if (mapped != 0) {
                const std::size_t idx = uvMap.get(f, c);
                ofs.write(
                    reinterpret_cast<const char*>(&idx), sizeof(std::size_t));
            }
        }
    }

    ofs.close();
}

auto rt::ReadUVMap(const fs::path& path) -> rt::UVMap
{
    std::ifstream ifs{path.string(), std::ios::binary};
    if (!ifs.is_open()) {
        auto msg = "could not open file '" + path.string() + "'";
        throw IOException(msg);
    }

    struct Header {
        std::string fileType;
        int version{0};
        std::string type;
        float aspect{1.0F};
        std::size_t uvs{0};
        std::size_t faces{0};
        // Legacy (v1) fields
        std::size_t size{0};
        double width{0};
        double height{0};
    };

    // Header keys
    std::regex comments{"^#"};
    constexpr std::string_view fileType{"filetype"};
    constexpr std::string_view version{"version"};
    constexpr std::string_view type{"type"};
    constexpr std::string_view aspect{"aspect"};
    constexpr std::string_view uvs{"uvs"};
    constexpr std::string_view faces{"faces"};
    // Legacy (v1) keys
    constexpr std::string_view size{"size"};
    constexpr std::string_view width{"width"};
    constexpr std::string_view height{"height"};
    std::regex headerTerminator{"^<>$"};

    Header h;
    std::string line;
    while (std::getline(ifs, line)) {
        line = trim(line);
        auto strs = split(line, ":");
        std::transform(
            std::begin(strs), std::end(strs), std::begin(strs), &trim);

        if (std::regex_match(std::string(strs[0]), comments)) {
            continue;
        } else if (strs[0] == fileType) {
            h.fileType = strs[1];
        } else if (strs[0] == version) {
            h.version = to_numeric<int>(strs[1]);
        } else if (strs[0] == type) {
            h.type = strs[1];
        } else if (strs[0] == aspect) {
            h.aspect = to_numeric<float>(strs[1]);
        } else if (strs[0] == uvs) {
            h.uvs = to_numeric<std::size_t>(strs[1]);
        } else if (strs[0] == faces) {
            h.faces = to_numeric<std::size_t>(strs[1]);
        } else if (strs[0] == size) {
            h.size = to_numeric<std::size_t>(strs[1]);
        } else if (strs[0] == width) {
            h.width = to_numeric<double>(strs[1]);
        } else if (strs[0] == height) {
            h.height = to_numeric<double>(strs[1]);
        } else if (std::regex_match(line, headerTerminator)) {
            break;
        } else {
            continue;
        }
    }

    // Sanity check the header
    if (h.fileType.empty()) {
        throw IOException("Must provide file type");
    } else if (h.fileType != "uvmap") {
        throw IOException("File is not a UVMap");
    }

    UVMap map;

    // ---- Legacy v1 (per-face) caches: convert on load ----
    // v1 stored a flat UV pool (2 doubles each, already top-left origin) and a
    // map of face index -> 3 pool indices. There is no chart data (default 0)
    // and no v-flip (top-left storage matches the in-memory invariant). The
    // aspect is recovered from the old width/height ratio.
    if (h.version == 1) {
        map.aspect =
            (h.height != 0.0) ? static_cast<float>(h.width / h.height) : 1.0F;

        // UV pool (2 doubles -> float)
        for (std::size_t i = 0; i < h.size; ++i) {
            double uv[2]{0.0, 0.0};
            ifs.read(reinterpret_cast<char*>(uv), 2 * sizeof(double));
            std::ignore = map.insert(
                static_cast<float>(uv[0]), static_cast<float>(uv[1]));
        }

        // Faces: (face index, 3 pool indices) -> per-wedge mapping
        for (std::size_t i = 0; i < h.faces; ++i) {
            std::size_t idx{0};
            std::size_t f[3]{0, 0, 0};
            ifs.read(reinterpret_cast<char*>(&idx), sizeof(std::size_t));
            ifs.read(reinterpret_cast<char*>(f), 3 * sizeof(std::size_t));
            for (std::size_t c = 0; c < 3; ++c) {
                map.map(idx, c, f[c]);
            }
        }

        return map;
    }

    // ---- Current v2 (per-wedge) format ----
    if (h.version != kUVMapVersion) {
        auto msg = "Version mismatch. UVMap file version is " +
                   std::to_string(h.version) + ", processing version is " +
                   std::to_string(kUVMapVersion) + ".";
        throw IOException(msg);
    } else if (h.type != "per-wedge") {
        throw IOException("UVMap type not supported: " + h.type);
    }

    map.aspect = h.aspect;

    // Read the UV coordinate pool
    for (std::size_t i = 0; i < h.uvs; ++i) {
        float u{0.F};
        float v{0.F};
        std::size_t chart{0};
        ifs.read(reinterpret_cast<char*>(&u), sizeof(float));
        ifs.read(reinterpret_cast<char*>(&v), sizeof(float));
        ifs.read(reinterpret_cast<char*>(&chart), sizeof(std::size_t));
        const auto idx = map.insert(u, v);
        map.at(idx).chart = chart;
    }

    // Read the per-wedge mapping
    for (std::size_t f = 0; f < h.faces; ++f) {
        std::size_t corners{0};
        ifs.read(reinterpret_cast<char*>(&corners), sizeof(std::size_t));
        for (std::size_t c = 0; c < corners; ++c) {
            std::uint8_t mapped{0};
            ifs.read(reinterpret_cast<char*>(&mapped), sizeof(mapped));
            if (mapped != 0) {
                std::size_t idx{0};
                ifs.read(reinterpret_cast<char*>(&idx), sizeof(std::size_t));
                map.map(f, c, idx);
            }
        }
    }

    return map;
}
