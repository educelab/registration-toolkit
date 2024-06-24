#include "rt/io/UVMapIO.hpp"

#include <cstddef>
#include <fstream>
#include <regex>
#include <sstream>
#include <string_view>

#include <educelab/core/utils/String.hpp>

#include "rt/types/Exceptions.hpp"

using namespace educelab;
namespace fs = rt::filesystem;

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
    ss << "version: 1\n";
    ss << "type: per-face\n";
    ss << "size: " << uvMap.size() << "\n";
    ss << "width: " << uvMap.ratio().width << "\n";
    ss << "height: " << uvMap.ratio().height << "\n";
    ss << "origin: " << static_cast<int>(uvMap.origin()) << "\n";
    ss << "faces: " << uvMap.size_faces() << "\n";
    ss << "<>\n";
    ofs << ss.rdbuf();

    // Write the UV coords
    for (const auto& uv : uvMap.uvs_as_vector()) {
        ofs.write(reinterpret_cast<const char*>(uv.val), 2 * sizeof(double));
    }

    // Write the faces
    for (const auto& f : uvMap.faces_as_map()) {
        ofs.write(reinterpret_cast<const char*>(&f.first), sizeof(std::size_t));
        ofs.write(
            reinterpret_cast<const char*>(f.second.val),
            3 * sizeof(std::size_t));
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
        std::size_t size{0};
        double width{0};
        double height{0};
        int origin{-1};
        std::size_t faces{0};
    };

    // Regexes
    std::regex comments{"^#"};
    constexpr std::string_view fileType{"filetype"};
    constexpr std::string_view version{"version"};
    constexpr std::string_view type{"type"};
    constexpr std::string_view size{"size"};
    constexpr std::string_view width{"width"};
    constexpr std::string_view height{"height"};
    constexpr std::string_view origin{"origin"};
    constexpr std::string_view faces{"faces"};
    std::regex headerTerminator{"^<>$"};

    Header h;
    std::string line;
    while (std::getline(ifs, line)) {
        line = trim(line);
        auto strs = split(line, ':');
        std::transform(
            std::begin(strs), std::end(strs), std::begin(strs), &trim);

        // Comments: look like:
        // # This is a comment
        //    # This is another comment
        if (std::regex_match(std::string(strs[0]), comments)) {
            continue;
        }

        // File type
        else if (strs[0] == fileType) {
            h.fileType = strs[1];
        }

        // Version
        else if (strs[0] == version) {
            h.version = to_numeric<int>(strs[1]);
        }

        // Type
        else if (strs[0] == type) {
            h.type = strs[1];
        }

        // Size
        else if (strs[0] == size) {
            h.size = to_numeric<std::size_t>(strs[1]);
        }

        // Width
        else if (strs[0] == width) {
            h.width = to_numeric<double>(strs[1]);
        }

        // Height
        else if (strs[0] == height) {
            h.height = to_numeric<double>(strs[1]);
        }

        // Origin
        else if (strs[0] == origin) {
            h.origin = to_numeric<int>(strs[1]);
        }

        // Faces
        else if (strs[0] == faces) {
            h.faces = to_numeric<std::size_t>(strs[1]);
        }

        // End of the header
        else if (std::regex_match(line, headerTerminator)) {
            break;
        }

        // Ignore everything else
        else {
            continue;
        }
    }

    // Sanity check. Do we have a valid UVMap header?
    if (h.fileType.empty()) {
        throw IOException("Must provide file type");
    } else if (h.fileType != "uvmap") {
        throw IOException("File is not a UVMap");
    } else if (h.version != 1) {
        auto msg = "Version mismatch. UVMap file version is " +
                   std::to_string(h.version) + ", processing version is 1.";
        throw IOException(msg);
    } else if (h.type.empty()) {
        throw IOException("Must provide UVMap type");
    } else if (h.type != "per-face") {
        throw IOException("UVMap type not supported: " + h.type);
    } else if (h.width == 0 or h.height == 0) {
        throw IOException("UVMap cannot have dimensions == 0");
    } else if (h.origin == -1) {
        throw IOException("UVMap file does not contain origin");
    }

    // Construct the UVMap
    UVMap map;
    map.setOrigin(static_cast<UVMap::Origin>(h.origin));
    map.ratio(h.width, h.height);

    // Read all of the points
    for (std::size_t i = 0; i < h.size; i++) {
        std::ignore = i;
        cv::Vec2d uv;
        ifs.read(reinterpret_cast<char*>(uv.val), 2 * sizeof(double));
        map.addUV(uv);
    }

    // Read all of the faces
    for (std::size_t i = 0; i < h.faces; i++) {
        std::ignore = i;
        std::size_t idx{0};
        ifs.read(reinterpret_cast<char*>(&idx), sizeof(std::size_t));
        UVMap::Face f;
        ifs.read(reinterpret_cast<char*>(f.val), 3 * sizeof(std::size_t));
        map.addFace(idx, f);
    }

    return map;
}
