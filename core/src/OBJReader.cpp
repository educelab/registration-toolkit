#include "rt/io/OBJReader.hpp"

#include <cstddef>
#include <fstream>
#include <regex>
#include <string>

#include <educelab/core/utils/String.hpp>
#include <opencv2/imgcodecs.hpp>

#include "rt/types/Exceptions.hpp"

using namespace rt;
using namespace rt::io;
using namespace educelab;

namespace fs = rt::filesystem;

namespace
{
// Validation enumeration to ensure proper parsing vertices
enum class RefType {
    Invalid,
    Vertex,
    VertexWithTexture,
    VertexWithNormal,
    VertexWithTextureAndNormal
};

auto ClassifyVertexRef(const std::string& ref) -> RefType
{
    constexpr char delimiter = '/';
    const auto slashCount = std::count(ref.begin(), ref.end(), delimiter);

    // Invalid slash positions
    if (ref.front() == delimiter || ref.back() == delimiter) {
        throw IOException("Invalid face in obj file");
    }

    // No slashes
    if (slashCount == 0) {
        return RefType::Vertex;
    }

    // One slash
    else if (slashCount == 1) {
        return RefType::VertexWithTexture;
    }

    // Two slashes
    else if (slashCount == 2) {
        // Get the two slash positions
        const auto pos0 = ref.find(delimiter, 0);
        const auto pos1 = ref.find(delimiter, pos0 + 1);

        // If positions differ by 1, then v//vn
        // else v/vt/vn
        if (pos1 - pos0 == 1) {
            return RefType::VertexWithNormal;
        } else {
            return RefType::VertexWithTextureAndNormal;
        }
    }
    return RefType::Invalid;
}

auto ParseVertex(const std::vector<std::string_view>& strs) -> cv::Vec3d
{
    auto a = to_numeric<double>(strs[1]);
    auto b = to_numeric<double>(strs[2]);
    auto c = to_numeric<double>(strs[3]);
    return {a, b, c};
}

auto ParseNormal(const std::vector<std::string_view>& strs) -> cv::Vec3d
{
    auto a = to_numeric<double>(strs[1]);
    auto b = to_numeric<double>(strs[2]);
    auto c = to_numeric<double>(strs[3]);
    return {a, b, c};
}

auto ParseTCoord(const std::vector<std::string_view>& strs) -> cv::Vec2d
{
    auto u = to_numeric<double>(strs[1]);
    auto v = to_numeric<double>(strs[2]);
    return {u, v};
}

// Local version of vertex refs/faces
using VRef = std::array<std::optional<std::size_t>, 3>;
using Face = std::vector<VRef>;

auto ParseFace(const std::vector<std::string_view>& strs) -> Face
{
    Face f;
    const std::vector<std::string> sub(std::begin(strs) + 1, std::end(strs));
    const auto faceType = ClassifyVertexRef(sub[0]);

    for (const auto& s : sub) {
        auto vinfo = split(s, '/');
        VRef v{};
        v[0] = to_numeric<std::size_t>(vinfo[0]);
        if (faceType == RefType::VertexWithTexture or
            faceType == RefType::VertexWithTextureAndNormal) {
            v[1] = to_numeric<std::size_t>(vinfo[1]);
        }
        if (faceType == RefType::VertexWithNormal) {
            v[2] = to_numeric<std::size_t>(vinfo[1]);
        }
        if (faceType == RefType::VertexWithTextureAndNormal) {
            v[2] = to_numeric<std::size_t>(vinfo[2]);
        }
        if (faceType == RefType::Invalid) {
            throw IOException("Invalid face in obj file");
        }
        f.emplace_back(v);
    }
    return f;
}

auto ParseMTLLib(
    const filesystem::path& path,
    const std::vector<std::string_view>& strs) -> fs::path
{
    // Get mtl path, relative to OBJ directory
    // Two canonicals because path_ may be relative as well
    const fs::path mtlPath =
        fs::canonical(fs::canonical(path.parent_path()) / strs[1]);

    // Open the mtl file
    std::ifstream ifs(mtlPath.string());
    if (!ifs.good()) {
        throw IOException("Failed to open mtl file for reading");
    }

    // Find the map_kd line
    constexpr std::string_view mapKd{"map_Kd"};

    // Parse the file
    fs::path texturePath;
    std::string line;
    while (std::getline(ifs, line)) {
        line = trim(line);
        if (line.empty()) {
            continue;
        }
        auto mtlstrs = split(line, ' ');
        std::transform(mtlstrs.begin(), mtlstrs.end(), mtlstrs.begin(), &trim);

        // Handle map_Kd
        if (mtlstrs[0] == mapKd) {
            texturePath =
                fs::canonical(fs::canonical(path.parent_path()) / mtlstrs[1]);
        }
        mtlstrs.clear();
    }
    ifs.close();
    return texturePath;
}
}  // namespace

void OBJReader::setPath(const fs::path& p) { path_ = p; }

auto OBJReader::getMesh() -> ITKMesh::Pointer { return mesh_; }

auto OBJReader::getUVMap() -> UVMap { return uvMap_; }

// Read the file
auto OBJReader::read() -> ITKMesh::Pointer
{
    reset_();
    parse_();
    build_mesh_();
    return mesh_;
}

// Get texture image
auto OBJReader::getTextureMat() const -> cv::Mat
{
    if (texturePath_.empty() || !fs::exists(texturePath_)) {
        throw IOException("Invalid or unset texture image path");
    }

    return cv::imread(texturePath_.string(), -1);
}

auto OBJReader::getTexturePath() -> fs::path { return texturePath_; }

// Prepare all data structures to read a new file
void OBJReader::reset_()
{
    vertices_.clear();
    normals_.clear();
    uvs_.clear();
    faces_.clear();
    texturePath_.clear();
}

// Parse the file
void OBJReader::parse_()
{
    constexpr std::string_view vertex{"v"};
    constexpr std::string_view normal{"vn"};
    constexpr std::string_view tcoord{"vt"};
    constexpr std::string_view face{"f"};
    constexpr std::string_view mtllib("mtllib");

    std::ifstream ifs(path_.string());
    if (!ifs.good()) {
        throw IOException("Failed to open file for reading");
    }

    std::string line;
    while (std::getline(ifs, line)) {
        // Parse the line
        line = trim(line);
        if (line.empty()) {
            continue;
        }
        auto strs = split(line, ' ');
        std::transform(strs.begin(), strs.end(), strs.begin(), &trim);

        // Handle vertices
        if (strs[0] == vertex) {
            vertices_.emplace_back(ParseVertex(strs));
        }

        // Handle normals
        else if (strs[0] == normal) {
            normals_.push_back(ParseNormal(strs));
        }

        // Handle texture coordinates
        else if (strs[0] == tcoord) {
            uvs_.push_back(ParseTCoord(strs));
        }

        // Handle faces
        else if (strs[0] == face) {
            faces_.push_back(ParseFace(strs));
        }

        // Handle mtllib
        else if (strs[0] == mtllib) {
            texturePath_ = ParseMTLLib(path_, strs);
        }
    }

    ifs.close();
}

void OBJReader::build_mesh_()
{
    // Reset output structures
    uvMap_ = UVMap();
    uvMap_.setOrigin(UVMap::Origin::BottomLeft);
    mesh_ = ITKMesh::New();

    // Add the vertices to the mesh
    if (vertices_.empty()) {
        throw IOException("No vertices in OBJ file");
    }

    ITKMesh::PointIdentifier pid = 0;
    for (const auto& v : vertices_) {
        mesh_->SetPoint(pid++, v.val);
    }

    for (const auto& uv : uvs_) {
        uvMap_.addUV(uv);
    }

    // Build the faces and UV Map
    // Note: OBJs index vert info from 1
    ITKCell::CellAutoPointer cell;
    ITKMesh::CellIdentifier cid = 0;
    for (const auto& face : faces_) {
        if (face.size() != 3) {
            throw IOException("Parsed unsupported, non-triangular face");
        }

        // Setup output objects
        cell.TakeOwnership(new ITKTriangle);
        UVMap::Face uvFace;
        bool uvFaceGood{true};

        int idInCell{0};
        for (const auto& vinfo : face) {
            // Get the first vertex property (always present)
            const auto vertexID = vinfo[0].value() - 1;
            if (vertexID >= vertices_.size()) {
                throw IOException("Out-of-range vertex reference");
            }
            cell->SetPointId(idInCell, vertexID);

            // Get the UV property index (optional)
            if (vinfo[1]) {
                if (vinfo[1].value() - 1 >= uvs_.size()) {
                    throw IOException("Out-of-range UV reference");
                }
                uvFace[idInCell] = vinfo[1].value() - 1;
            } else {
                uvFaceGood = false;
            }

            // Get the normal property index (optional)
            if (vinfo[2]) {
                const auto nIdx = vinfo[2].value() - 1;
                if (nIdx >= normals_.size()) {
                    throw IOException("Out-of-range normal reference");
                }
                mesh_->SetPointData(vertexID, normals_[nIdx].val);
            }

            idInCell++;
        }
        if (uvFaceGood) {
            uvMap_.addFace(cid, uvFace);
        }
        mesh_->SetCell(cid++, cell);
    }
    uvMap_.setOrigin(UVMap::Origin::TopLeft);
}
