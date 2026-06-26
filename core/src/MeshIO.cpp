#include "rt/io/MeshIO.hpp"

#include <vector>

#include <educelab/core/io/MeshIO.hpp>

#include "rt/Logging.hpp"
#include "rt/io/ImageIO.hpp"

using namespace educelab;
namespace fs = rt::filesystem;

namespace
{
/**
 * Flip the `v` component of every pool coordinate (top-left <-> bottom-left).
 * The transform is its own inverse, so the same routine is used on read and
 * write. Returns a flipped copy, preserving `aspect` and chart indices.
 */
auto FlipV(const rt::UVMap& in) -> rt::UVMap
{
    rt::UVMap out;
    out.aspect = in.aspect;
    for (std::size_t i = 0; i < in.size(); ++i) {
        const auto& uv = in.at(i);
        const auto idx = out.insert(uv[0], 1.0F - uv[1]);
        out.at(idx).chart = uv.chart;
    }
    for (std::size_t f = 0; f < in.num_faces(); ++f) {
        const auto corners = in.face_corner_count(f);
        for (std::size_t c = 0; c < corners; ++c) {
            if (in.has(f, c)) {
                out.map(f, c, in.get(f, c));
            }
        }
    }
    return out;
}
}  // namespace

auto rt::io::ReadMesh(const fs::path& path) -> rt::io::MeshReadResult
{
    rt::logger()->debug("Reading mesh: {}", path.string());

    MeshReadResult result;
    result.mesh = Mesh::New();
    std::vector<std::filesystem::path> texturePaths;

    // libcore reads UVs with the file's (bottom-left) v origin
    read_mesh(path, *result.mesh, result.uvMap, texturePaths);

    // Flip v to the in-memory top-left invariant
    result.uvMap = FlipV(result.uvMap);

    // Resolve and load the first referenced texture, if any
    if (not texturePaths.empty()) {
        fs::path texPath = path.parent_path() / texturePaths.front().string();
        if (fs::exists(texPath)) {
            result.texturePath = texPath;
            result.texture = rt::ReadImage(texPath);
        } else {
            rt::logger()->warn(
                "Referenced texture not found: {}", texPath.string());
        }
    }

    return result;
}

void rt::io::WriteMesh(
    const fs::path& path,
    const Mesh& mesh,
    const UVMap& uvMap,
    const cv::Mat& texture,
    const fs::path& textureSource)
{
    rt::logger()->debug("Writing mesh: {}", path.string());

    // No UV map: write geometry (and normals) only
    if (uvMap.empty()) {
        write_mesh(path, mesh);
        return;
    }

    // Flip UVs back to the file's (bottom-left) v origin
    const auto flipped = FlipV(uvMap);

    const bool haveTexture = not texture.empty();
    const bool haveSource = not textureSource.empty();

    if (haveTexture or haveSource) {
        // Resolve the output texture path (next to the mesh)
        fs::path texOut = path;
        if (haveTexture) {
            texOut.replace_extension("tif");
        } else {
            texOut.replace_extension(textureSource.extension());
        }

        // Write the mesh + material referencing the texture filename
        write_mesh(path, mesh, flipped, texOut.filename());

        // Write or copy the texture image itself
        if (haveTexture) {
            rt::logger()->debug("Writing texture image: {}", texOut.string());
            rt::WriteImage(texOut, texture);
        } else {
            rt::logger()->debug(
                "Copying texture image: {} -> {}", textureSource.string(),
                texOut.string());
            fs::copy_file(
                textureSource, texOut, fs::copy_options::overwrite_existing);
        }
    } else {
        // UVs but no texture image
        write_mesh(path, mesh, flipped);
    }
}
