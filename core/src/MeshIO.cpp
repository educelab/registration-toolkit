#include "rt/io/MeshIO.hpp"

#include <vector>

#include <educelab/core/io/MeshIO.hpp>

#include "rt/Logging.hpp"
#include "rt/io/ImageIO.hpp"

using namespace educelab;
namespace fs = std::filesystem;

namespace
{
/**
 * Return a copy of @p in with the `v` component of every pool coordinate
 * flipped (top-left <-> bottom-left). The transform is its own inverse, so the
 * same routine is used on read and write. The per-wedge mapping, chart indices,
 * and `aspect` ride along on the copy untouched; only the pool's `v` values
 * change — no remapping/rebuild required.
 */
auto FlipV(const rt::UVMap& in) -> rt::UVMap
{
    rt::UVMap out = in;
    for (std::size_t i = 0; i < out.size(); ++i) {
        out.at(i)[1] = 1.0F - out.at(i)[1];
    }
    return out;
}

/**
 * Shared write implementation. At most one of @p texture / @p textureSource is
 * expected to be non-empty. When @p uvMap is empty there is nothing to map a
 * texture to, so any provided texture is dropped with a warning.
 */
void WriteMeshImpl(
    const fs::path& path,
    const rt::Mesh& mesh,
    const rt::UVMap& uvMap,
    const cv::Mat& texture,
    const fs::path& textureSource)
{
    rt::logger()->debug("Writing mesh: {}", path.string());

    const bool haveTexture = not texture.empty();
    const bool haveSource = not textureSource.empty();

    // No UV map: write geometry (and normals) only
    if (uvMap.empty()) {
        if (haveTexture or haveSource) {
            rt::logger()->warn(
                "Mesh has no UV map; the provided texture will not be "
                "written.");
        }
        write_mesh(path, mesh);
        return;
    }

    // Flip UVs back to the file's (bottom-left) v origin
    const auto flipped = FlipV(uvMap);

    // UVs but no texture image
    if (not haveTexture and not haveSource) {
        write_mesh(path, mesh, flipped);
        return;
    }

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
}
}  // namespace

auto rt::ReadMesh(const fs::path& path) -> rt::MeshReadResult
{
    rt::logger()->debug("Reading mesh: {}", path.string());

    MeshReadResult result;
    result.mesh = Mesh::New();
    std::vector<std::filesystem::path> texturePaths;

    // libcore reads UVs with the file's (bottom-left) v origin
    read_mesh(path, *result.mesh, result.uvMap, texturePaths);

    // Flip v to the in-memory top-left invariant
    result.uvMap = FlipV(result.uvMap);

    // Resolve and load every referenced texture, keeping the vectors aligned
    // with the UV map's chart indices (chart i ↔ textures[i]). A missing image
    // is stored as an empty cv::Mat / empty path so alignment is preserved.
    result.textures.reserve(texturePaths.size());
    result.texturePaths.reserve(texturePaths.size());
    for (const auto& rel : texturePaths) {
        fs::path texPath = path.parent_path() / rel.string();
        if (fs::exists(texPath)) {
            result.texturePaths.push_back(texPath);
            result.textures.push_back(rt::ReadImage(texPath));
        } else {
            rt::logger()->warn(
                "Referenced texture not found: {}", texPath.string());
            result.texturePaths.emplace_back();
            result.textures.emplace_back();
        }
    }

    return result;
}

void rt::WriteMesh(
    const fs::path& path,
    const Mesh& mesh,
    const UVMap& uvMap,
    const cv::Mat& texture)
{
    WriteMeshImpl(path, mesh, uvMap, texture, {});
}

void rt::WriteMesh(
    const fs::path& path,
    const Mesh& mesh,
    const UVMap& uvMap,
    const fs::path& textureSource)
{
    WriteMeshImpl(path, mesh, uvMap, cv::Mat(), textureSource);
}
