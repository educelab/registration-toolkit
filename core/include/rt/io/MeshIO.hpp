#pragma once

/** @file */

#include <opencv2/core.hpp>

#include "rt/filesystem.hpp"
#include "rt/types/Mesh.hpp"
#include "rt/types/UVMap.hpp"

namespace rt::io
{

/** @brief Result of rt::io::ReadMesh */
struct MeshReadResult {
    /** Loaded mesh (carries vertex normals if the file provided them) */
    Mesh::Pointer mesh;
    /** Loaded UV map (empty if the file had no texture coordinates) */
    UVMap uvMap;
    /** Loaded texture image (empty if no texture was referenced/found) */
    cv::Mat texture;
    /** Resolved path to the texture image (empty if none) */
    filesystem::path texturePath;
};

/**
 * @brief Read a mesh (and its UV map + texture) from an OBJ or PLY file
 *
 * Orchestrates libcore's `read_mesh` (extension-dispatched OBJ/PLY) and applies
 * the toolkit's I/O-boundary conventions: file UV coordinates are bottom-left
 * origin, so the `v` component is flipped to the in-memory top-left invariant
 * (see CONTEXT.md). If the file references a texture image (OBJ `map_Kd` / PLY
 * `TextureFile`) that exists on disk, it is loaded into the result.
 *
 * @throws rt::IOException / std::runtime_error on read failure
 */
auto ReadMesh(const filesystem::path& path) -> MeshReadResult;

/**
 * @brief Write a mesh (and optional UV map + texture) to an OBJ or PLY file
 *
 * Orchestrates libcore's single-texture `write_mesh` and applies the toolkit's
 * conventions: in-memory top-left UV coordinates are flipped back to the
 * file's bottom-left `v` origin. Texture handling (only when @p uvMap is not
 * empty):
 *  - if @p texture is non-empty, it is written next to the mesh as a TIFF and
 *    referenced by the emitted material;
 *  - else if @p textureSource is non-empty, that image file is copied next to
 *    the mesh (preserving its extension) and referenced;
 *  - else the mesh is written with UV coordinates but no material/texture.
 *
 * @throws rt::IOException / std::runtime_error on write failure
 */
void WriteMesh(
    const filesystem::path& path,
    const Mesh& mesh,
    const UVMap& uvMap = {},
    const cv::Mat& texture = cv::Mat(),
    const filesystem::path& textureSource = {});

}  // namespace rt::io
