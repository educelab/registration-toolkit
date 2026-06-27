#pragma once

/** @file */

#include <filesystem>

#include <opencv2/core.hpp>

#include "rt/types/Mesh.hpp"
#include "rt/types/UVMap.hpp"

namespace rt
{

/** @brief Result of rt::ReadMesh */
struct MeshReadResult {
    /** Loaded mesh (carries vertex normals if the file provided them) */
    Mesh::Pointer mesh;
    /** Loaded UV map (empty if the file had no texture coordinates) */
    UVMap uvMap;
    /** Loaded texture image (empty if no texture was referenced/found) */
    cv::Mat texture;
    /** Resolved path to the texture image (empty if none) */
    std::filesystem::path texturePath;
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
auto ReadMesh(const std::filesystem::path& path) -> MeshReadResult;

/**
 * @brief Write a mesh (and optional UV map + texture image) to an OBJ/PLY file
 *
 * Orchestrates libcore's single-texture `write_mesh` and applies the toolkit's
 * conventions: in-memory top-left UV coordinates are flipped back to the file's
 * bottom-left `v` origin. When @p uvMap is non-empty and @p texture is
 * non-empty, the image is written next to the mesh as a TIFF and referenced by
 * the emitted material. A non-empty @p texture with an empty @p uvMap is
 * dropped (with a warning) — there are no UVs to map it to.
 *
 * @throws rt::IOException / std::runtime_error on write failure
 */
void WriteMesh(
    const std::filesystem::path& path,
    const Mesh& mesh,
    const UVMap& uvMap = {},
    const cv::Mat& texture = cv::Mat());

/**
 * @brief Write a mesh (and UV map) to an OBJ/PLY file, copying an existing
 * texture image file
 *
 * Like WriteMesh(path, mesh, uvMap, texture), but instead of writing a
 * `cv::Mat`, the image at @p textureSource is copied next to the mesh
 * (preserving its extension) and referenced by the emitted material. A
 * non-empty @p textureSource with an empty @p uvMap is dropped (with a
 * warning).
 *
 * @throws rt::IOException / std::runtime_error on write failure
 */
void WriteMesh(
    const std::filesystem::path& path,
    const Mesh& mesh,
    const UVMap& uvMap,
    const std::filesystem::path& textureSource);

}  // namespace rt
