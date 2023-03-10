#pragma once

/** @file */

#include <fstream>
#include <map>
#include <string>

#include <opencv2/core.hpp>

#include "rt/filesystem.hpp"
#include "rt/types/ITKMesh.hpp"
#include "rt/types/UVMap.hpp"

namespace rt::io
{
/**
 * @class OBJWriter
 * @author Seth Parker
 * @date 6/24/15
 *
 * @brief Write an ITKMesh to an OBJ file
 *
 * Writes both textured and untextured meshes in ASCII OBJ format. Texture
 * information is automatically written if a UV map is set and is not empty.
 *
 */
class OBJWriter
{

public:
    /**@{*/
    /** @brief Default constructor */
    OBJWriter() = default;

    /** @brief Constructor with output path and input mesh */
    OBJWriter(filesystem::path outputPath, ITKMesh::Pointer mesh);

    /** @brief Constructor with output path and textured mesh information */
    OBJWriter(
        filesystem::path outputPath,
        ITKMesh::Pointer mesh,
        UVMap uvMap,
        cv::Mat uvImg);
    /**@}*/

    /**@{*/
    /** @brief Set the output path
     *
     * write() and validate() will fail if path does not have an expected
     * file extension (.obj/.OBJ).
     */
    void setPath(const filesystem::path& path);

    /** @brief Set the input mesh */
    void setMesh(const ITKMesh::Pointer& mesh);

    /** @brief Set the input UV Map */
    void setUVMap(const UVMap& uvMap);

    /**
     * @brief Set the input texture image
     *
     * The texture image to be used as the output texture image. Calling this
     * function has the effect of clearing the path previously provided to
     * setTextureSource().
     */
    void setTexture(const cv::Mat& uvImg);

    /**
     * @brief Set the input texture image source path
     *
     * If provided, copy the image at the provided path to the output texture
     * path rather than writing a new image. Calling this function has the
     * effect of clearing the texture previously provided to setTexture().
     */
    void setTextureSource(const filesystem::path& path);

    /** @brief Validate parameters */
    auto validate() -> bool;
    /**@}*/

    /**@{*/
    /** @brief Write the OBJ to disk
     *
     * If UV Map is not empty, automatically writes MTL and texture image.
     */
    auto write() -> int;
    /**@}*/

private:
    /** Output file path */
    filesystem::path outputPath_;
    /** Output OBJ filestream */
    std::ofstream outputMesh_;
    /** Output MTL filestream */
    std::ofstream outputMTL_;

    /**
     * Keeps track of what info we have about each point in the mesh. Used for
     * building OBJ faces.
     *
     * v = vertex index number \n
     * vt = UV coordinate index number \n
     * vn = vertex normal index number \n
     */
    using PointLink = cv::Vec<std::size_t, 3>;

    /** [ Point Index, {v, vt, vn} ] */
    std::map<std::size_t, PointLink> pointLinks_;

    /** Input mesh */
    ITKMesh::Pointer mesh_;
    /** Input UV map */
    UVMap uvMap_;
    /** Input texture image */
    cv::Mat texture_;
    /** Input texture image path */
    filesystem::path textureSrc_;

    /** Write the OBJ file */
    auto write_obj_() -> int;
    /** Write the MTL file */
    auto write_mtl_() -> int;
    /** Write the texture file */
    auto write_texture_() -> int;

    /** Write the OBJ header */
    auto write_header_() -> int;
    /** Write the OBJ vertices */
    auto write_vertices_() -> int;
    /** Write the OBJ texture coordinates */
    auto write_texture_coordinates_() -> int;
    /** Write the OBJ faces */
    auto write_faces_() -> int;
};

}  // namespace rt
