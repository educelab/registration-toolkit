#pragma once

/** @file */

#include <fstream>

#include <opencv2/core.hpp>

#include "rt/filesystem.hpp"
#include "rt/types/ITKMesh.hpp"
#include "rt/types/UVMap.hpp"

namespace rt::io
{

/**
 * @class OBJReader
 * @author Zack Anderson, Seth Parker
 * @date 02/09/2017
 *
 * @brief Read an OBJ file into a ITKMesh
 *
 * Supports image mapped meshes. Image path is parsed from the OBJ's mtl
 * include. Other material properties are currently ignored. Throws
 * rt::IOException on error.
 *
 */
class OBJReader
{
public:
    /** @brief Set the OBJ file path */
    void setPath(const filesystem::path& p);

    /** @brief Read the mesh from file */
    auto read() -> ITKMesh::Pointer;

    /** @brief Return the parsed mesh */
    auto getMesh() -> ITKMesh::Pointer;

    /**
     * @brief Return parsed UV information
     *
     * If no UV information is read, returns an empty UVMap.
     * */
    auto getUVMap() -> UVMap;

    /**
     * @brief Return texture image as cv::Mat
     *
     * If no texture image was read from MTL file or if the file does not
     * exist, throws a rt::IOException.
     */
    auto getTextureMat() -> cv::Mat;

    /**
     * @brief Return the path to the texture image
     *
     * If no texture image was read from MTL file, returns an empty path.
     */
    auto getTexturePath() -> filesystem::path;

private:
    /**
     * 3-Tuple linking a vertex to its position, UV, and normal elements
     *
     * The return value for each element is the [index + 1] of a vertex's
     * position, UV coordinate, or surface normal inside the
     * OBJReader::vertices_, OBJReader::uvs_, or OBJReader::normals_ lists.
     *
     * VertexRefs { v, vt, vn }
     */
    using VertexRefs = cv::Vec<std::size_t, 3>;

    /** A list of at least three OBJReader::VertexRefs comprise a face */
    using Face = std::vector<VertexRefs>;

    /** Clear all temporary data structures */
    void reset_();

    /** Parse the mesh */
    void parse_();
    /** Handle parsed vertex lines */
    void parse_vertex_(const std::vector<std::string>& strs);
    /** Handle parsed vertex normal lines */
    void parse_normal_(const std::vector<std::string>& strs);
    /** Handle parsed vertex UV coordinate lines */
    void parse_tcoord_(const std::vector<std::string>& strs);
    /** Handle parsed face lines */
    void parse_face_(const std::vector<std::string>& strs);
    /** Handle parsed mtllib lines */
    void parse_mtllib_(const std::vector<std::string>& strs);
    /** Classify a OBJReader::VertexRefs as an OBJReader::RefType */

    /** Construct a mesh from the parsed information */
    void build_mesh_();

    /** Path to the OBJ file */
    filesystem::path path_;
    /** Path to the parsed texture image */
    filesystem::path texturePath_;
    /** Internal representation of mesh structure */
    ITKMesh::Pointer mesh_;
    /** Internal representation of UV Map */
    UVMap uvMap_;

    /** List of parsed vertex positions */
    std::vector<cv::Vec3d> vertices_;
    /** List of parsed vertex normals */
    std::vector<cv::Vec3d> normals_;
    /** List of parsed vertex UV coordinates */
    std::vector<cv::Vec2d> uvs_;
    /** List of parsed faces */
    std::vector<OBJReader::Face> faces_;
};

}  // namespace rt
