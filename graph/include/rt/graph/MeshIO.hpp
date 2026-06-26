#pragma once

/** @file */

#include <opencv2/core.hpp>
#include <smgl/Node.hpp>
#include <smgl/Ports.hpp>

#include "rt/filesystem.hpp"
#include "rt/types/Mesh.hpp"
#include "rt/types/UVMap.hpp"

namespace rt::graph
{

/**
 * @brief Mesh File Reader
 * @see rt::io::ReadMesh
 */
class MeshReadNode : public smgl::Node
{
public:
    /** Default constructor */
    MeshReadNode();

    /** @name Input Ports */
    /**@{*/
    /** @brief Mesh path port */
    smgl::InputPort<filesystem::path> path{&path_};
    /**@}*/

    /** @name Output Ports */
    /**@{*/
    /** @brief Loaded mesh port */
    smgl::OutputPort<Mesh::Pointer> mesh{&mesh_};
    /** @brief Loaded image port */
    smgl::OutputPort<cv::Mat> image{&img_};
    /** @brief Loaded image path port */
    smgl::OutputPort<filesystem::path> imagePath{&imgPath_};
    /** @brief Load UV Map port */
    smgl::OutputPort<UVMap> uvMap{&uv_};
    /**@}*/

private:
    /** File path */
    filesystem::path path_;
    /** Loaded mesh */
    Mesh::Pointer mesh_;
    /** Loaded image */
    cv::Mat img_;
    /** Loaded image path */
    filesystem::path imgPath_;
    /** Loaded UV map */
    UVMap uv_;
    /** Graph serialize */
    smgl::Metadata serialize_(
        bool /*unused*/, const filesystem::path& /*unused*/) override;
    /** Graph deserialize */
    void deserialize_(
        const smgl::Metadata& meta,
        const filesystem::path& /*unused*/) override;
};

/**
 * @brief Mesh File Writer
 * @see rt::io::WriteMesh
 */
class MeshWriteNode : public smgl::Node
{
public:
    /** Default constructor */
    MeshWriteNode();

    /** @name Input Ports */
    /**@{*/
    /** @brief Mesh path port */
    smgl::InputPort<filesystem::path> path{&path_};
    /** @brief Mesh port */
    smgl::InputPort<Mesh::Pointer> mesh{&mesh_};
    /** @brief Texture image port */
    smgl::InputPort<cv::Mat> image{&img_};
    /** @brief Texture image source path port */
    smgl::InputPort<filesystem::path> imageSource{&imgSource_};
    /** @brief UVMap port */
    smgl::InputPort<UVMap> uvMap{&uv_};
    /**@}*/

private:
    /** File path */
    filesystem::path path_;
    /** Mesh to write */
    Mesh::Pointer mesh_;
    /** Texture image */
    cv::Mat img_;
    /** Texture image source path */
    filesystem::path imgSource_;
    /** UV map */
    UVMap uv_;
    /** Graph serialize */
    smgl::Metadata serialize_(
        bool /*unused*/, const filesystem::path& /*unused*/) override;
    /** Graph deserialize */
    void deserialize_(
        const smgl::Metadata& meta,
        const filesystem::path& /*unused*/) override;
};

}  // namespace rt