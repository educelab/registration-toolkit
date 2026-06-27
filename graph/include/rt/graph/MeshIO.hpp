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
 * @see rt::ReadMesh
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
 * @see rt::WriteMesh
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
    /**
     * @brief Texture image port
     *
     * Mutually exclusive with imageSource: whichever of the two received the
     * most recent update is the one used when writing (smgl ports retain their
     * last value and cannot be cleared, so recency — not a fixed precedence —
     * decides). Both setters are wired up in the constructor.
     */
    smgl::InputPort<cv::Mat> image;
    /** @brief Texture image source path port (see image) */
    smgl::InputPort<filesystem::path> imageSource;
    /** @brief UVMap port */
    smgl::InputPort<UVMap> uvMap{&uv_};
    /**@}*/

private:
    /** Which texture input was most recently set */
    enum class TextureInput { None, Image, Source };
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
    /** Most recently updated texture input (image vs. source path) */
    TextureInput lastTexture_{TextureInput::None};
    /** Graph serialize */
    smgl::Metadata serialize_(
        bool useCache, const filesystem::path& cacheDir) override;
    /** Graph deserialize */
    void deserialize_(
        const smgl::Metadata& meta,
        const filesystem::path& cacheDir) override;
};

}  // namespace rt