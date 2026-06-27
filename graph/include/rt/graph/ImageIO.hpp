#pragma once

/** @file */

#include <filesystem>

#include <opencv2/core.hpp>
#include <smgl/Node.hpp>
#include <smgl/Ports.hpp>

namespace rt::graph
{

/**
 * @brief Image File Reader
 * @see ReadImage
 */
class ReadImageNode : public smgl::Node
{
public:
    /** Default constructor */
    ReadImageNode();

    /** @name Input Ports */
    /**@{*/
    /** @brief Image path port */
    smgl::InputPort<std::filesystem::path> path{&path_};
    /**@}*/

    /** @name Output Ports */
    /**@{*/
    /** @brief Read image port */
    smgl::OutputPort<cv::Mat> image{&img_};
    /**@}*/

private:
    /** File path */
    std::filesystem::path path_;
    /** Loaded image */
    cv::Mat img_;
    /** Graph serialize */
    smgl::Metadata serialize_(
        bool /*unused*/, const std::filesystem::path& /*unused*/) override;
    /** Graph deserialize */
    void deserialize_(
        const smgl::Metadata& meta,
        const std::filesystem::path& /*unused*/) override;
};

/**
 * @brief Image File Writer
 * @see WriteImage
 */
class WriteImageNode : public smgl::Node
{
public:
    /** Default constructor */
    WriteImageNode();

    /** @name Input Ports */
    /**@{*/
    /** @brief Image path port */
    smgl::InputPort<std::filesystem::path> path{&path_};
    /** @brief Image port */
    smgl::InputPort<cv::Mat> image{&img_};
    /**@}*/

private:
    /** File path */
    std::filesystem::path path_;
    /** Image to write */
    cv::Mat img_;
    /** Graph serialize */
    smgl::Metadata serialize_(
        bool /*unused*/, const std::filesystem::path& /*unused*/) override;
    /** Graph deserialize */
    void deserialize_(
        const smgl::Metadata& meta,
        const std::filesystem::path& /*unused*/) override;
};

/**
 * @brief Image Series File Writer
 *
 * List WriteImageNode, but writes a series of images
 */
class WriteImageSeriesNode : public smgl::Node
{
public:
    /** List of images type */
    using ImageList = std::vector<cv::Mat>;

    /** Default constructor */
    WriteImageSeriesNode();

    /** @name Input Ports */
    /**@{*/
    /** @brief Image path port */
    smgl::InputPort<std::filesystem::path> path{&path_};
    /** @brief Image port */
    smgl::InputPort<ImageList> images{&imgs_};
    /**@}*/

private:
    /** File path */
    std::filesystem::path path_;
    /** Image to write */
    ImageList imgs_;
    /** Graph serialize */
    smgl::Metadata serialize_(
        bool /*unused*/, const std::filesystem::path& /*unused*/) override;
    /** Graph deserialize */
    void deserialize_(
        const smgl::Metadata& meta,
        const std::filesystem::path& /*unused*/) override;
};

}  // namespace rt::graph