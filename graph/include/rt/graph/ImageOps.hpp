#pragma once

/** @file */

#include <opencv2/core.hpp>
#include <smgl/Node.hpp>
#include <smgl/Ports.hpp>

#include "rt/filesystem.hpp"

namespace rt::graph
{

/** @copybrief rt::ColorConvertImage */
class ColorConvertNode : public smgl::Node
{
public:
    /** Default constructor */
    ColorConvertNode();

    /** @brief Input image */
    smgl::InputPort<cv::Mat> imageIn;
    /** @brief Number of channels in the output image */
    smgl::InputPort<std::size_t> channels;
    /** @brief Output image */
    smgl::OutputPort<cv::Mat> imageOut;

private:
    /** Input image */
    cv::Mat input_;
    /** Channels */
    std::size_t cns_{3};
    /** Output image */
    cv::Mat output_;

    /** Graph serialize */
    auto serialize_(bool useCache, const filesystem::path& cacheDir)
        -> smgl::Metadata override;
    /** Graph deserialize */
    void deserialize_(
        const smgl::Metadata& meta, const filesystem::path& cacheDir) override;
};

/**
 * @brief Adjust the per-axis value range of a 3D position map
 *
 * Transforms a per-pixel XYZ position map (CV_32FC3) by independently adjusting
 * each axis over its finite (intersected) pixels. NaN pixels (no surface
 * intersection) are always preserved.
 */
class PositionMapTransformNode : public smgl::Node
{
public:
    /** @brief Per-axis value range adjustment */
    enum class Mode {
        Raw,        /** Pass the surface coordinates through unchanged */
        Shifted,    /** Subtract each axis's minimum so values lie in
                       [0, extent), preserving the surface's scale */
        Normalized  /** Rescale each axis by its min/max into [0, 1] */
    };

    /** Default constructor */
    PositionMapTransformNode();

    /** @brief Input position map (CV_32FC3) */
    smgl::InputPort<cv::Mat> imageIn;
    /** @brief Per-axis value range adjustment */
    smgl::InputPort<Mode> mode;
    /** @brief Output position map */
    smgl::OutputPort<cv::Mat> imageOut;

private:
    /** Input position map */
    cv::Mat input_;
    /** Value range adjustment */
    Mode mode_{Mode::Shifted};
    /** Output position map */
    cv::Mat output_;

    /** Graph serialize */
    auto serialize_(bool useCache, const filesystem::path& cacheDir)
        -> smgl::Metadata override;
    /** Graph deserialize */
    void deserialize_(
        const smgl::Metadata& meta, const filesystem::path& cacheDir) override;
};

}  // namespace rt::graph