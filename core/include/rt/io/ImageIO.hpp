#pragma once

/** @file */

#include <filesystem>

#include <opencv2/core.hpp>

namespace rt
{

/**
 * @brief Read an image from the specified path
 *
 * Currently just a wrapper around:
 *
 * @code
 * cv::imread(path.string(), cv::IMREAD_UNCHANGED);
 * @endcode
 */
auto ReadImage(const std::filesystem::path& path) -> cv::Mat;

/**
 * @brief Write image to the specified path
 *
 * Use rt::WriteTIFF for all tiff images, which includes support for
 * transparency and floating-point images. Otherwise, uses cv::imwrite.
 */
void WriteImage(const std::filesystem::path& path, const cv::Mat& img);

}  // namespace rt