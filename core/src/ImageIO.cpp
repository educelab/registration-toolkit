#include "rt/io/ImageIO.hpp"

#include <iostream>

#include <opencv2/imgcodecs.hpp>

#include "rt/Logging.hpp"
#include "rt/io/FileExtensionFilter.hpp"
#include "rt/io/TIFFIO.hpp"
#include "rt/util/ImageConversion.hpp"

namespace fs = rt::filesystem;

static const auto IsFormat = rt::FileExtensionFilter;

auto rt::ReadImage(const fs::path& path) -> cv::Mat
{
    // Attempt to read with OpenCV
    rt::logger()->debug("Loading image: {}", path.string());
    auto img = cv::imread(path.string(), cv::IMREAD_UNCHANGED);

    // If OpenCV failed and is a TIFF, try our reader
    if (img.empty() and IsFormat(path, {"tif", "tiff"})) {
        rt::logger()->debug("Falling back to rt::ReadRawTIFF");
        img = io::ReadRawTIFF(path);
    }

    if (img.empty()) {
        rt::logger()->warn("Loaded image is empty: {}", path.string());
    } else {
        rt::logger()->debug("Loaded image: {}x{}", img.cols, img.rows);
    }
    return img;
}

void rt::WriteImage(const fs::path& path, const cv::Mat& img)
{
    // Do nothing on empty images
    if (img.empty()) {
        rt::logger()->warn("Not writing empty image: {}", path.string());
        return;
    }

    // Use our TIFF writer
    if (IsFormat(path, {"tif", "tiff"})) {
        rt::io::WriteTIFF(path, img);
    } else {
        cv::Mat output = img.clone();
        if (img.depth() == CV_32F or img.depth() == CV_64F) {
            rt::logger()->warn(
                "Image is floating-point but format {} "
                "does not support floating-point images. "
                "Results may be incorrect.",
                path.extension().string());
        }

        if (img.channels() == 4 and IsFormat(path, {"jpg", "jpeg"})) {
            rt::logger()->warn(
                "Image is 4-channel (RGBA) but format {} "
                "does not support 4-channels. Extra channel "
                "will be removed.",
                path.extension().string());
            output = rt::ColorConvertImage(img, 3);
        } else if (img.channels() == 2) {
            rt::logger()->warn(
                "Image is 2-channel (Gray + Alpha) but "
                "format {} does not support 2-channels. Extra "
                "channel will be removed.",
                path.extension().string());
            output = rt::ColorConvertImage(img, 1);
        }

        cv::imwrite(path.string(), output);
    }
}