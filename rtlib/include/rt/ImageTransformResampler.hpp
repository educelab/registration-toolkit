#pragma once

/** @file */

#include <itkCompositeTransform.h>
#include <opencv2/core.hpp>

namespace rt
{

/**
 * @brief Resample a moving image using a pre-generated transform. Output image
 * is of size s.
 *
 */
cv::Mat ImageTransformResampler(
    const cv::Mat& m,
    const cv::Size& s,
    const itk::CompositeTransform<double, 2>::Pointer& transform);
}