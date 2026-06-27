#pragma once

/** @file */

#include <filesystem>

#include <opencv2/core.hpp>
#include <smgl/Node.hpp>
#include <smgl/Ports.hpp>

#include "rt/DeformableRegistration.hpp"
#include "rt/types/Transforms.hpp"

namespace rt::graph
{

/**
 * @brief B-Spline Deformable Registration
 * @see DeformableRegistration
 */
class DeformableRegistrationNode : public smgl::Node
{
public:
    /** Default constructor */
    DeformableRegistrationNode();

    /** @name Input Ports */
    /**@{*/
    /** @brief Fixed image */
    smgl::InputPort<cv::Mat> fixedImage;
    /** @brief Moving image */
    smgl::InputPort<cv::Mat> movingImage;
    /** @brief Mesh Fill Size */
    smgl::InputPort<unsigned> meshFillSize;
    /** @brief Gradient magnitude tolerance */
    smgl::InputPort<double> gradientTolerance;
    /** @brief Deformable iterations */
    smgl::InputPort<int> iterations;
    /** @copydoc DeformableRegistration::setReportMetrics(bool) */
    smgl::InputPort<bool> reportMetrics;
    /** @copydoc DeformableRegistration::setCaptureIntermediates(bool) */
    smgl::InputPort<bool> captureIntermediates;
    /**@}*/

    /** @name Output Ports */
    /**@{*/
    /** @brief Final transform port */
    smgl::OutputPort<Transform::Pointer> transform;
    /** @copydoc DeformableRegistration::getIntermediates() */
    smgl::OutputPort<std::vector<Transform::Pointer>> intermediates;
    /**@}*/

private:
    /** Registration method */
    DeformableRegistration reg_;
    /** Final transform */
    Transform::Pointer tfm_;
    /** Intermediate transforms */
    std::vector<Transform::Pointer> intermediates_;
    /** Graph serialize */
    auto serialize_(bool useCache, const std::filesystem::path& cacheDir)
        -> smgl::Metadata override;
    /** Graph deserialize */
    void deserialize_(
        const smgl::Metadata& meta, const std::filesystem::path& cacheDir) override;
};

}  // namespace rt