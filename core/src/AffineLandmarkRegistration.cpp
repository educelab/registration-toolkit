#include "rt/AffineLandmarkRegistration.hpp"

#include <itkLandmarkBasedTransformInitializer.h>

#include "rt/ITKImageTypes.hpp"

using namespace rt;

void AffineLandmarkRegistration::setReportMetrics(bool i)
{
    reportMetrics_ = i;
}

bool AffineLandmarkRegistration::getReportMetrics() { return reportMetrics_; }

auto AffineLandmarkRegistration::compute()
    -> AffineLandmarkRegistration::Transform::Pointer
{
    using TransformInitializer =
        itk::LandmarkBasedTransformInitializer<Transform, Image8UC3, Image8UC3>;

    // Setup new transform
    output_ = Transform::New();

    // Initialize transform
    auto landmarkTransformInit = TransformInitializer::New();
    landmarkTransformInit->SetFixedLandmarks(fixedLdmks_);
    landmarkTransformInit->SetMovingLandmarks(movingLdmks_);
    output_->SetIdentity();
    landmarkTransformInit->SetTransform(output_);
    landmarkTransformInit->InitializeTransform();

    if (reportMetrics_) {
        std::cout << "Affine Metric: " << output_->Metric() << std::endl;
    }

    return output_;
}

auto AffineLandmarkRegistration::getTransform()
    -> AffineLandmarkRegistration::Transform::Pointer
{
    return output_;
}