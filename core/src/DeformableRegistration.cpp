#include "rt/DeformableRegistration.hpp"

#include <itkCommand.h>
#include <itkImageRegistrationMethod.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkMattesMutualInformationImageToImageMetric.h>
#include <itkRegularStepGradientDescentOptimizer.h>

#include "rt/ITKImageTypes.hpp"
#include "rt/util/ITKOpenCVBridge.hpp"

using namespace rt;

using GrayInterpolator =
    itk::LinearInterpolateImageFunction<Image8UC1, double>;
using Metric =
    itk::MattesMutualInformationImageToImageMetric<Image8UC1, Image8UC1>;
using Optimizer = itk::RegularStepGradientDescentOptimizer;
using Registration = itk::ImageRegistrationMethod<Image8UC1, Image8UC1>;
using BSplineParameters = DeformableRegistration::Transform::ParametersType;
using Transform = DeformableRegistration::Transform;

namespace
{
constexpr double DEFAULT_MAX_STEP_FACTOR = 1.0 / 500.0;
constexpr double DEFAULT_MIN_STEP_FACTOR = 1.0 / 500000.0;

/* The metric requires two parameters to be selected: the number
of bins used to compute the entropy and the number of spatial samples
used to compute the density estimates. In typical application, 50
histogram bins are sufficient and the metric is relatively insensitive
to changes in the number of bins. The number of spatial samples
to be used depends on the content of the image. If the images are
smooth and do not contain much detail, then using approximately
1 percent of the pixels will do. On the other hand, if the images
are detailed, it may be necessary to use a much higher proportion,
such as 20 percent. */
constexpr std::size_t DEFAULT_HISTOGRAM_BINS = 50;
constexpr double DEFAULT_SAMPLE_FACTOR = 1.0 / 80.0;

// Callback for printing iteration metrics to the command line
class ReportMetricCallback final : public itk::Command
{
protected:
    ReportMetricCallback() = default;

public:
    using Pointer = itk::SmartPointer<ReportMetricCallback>;
    using Optimizer = itk::RegularStepGradientDescentOptimizer;

    static auto New() -> Pointer
    {
        Pointer smartPtr = itk::ObjectFactory<ReportMetricCallback>::Create();
        if (smartPtr == nullptr) {
            smartPtr = new ReportMetricCallback;
        }
        smartPtr->UnRegister();
        return smartPtr;
    }

    void Execute(Object* caller, const itk::EventObject& event) override
    {
        Execute(reinterpret_cast<const Object*>(caller), event);
    }

    void Execute(const Object* object, const itk::EventObject& event) override
    {
        const auto* optimizer = dynamic_cast<const Optimizer*>(object);
        if (not itk::IterationEvent().CheckEvent(&event)) {
            return;
        }
        std::cerr << optimizer->GetValue() << "\n";
    }
};

// Callback for printing iteration metrics to the command line
class SaveTransformCallback final : public itk::Command
{
protected:
    SaveTransformCallback() = default;

public:
    using Pointer = itk::SmartPointer<SaveTransformCallback>;
    using Optimizer = itk::RegularStepGradientDescentOptimizer;

    Transform::Pointer prototype;
    std::vector<Transform::Pointer> tfms;

    static auto New(const Transform::Pointer& prototype) -> Pointer
    {
        Pointer smartPtr = itk::ObjectFactory<SaveTransformCallback>::Create();
        if (smartPtr == nullptr) {
            smartPtr = new SaveTransformCallback;
        }
        smartPtr->UnRegister();
        smartPtr->prototype = prototype->Clone();
        return smartPtr;
    }

    void Execute(Object* caller, const itk::EventObject& event) override
    {
        Execute(reinterpret_cast<const Object*>(caller), event);
    }

    void Execute(const Object* object, const itk::EventObject& event) override
    {
        const auto* optimizer = dynamic_cast<const Optimizer*>(object);
        if (not itk::IterationEvent().CheckEvent(&event)) {
            return;
        }
        auto tfm = prototype->Clone();
        tfm->SetParameters(optimizer->GetCurrentPosition());
        tfms.emplace_back(tfm);
    }
};
}  // namespace

void DeformableRegistration::setFixedImage(const cv::Mat& i)
{
    fixedImage_ = i;
}

void DeformableRegistration::setMovingImage(const cv::Mat& i)
{
    movingImage_ = i;
}

void DeformableRegistration::setNumberOfIterations(std::size_t i)
{
    iterations_ = i;
}

auto DeformableRegistration::getNumberOfIterations() const -> std::size_t
{
    return iterations_;
}

auto DeformableRegistration::getTransform() -> Transform::Pointer
{
    return output_;
}

void DeformableRegistration::setMeshFillSize(std::uint32_t i)
{
    meshFillSize_ = i;
}

auto DeformableRegistration::getMeshFillSize() const -> std::uint32_t
{
    return meshFillSize_;
}

void DeformableRegistration::setGradientMagnitudeTolerance(double i)
{
    gradMagTol_ = i;
}

auto DeformableRegistration::getGradientMagnitudeTolerance() const -> double
{
    return gradMagTol_;
}

void DeformableRegistration::setReportMetrics(const bool i)
{
    reportMetrics_ = i;
}

auto DeformableRegistration::getReportMetrics() const -> bool
{
    return reportMetrics_;
}

void DeformableRegistration::setCaptureIntermediates(const bool i)
{
    captureIntermediates_ = i;
}

auto DeformableRegistration::getCaptureIntermediates() const -> bool
{
    return captureIntermediates_;
}

auto DeformableRegistration::getIntermediates() const
    -> std::vector<Transform::Pointer>
{
    return intermediates_;
}

auto DeformableRegistration::compute() -> Transform::Pointer
{
    ///// Create grayscale images /////
    const auto fixed8u = QuantizeImage(fixedImage_, CV_8U);
    const auto fixed = CVMatToITKImage<Image8UC1>(fixed8u);
    const auto moving8u = QuantizeImage(movingImage_, CV_8U);
    const auto moving = CVMatToITKImage<Image8UC1>(moving8u);

    ///// Setup the BSpline Transform /////
    output_ = Transform::New();
    Transform::PhysicalDimensionsType fixedPhysicalDims;
    Transform::MeshSizeType meshSize;
    Transform::OriginType fixedOrigin;

    for (auto i = 0; i < 2; i++) {
        fixedOrigin[i] = fixed->GetOrigin()[i];
        fixedPhysicalDims[i] =
            fixed->GetSpacing()[i] *
            static_cast<double>(
                fixed->GetLargestPossibleRegion().GetSize()[i] - 1);
    }
    meshSize.Fill(meshFillSize_);

    output_->SetTransformDomainOrigin(fixedOrigin);
    output_->SetTransformDomainPhysicalDimensions(fixedPhysicalDims);
    output_->SetTransformDomainMeshSize(meshSize);
    output_->SetTransformDomainDirection(fixed->GetDirection());

    const auto numParams = output_->GetNumberOfParameters();
    BSplineParameters parameters(numParams);
    parameters.Fill(0.0);
    output_->SetParameters(parameters);

    ///// Setup Registration and Metrics /////
    const auto metric = Metric::New();
    const auto optimizer = Optimizer::New();
    const auto registration = Registration::New();
    const auto grayInterpolator = GrayInterpolator::New();
    if (reportMetrics_) {
        optimizer->AddObserver(
            itk::IterationEvent(), ReportMetricCallback::New());
    }
    itk::SmartPointer<SaveTransformCallback> captureTfms;
    intermediates_.clear();
    if (captureIntermediates_) {
        captureTfms = SaveTransformCallback::New(output_);
        optimizer->AddObserver(itk::IterationEvent(), captureTfms);
    }

    registration->SetFixedImage(fixed);
    registration->SetMovingImage(moving);
    registration->SetMetric(metric);
    registration->SetOptimizer(optimizer);
    registration->SetInterpolator(grayInterpolator);
    registration->SetTransform(output_);
    registration->SetInitialTransformParameters(output_->GetParameters());

    const auto fixedRegion = fixed->GetBufferedRegion();
    registration->SetFixedImageRegion(fixedRegion);

    metric->SetNumberOfHistogramBins(DEFAULT_HISTOGRAM_BINS);
    const auto numSamples = static_cast<std::size_t>(
        static_cast<double>(fixedRegion.GetNumberOfPixels()) *
        DEFAULT_SAMPLE_FACTOR);
    metric->SetNumberOfSpatialSamples(numSamples);

    ///// Setup Optimizer /////
    const auto regionWidth =
        static_cast<double>(fixed->GetLargestPossibleRegion().GetSize()[0]);
    const auto maxStepLength = regionWidth * DEFAULT_MAX_STEP_FACTOR;
    const auto minStepLength = regionWidth * DEFAULT_MIN_STEP_FACTOR;

    optimizer->MinimizeOn();
    optimizer->SetMaximumStepLength(maxStepLength);
    optimizer->SetMinimumStepLength(minStepLength);
    optimizer->SetRelaxationFactor(relaxationFactor_);
    optimizer->SetNumberOfIterations(iterations_);
    optimizer->SetGradientMagnitudeTolerance(gradMagTol_);

    ///// Run Registration /////
    registration->Update();

    // Report final values as requested
    if (reportMetrics_) {
        std::cerr << "Stop Condition: ";
        std::cerr << optimizer->GetStopConditionDescription() << "\n";
        std::cerr << "Final Metric Value: " << optimizer->GetValue() << "\n";
    }
    if (captureIntermediates_) {
        intermediates_ = captureTfms->tfms;
    }

    output_->SetParameters(registration->GetLastTransformParameters());
    return output_;
}