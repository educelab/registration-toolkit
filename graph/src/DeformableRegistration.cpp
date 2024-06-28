#include "rt/graph/DeformableRegistration.hpp"

#include <educelab/core/utils/Iteration.hpp>

using namespace educelab;
namespace rtg = rt::graph;
namespace fs = rt::filesystem;

using Meta = smgl::Metadata;

rtg::DeformableRegistrationNode::DeformableRegistrationNode()
    : Node{true}
    , fixedImage{&reg_, &DeformableRegistration::setFixedImage}
    , movingImage{&reg_, &DeformableRegistration::setMovingImage}
    , meshFillSize{&reg_, &DeformableRegistration::setMeshFillSize}
    , gradientTolerance{&reg_, &DeformableRegistration::setGradientMagnitudeTolerance}
    , iterations{&reg_, &DeformableRegistration::setNumberOfIterations}
    , reportMetrics{&reg_, &DeformableRegistration::setReportMetrics}
    , captureIntermediates{&reg_, &DeformableRegistration::setCaptureIntermediates}
    , transform{&tfm_}
    , intermediates{&intermediates_}
{
    registerInputPort("fixedImage", fixedImage);
    registerInputPort("movingImage", movingImage);
    registerInputPort("iterations", iterations);
    registerInputPort("meshFillSize", meshFillSize);
    registerInputPort("gradientTolerance", gradientTolerance);
    registerInputPort("reportMetrics", reportMetrics);
    registerInputPort("captureIntermediates", captureIntermediates);
    registerOutputPort("transform", transform);
    registerOutputPort("intermediates", intermediates);

    compute = [=]() {
        std::cout << "Running deformable registration..." << std::endl;
        tfm_ = reg_.compute();
        auto inters = reg_.getIntermediates();
        intermediates_.clear();
        intermediates_.reserve(inters.size());
        std::copy(
            inters.begin(), inters.end(), std::back_inserter(intermediates_));
    };
}

auto rtg::DeformableRegistrationNode::serialize_(
    const bool useCache, const fs::path& cacheDir) -> smgl::Metadata
{
    Meta m;
    m["iterations"] = reg_.getNumberOfIterations();
    m["meshFillSize"] = reg_.getMeshFillSize();
    m["gradientTolerance"] = reg_.getGradientMagnitudeTolerance();
    m["reportMetrics"] = reg_.getReportMetrics();
    m["captureIntermediates"] = reg_.getCaptureIntermediates();
    if (useCache and tfm_) {
        WriteTransform(cacheDir / "deformable.tfm", tfm_);
        m["transform"] = "deformable.tfm";
    }
    if (useCache and not intermediates_.empty()) {
        const auto interDir = cacheDir / "intermediates";
        fs::create_directory(interDir);
        for (auto [idx, tfm] : enumerate(intermediates_)) {
            WriteTransform(interDir / (std::to_string(idx) + ".tfm"), tfm);
        }
        m["intermediates"] =
            std::make_pair("intermediates", intermediates_.size());
    }
    return m;
}

void rtg::DeformableRegistrationNode::deserialize_(
    const Meta& meta, const fs::path& cacheDir)
{
    reg_.setNumberOfIterations(meta["iterations"].get<int>());
    reg_.setMeshFillSize(meta["meshFillSize"].get<unsigned>());
    reg_.setGradientMagnitudeTolerance(meta["gradientTolerance"].get<double>());
    reg_.setReportMetrics(meta["reportMetrics"].get<bool>());
    if (meta.contains("captureIntermediates")) {
        reg_.setCaptureIntermediates(meta["captureIntermediates"].get<bool>());
    }
    if (meta.contains("transform")) {
        const auto file = meta["transform"].get<std::string>();
        tfm_ = ReadTransform(cacheDir / file);
    }
    if (meta.contains("intermediates")) {
        const auto [dir, num] =
            meta["intermediates"].get<std::pair<std::string, std::size_t>>();
        intermediates_.clear();
        for (const auto i : range(num)) {
            auto path = cacheDir / dir / (std::to_string(i) + ".tfm)");
            intermediates_.emplace_back(ReadTransform(path));
        }
    }
}
