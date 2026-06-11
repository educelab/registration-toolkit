#include "rt/util/CalculateNormals.hpp"

#include <cstdint>

using namespace rt;

CalculateNormals::CalculateNormals(const ITKMesh::Pointer& mesh)
    : input_{mesh}, output_{ITKMesh::New()}
{
    DeepCopy(input_, output_);
}

///// Input/Output /////
void CalculateNormals::setMesh(const ITKMesh::Pointer& mesh) { input_ = mesh; }

///// Processing /////
auto CalculateNormals::compute() -> ITKMesh::Pointer
{
    output_ = ITKMesh::New();
    DeepCopy(input_, output_);

    compute_normals_();
    assign_to_mesh_();

    return output_;
}

void CalculateNormals::compute_normals_()
{
    vertexNormals_.assign(output_->GetNumberOfPoints(), cv::Vec3d::all(0));

    for (auto cellIt = input_->GetCells()->Begin();
         cellIt != input_->GetCells()->End(); ++cellIt) {

        // Only triangles contribute a face normal
        if (cellIt->Value()->GetNumberOfPoints() != 3) {
            continue;
        }

        // Collect the point id's for this cell
        std::array<std::uint64_t, 3> pointIds{};
        std::size_t i{0};
        for (auto p = cellIt->Value()->PointIdsBegin();
             p != cellIt->Value()->PointIdsEnd(); ++p) {
            pointIds[i++] = *p;
        }

        cv::Vec3d v[3];
        for (std::size_t k = 0; k < 3; ++k) {
            const auto vert = input_->GetPoint(pointIds[k]);
            v[k] = {vert[0], vert[1], vert[2]};
        }

        // Cross product magnitude is 2*area, so the un-normalized result
        // gives area-weighted vertex normals when summed.
        const cv::Vec3d faceNormal = (v[1] - v[0]).cross(v[2] - v[0]);

        for (const auto id : pointIds) {
            vertexNormals_[id] += faceNormal;
        }
    }
}

void CalculateNormals::assign_to_mesh_()
{
    for (auto point = input_->GetPoints()->Begin();
         point != input_->GetPoints()->End(); ++point) {
        cv::Vec3d norm = vertexNormals_[point.Index()];
        const auto mag = cv::norm(norm);
        if (mag > 0.0) {
            norm /= mag;
        }
        output_->SetPointData(point.Index(), norm.val);
    }
}
