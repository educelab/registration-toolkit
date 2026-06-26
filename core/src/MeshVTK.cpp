#include "rt/types/MeshVTK.hpp"

#include <array>

#include <vtkCellArray.h>
#include <vtkDoubleArray.h>
#include <vtkIdList.h>
#include <vtkPointData.h>
#include <vtkPoints.h>

using namespace rt;

void rt::MeshToVTK(const Mesh& input, vtkSmartPointer<vtkPolyData>& output)
{
    // points + normals
    const auto points = vtkSmartPointer<vtkPoints>::New();
    const auto pointNormals = vtkSmartPointer<vtkDoubleArray>::New();
    pointNormals->SetNumberOfComponents(3);  // 3d normals (i.e. x,y,z)

    for (std::size_t vid = 0; vid < input.num_vertices(); ++vid) {
        const auto& v = input.vertex(vid);
        points->InsertPoint(
            static_cast<vtkIdType>(vid), v[0], v[1], v[2]);

        if (v.normal.has_value()) {
            const auto& n = *v.normal;
            std::array<double, 3> ptNorm = {n[0], n[1], n[2]};
            pointNormals->InsertTuple(
                static_cast<vtkIdType>(vid), ptNorm.data());
        }
    }

    // cells
    const auto polys = vtkSmartPointer<vtkCellArray>::New();
    for (std::size_t fid = 0; fid < input.num_faces(); ++fid) {
        const auto& face = input.face(fid);
        auto poly = vtkSmartPointer<vtkIdList>::New();
        for (const auto vid : face) {
            poly->InsertNextId(static_cast<vtkIdType>(vid));
        }
        polys->InsertNextCell(poly);
    }

    // assign to the mesh
    output->SetPoints(points);
    output->SetPolys(polys);
    if (pointNormals->GetNumberOfTuples() > 0) {
        output->GetPointData()->SetNormals(pointNormals);
    }
}

auto rt::MeshToVTK(const Mesh& input) -> vtkSmartPointer<vtkPolyData>
{
    auto output = vtkSmartPointer<vtkPolyData>::New();
    MeshToVTK(input, output);
    return output;
}
