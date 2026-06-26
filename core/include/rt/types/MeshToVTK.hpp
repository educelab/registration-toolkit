#pragma once

/** @file */

#include <vtkPolyData.h>
#include <vtkSmartPointer.h>

#include "rt/types/Mesh.hpp"

namespace rt
{
/**
 * @brief Convert an rt::Mesh to VTK PolyData
 *
 * Copies vertices, vertex normals (when present), and faces (as cells) from
 * @p input to @p output. Replaces the ITK side of the former ITK2VTK bridge;
 * algorithms that compute in VTK (e.g. ReorderUnorganizedTexture) convert the
 * canonical mesh at their boundary.
 */
void MeshToVTK(const Mesh& input, vtkSmartPointer<vtkPolyData>& output);

/** @copydoc MeshToVTK */
auto MeshToVTK(const Mesh& input) -> vtkSmartPointer<vtkPolyData>;
}  // namespace rt
