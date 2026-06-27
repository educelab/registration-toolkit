#pragma once

/** @file */

#include <educelab/core/types/Mesh.hpp>

namespace rt
{

/** @brief Per-vertex traits for the canonical mesh: optional vertex normals */
struct MeshVertexTraits : educelab::traits::WithNormal<double, 3> {
};

/**
 * @brief Canonical toolkit mesh type
 *
 * A 3D, double-precision `educelab::Mesh` carrying optional per-vertex normals
 * (libcore's @ref educelab::traits::WithNormal). This is the single in-memory
 * mesh representation throughout the toolkit; it is converted to other
 * representations (e.g. VTK polydata via rt::MeshToVTK) only at the boundary of
 * algorithms that require them. See
 * docs/adr/0001-educelab-mesh-as-canonical-mesh-type.md.
 */
using Mesh = educelab::Mesh<double, 3, MeshVertexTraits>;

/** @brief Shared pointer to the canonical mesh type */
using MeshPointer = Mesh::Pointer;

}  // namespace rt
