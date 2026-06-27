# Registration Toolkit

Tools for registering 2D images to textured 3D surfaces and reordering/retexturing
those surfaces. Registration operates on images; meshes carry the surface geometry
and its texture parameterization.

## Language

**Mesh**:
A triangulated 3D surface. Canonically an `educelab::Mesh3d` throughout the toolkit;
converted to other representations (e.g. VTK polydata) only at the boundary of
algorithms that require them.

**UV Map**:
A per-corner (per-wedge) mapping from a mesh face's vertices into 2D texture
parameter space. A single 3D vertex may carry different UV coordinates in adjacent
faces (a UV seam).

**UV origin**:
The corner of texture space that a UV coordinate `(0,0)` refers to. In-memory UV
coordinates are always **top-left** origin: `v` increases downward, matching image
row order so a UV maps directly to a pixel. File formats that store coordinates
bottom-left (e.g. OBJ `vt`) are converted at the I/O boundary.
_Avoid_: storage origin, anchor

**Chart**:
A contiguous region of a UV Map backed by a single texture. A multi-chart mesh
partitions its faces across several textures.

**Aspect**:
The width-to-height ratio of the output texture space a UV Map is intended to fill.
Travels with the UV Map and is the fallback a rasterizer uses to size a generated
image when no source texture is present to dictate dimensions.
