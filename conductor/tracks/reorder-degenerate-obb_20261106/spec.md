# Specification: ReorderUnorganizedTexture: invalid output size for degenerate/tiny meshes

**Track ID:** reorder-degenerate-obb_20261106
**Type:** Bug
**Created:** 2026-11-06
**Status:** Stub
**Source:** [#20](https://github.com/educelab/registration-toolkit/issues/20)

## Summary

The orthographic reorder path derives a ~INT_MAX output size from ill-conditioned `vtkOBBTree` axes on planar or tiny meshes, then throws OutOfMemoryError allocating it.

## Acceptance Criteria

- [ ] A tiny planar mesh (6 vertices, 4 triangles, z=0) in `ProjectionMode::Orthographic` either produces a correct texture or fails with a clear error, never an overflowed allocation
- [ ] Non-finite or out-of-bound OBB extents / output dimensions are detected
- [ ] `ReorderMultiTexture.*` tests can exercise the orthographic path

## Dependencies

TBD

## Out of Scope

TBD

## Technical Notes

Options in the issue: validate extents and fail clearly, replace/augment `vtkOBBTree` with a PCA-based OBB guarded for rank-deficient covariance, or guard `create_texture_` against infinite transforms. Interacts with `OrientationMode` (ADR 0003).

---

_Stub generated from GitHub issue #20. See the issue for full context; refine before planning._
