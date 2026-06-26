---
status: accepted
---

# educelab::Mesh3d is the canonical mesh type; ITKMesh is removed

The toolkit historically used `itk::Mesh` (`rt::ITKMesh`) as its in-memory mesh
everywhere meshes appear. We adopt `educelab::Mesh3d` as the single canonical mesh
type and **delete** `ITKMesh.hpp`, `rt::DeepCopy`, and the ITK side of `ITK2VTK`.
This was driven by the finding that the registration pipeline operates purely on
ITK *images* — no registration code consumes a mesh — so `ITKMesh` only ever served
mesh I/O and a handful of geometry utilities. Adopting libcore's mesh + multi-format
I/O (`read_mesh`/`write_mesh`, OBJ and PLY) makes `educelab::Mesh3d` the natural
interchange type, and unifies on libcore rather than maintaining a parallel ITK mesh
representation.

## Considered Options

- **Full migration to educelab::Mesh3d (chosen).** Possible precisely because
  registration is image-only; the only mesh consumers were I/O, normals, and
  `ReorderUnorganizedTexture`.
- **Keep ITKMesh as an internal VTK adapter.** Rejected — it retained an ITK mesh
  dependency for no compute benefit, since `ReorderUnorganizedTexture` immediately
  converts to `vtkPolyData` anyway and `ITK2VTK` had no other caller.
- **Don't adopt libcore mesh I/O.** Rejected — would forgo multi-format support and
  keep a hand-rolled OBJ parser.

## Consequences

- `ReorderUnorganizedTexture` gets a new `Mesh3d→vtkPolyData` bridge (replacing the
  ITK side of `ITK2VTK`); it still computes in VTK as before.
- `CalculateNormals` is deleted in favor of `educelab::vertex_normal()`, which is
  angle-weighted rather than the previous area-weighted scheme — texture output of
  `RetextureMesh` shifts slightly at irregular triangulations.
- This is a **breaking API change** (public interfaces no longer expose `ITKMesh`),
  and ships as a **major version bump**.
