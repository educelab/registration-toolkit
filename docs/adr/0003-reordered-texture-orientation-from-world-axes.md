---
status: accepted
---

# Reordered texture orientation is resolved against the world axes, opt-in

`ReorderUnorganizedTexture` builds its sampling frame from a `vtkOBBTree` oriented
bounding box. That box's axes are ordered by extent and carry **arbitrary signs**
(`vtkOBBTree` also makes no handedness guarantee), so the orientation of the
reordered image is unrelated to the orientation of the mesh handed to it. An
upstream step that has already put the mesh in a canonical frame — mesh right along
+X, mesh up along +Y, surface normal along +Z, as `pgs-sfm-orient` produces — has
that work discarded here.

The practical cost is a **mirrored deliverable**: on a real scan the reordered image
came out horizontally flipped, so the label text and the papyrus itself read
backwards. A mirror is not a cosmetic problem — it cannot be undone by rotating the
image, and it silently breaks frame-to-frame comparability, which the downstream
`acquisition-workflow` registration pipeline depends on.

We add `ReorderUnorganizedTexture::OrientationMode`, plumbed through
`ReorderTextureNode` and exposed as `rt_reorder_texture --orientation`:

- **`OBB`** (default) — take the box's axes as computed. Existing behavior, except
  for the reflection fix described below.
- **`Canonical`** — resolve the box's axis directions against the world axes.

`Canonical` asserts a precondition about the *input*, not a preference about the
output: it is only meaningful if the mesh is already canonically oriented. On an
arbitrarily posed mesh it trades one arbitrary orientation for another. The default
therefore stays `OBB`, and callers opt in.

## How the signs are derived

Working backwards through the conventions already in `create_texture_()`: it rotates
OBB x → +X and OBB y → −Y, then samples with `u` along −X and `v` along +Y, tracing
rays from `z_max` toward −Z (the default `!useFirstIntersection_`). In terms of the
*input* axes that means `u = -obb.x`, `v = -obb.y`, and the sampled face is the one
pointing along `-obb.z`.

A canonical image wants `u = +X` (image right = mesh right), `v = -Y` (image down =
mesh down), and the +Z-facing surface sampled. So the box must satisfy

    obb.x = -X,  obb.y = +Y,  obb.z = -Z

which is a consistent right-handed triple, since `(-X) x (+Y) = -Z`. The negative
sign on `obb.x` is counterintuitive and follows from the flipped `xAxis` the existing
code derives from the AABB (`xAxis = xmin - xmax`); it was confirmed empirically
against a real scan before being committed, not just derived on paper.

Axes are additionally **assigned by world-axis agreement rather than by extent**.
`vtkOBBTree` sorts axes largest-first, so on a near-square fragment the in-plane axes
can land on opposite world axes — a 90° error that no amount of sign correction
undoes. Real fragments get close to this: the scan used for verification had x and y
extents within 9% of each other.

Only axis *directions* and their *assignment* change. The box, its extents and the
sampling plane are untouched, so foreshortening behaves exactly as before. The
assignment rule does have one visible consequence: where it picks different in-plane
axes than the extent ordering would, the extent that the image width is measured
along changes with it, so `--sampling-mode width`/`height` can produce a different
pixel scale than `obb` does.

What this fixes is the **discrete** part of the ambiguity — which axis, and which
direction along it — so the 90°, 180° and mirrored outputs go away. It does not touch
the small **continuous** in-plane rotation that `MinimizeAABB` applies after
realignment while tightening the bounding box. That rotation is bounded by ±45° and
is a fraction of a degree in practice: on the verification scan, fitting the emitted
UV map against world coordinates gives `u` along world +X and image `v` along world
−Y with a residual rotation of 0.55°, which is one step of `MinimizeAABB`'s 0.5°
search. Downstream registration absorbs a fraction of a degree; it cannot absorb a
mirror.

## A reflection bug this exposed

`AlignVectorToVector`'s anti-parallel branch intends a 180° rotation about the
fallback axis, `diag(-1, 1, -1)`. It wrote that with `r.diag() = d` where `d` is a
`cv::Vec3d` — which implicitly converts to `cv::Scalar` and selects
`Mat::operator=(const Scalar&)`, filling the **whole** diagonal with `d[0]`. The
result was `diag(-1, -1, -1)`, determinant −1: a reflection, not a rotation.

A reflected realignment leaves `u` and `v` correct but maps world +Z to −Z, so the
sampling rays enter from behind and the **far** surface is sampled and the depth and
position maps are inverted. The branch fires when an OBB in-plane axis is within
~2.56° of anti-parallel to its world axis — which is common for exactly the squared-up
canonical meshes this mode targets. It is now written element-wise, and the fixed
branch also rotates `a` onto `b` so the early-return fires, avoiding an
ill-conditioned Rodrigues evaluation with a near-zero denominator.

This was not caught by the original image-based verification, and could not have
been: a single-layer sheet reads identically from either side, so every quadrant
colour check and the real-scan UV fit passed while the wrong face was being sampled.
The regression tests for it use a **two-layer** mesh and a depth-map parity
assertion.

## Considered Options

- **Opt-in OBB axis direction resolution (chosen).** A sign/assignment fix applied
  immediately after `ComputeOBB`, before the existing realignment. Everything
  downstream — the realignment, the ±45° `MinimizeAABB` search (a rotation, so it
  cannot reintroduce a sign error), the sampling loop, `CreateUVMap` — is unchanged.
  Keeps the sample plane on the OBB plane, so foreshortening behavior is preserved
  exactly.
- **Re-frame reordering onto the world axes entirely** (skip the OBB and
  `MinimizeAABB`, sample along world −Z). Simpler, and it would put the position map
  in oriented world coordinates. Rejected because it costs foreshortening when a
  fragment sits tilted relative to the sample square — the sample plane would no
  longer follow the fragment. Still worth considering later as a second mode under
  the same enum, which is why this is an enum and not a bool.
- **Post-hoc detection and correction** (the older orientation-detector approach:
  flip the image, rewrite the UVs). Deterministic if driven from the OBJ's world
  vertices rather than image content, but it requires keeping image, UV map, depth
  map, and position map mutually consistent through a flip — four artifacts — and it
  undoes work that should not have happened. The original justification for that tool
  (orientation unknown at that stage) no longer holds.
- **`--projection camera` with a synthesized canonical pose.** Needs no change here
  and the plumbing is already proven, but it is pinhole rather than orthographic and
  `CreateProjectiveUVMap` has no near-plane clipping — unsuitable for a measurement
  deliverable.

## Consequences

- **Existing deliverables were produced under `OBB`** and some are mirrored. Whether
  they are regenerated is a separate call; nothing here rewrites them.
- The downstream pipeline must **pass `--orientation canonical` explicitly** to get
  the fix. Until it does, its output is unchanged.
- `ProjectionMode::Camera` honors the setting through the camera it auto-derives.
  `AutoCamera` read the same ambiguous box and inherited all three failure modes:
  which side of the surface it views from (a mirror), its up vector (180°), and which
  in-plane axis becomes the image width (90°). Measured on a canonically oriented
  sheet, the auto camera came out **180° rotated**. Under `Canonical` it is now placed
  on the world +Z side with image right along +X and image down along −Y, matching the
  orthographic convention.

  Note the sign differs from the orthographic frame: `CanonicalizeOBB` resolves the
  thin axis onto −Z, which is what the sampling frame wants, so the camera negates it
  to sit on the +Z side. A camera passed explicitly via `setProjectionParams()` /
  `--camera-file` is used verbatim; the CLI warns when that combination is given.
- The emitted UV map needs no separate correction: `CreateUVMap` derives from the
  same realigned mesh and bounding box the sampling used, so the mesh↔texture pairing
  in the written OBJ follows the image automatically. Verified on a real scan by
  fitting the OBJ's `vt` coordinates against world position (fit residual < 1e-4).
- **`--orientation canonical` requires `--sampling-origin tl`** under
  `--projection orthographic`, and the CLI rejects any other combination (the camera
  path never reads the sampling origin, so it is unconstrained there). `SamplingOrigin` negates the sampling axes downstream of
  canonicalization, so `tr` mirrors the image — reintroducing precisely the defect
  this mode exists to remove. The library API still allows both to be set; only the
  CLI refuses.
- **The `AlignVectorToVector` fix changes `OrientationMode::OBB` output too**, since
  that path hits the same branch. On the verification scan it altered 6.2% of pixels
  (the branch fired for the second alignment, where the buggy form fell through to an
  ill-conditioned Rodrigues evaluation); the image orientation is unchanged. This is
  a deliberate exception to "the default is preserved bit for bit" — the previous
  behavior in that branch was a silent reflection.
- Graph metadata gains an `orientationMode` key. Caches written before it exists load
  fine — `deserialize_` guards on `contains` and falls back to the `OBB` default.
