# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and test

C++20 / CMake 3.24+. Dependencies on macOS come from `brew bundle` (Brewfile).
`educelab/libcore` (≥0.3.0) and `educelab/smgl` (≥0.11.0) are found via
`find_package` and must be **installed**, not vendored; CI builds them from the
tags pinned in `.github/workflows/ci.yml`.

```shell
cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release -DRT_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build -V                      # all tests
ctest --test-dir build -R rt_TestMeshIO        # one test executable
build/bin/rt_TestMeshIO --gtest_filter='Suite.Case'   # one gtest case
```

- Use Release builds: mesh flattening (`rt_dewarp`, `rt_seam_flattening`) is
  dramatically slower in Debug.
- On macOS, Homebrew's `opencv@4` is keg-only; pass
  `-DOpenCV_DIR="$(brew --prefix opencv@4)/lib/cmake/opencv4"`.
- Executables (apps and tests) land in `build/bin/`; tests run with that as
  their working directory.
- Each test source is its own executable named `rt_<FileName>`. New tests must
  be added to `tests/CMakeLists.txt` — in `tests` (links `rt::core`) or
  `graph_tests` (also links `rt::graph`).
- Formatting: `.clang-format` (Google-based, 4-space indent, 80 cols).
  Lint config in `.clang-tidy`.

## Architecture

Three layers, each a CMake target:

- **`core/` → `rt::core`**: the algorithms and types. Registration
  (landmark detection, affine landmark registration, B-spline landmark
  warping, deformable registration, resampling) operates purely on **images**
  using ITK, with OpenCV for I/O and feature detection;
  `util/ITKOpenCVBridge.hpp` converts between `cv::Mat` and ITK images.
  Meshes appear only in I/O, retexturing, and `ReorderUnorganizedTexture`
  (which converts to VTK internally to compute).
- **`graph/` → `rt::graph`**: wraps core functionality as `smgl::Node`
  subclasses (input/output ports) so pipelines can be assembled as a render
  graph with on-disk caching. Every node must be registered in
  `RegisterNodesImpl()` in `graph/src/graph.cpp`, and nodes with state
  implement `serialize_`/`deserialize_` for the graph cache.
- **`apps/`**: Boost.ProgramOptions CLIs (`rt_*`). `rt_register` and
  `rt_reorder_texture` build an `smgl::Graph` at runtime from CLI flags (see
  `apps/src/Register.cpp`); the rest call `rt::core` directly.

Image-to-mesh registration is 2D-to-2D: the moving image is registered to the
mesh's texture image, and the resulting transform is applied to the UV map.

## Domain conventions

`CONTEXT.md` is the domain glossary (Mesh, UV Map, UV origin, Chart, Canonical
orientation, Aspect) and `docs/adr/` records the decisions behind them. Key
invariants:

- `rt::Mesh` (an `educelab::Mesh` alias in `rt/types/Mesh.hpp`) is the only
  mesh type; convert to VTK etc. only at algorithm boundaries (ADR 0001).
- `rt::UVMap` publicly subclasses libcore's `educelab::UVMap` to add `aspect`.
  Pass it as `rt::UVMap`; copying by value into the base type slices off
  `aspect` (ADR 0002).
- In-memory UV coordinates are **top-left origin** (`v` increases downward).
  Bottom-left formats like OBJ `vt` are flipped at the I/O boundary only.
- UV maps are per-wedge and may be multi-chart; meshes can carry multiple
  texture images indexed by chart.
- The installed package is consumed downstream: `cmake/Config.cmake.in`
  mirrors the dependencies and versions in `cmake/FindDependencies.cmake`
  (libcore, smgl, OpenCV, ITK, VTK, TIFF, spdlog), so keep them in sync when
  changing either.

## Landmarks files

Plain text, one pair per line as `fX fY mX mY` (fixed then moving pixel
coordinates), parsed with `std::stod`; `#` starts a comment.
