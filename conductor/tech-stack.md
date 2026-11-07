# Tech Stack: registration-toolkit

## Language and Build

- **C++20** (GCC 12+, Clang 15+, AppleClang 15+)
- **CMake 3.24+**, Ninja recommended
- Formatting: `.clang-format` (Google-based, 4-space indent, 80 cols); linting:
  `.clang-tidy`

## Frontend / Backend / Database

None. This is a C++ library plus command-line tools.

## Key Dependencies

| Dependency | Version | Role |
| --- | --- | --- |
| ITK | 4+ (5.4 on AppleClang ≤15) | Image registration, transforms |
| OpenCV | 4.x | Image I/O, feature detection, image ops |
| VTK | 6+ | Mesh geometry for texture reordering |
| libtiff | 4.0.9+ | TIFF I/O (must match the build ITK/OpenCV use) |
| EduceLab libcore | 0.3.0+ | Canonical mesh and UV map types, mesh I/O |
| EduceLab smgl | 0.11.0+ | Render graph (`rt::graph`) |
| spdlog | 1.9.0+ | Logging |
| Eigen3 | — | Mesh flattening apps |
| Boost.ProgramOptions | 1.70+ | CLI parsing (apps only) |
| bvh, OpenABF | pinned commits | Fetched at build time |
| GoogleTest | 1.17.0 | Unit tests (fetched when `RT_BUILD_TESTS=ON`) |

libcore and smgl are PUBLIC dependencies and must be installed on the system.
CI pins them in `.github/workflows/ci.yml`.

## Infrastructure and Distribution

Supported channels (changes must keep these working):

- **Homebrew tap:** `brew install educelab/code/registration-toolkit`
- **Docker:** `ghcr.io/educelab/registration-toolkit` (amd64 + arm64), built by
  `.github/workflows/build_docker.yml` on pushes to `develop` and `v*` tags
- **Source + CMake package:** `cmake --install`, consumed downstream via
  `find_package` (`rt::core`, `rt::graph`); `cmake/Config.cmake.in` must mirror
  `cmake/FindDependencies.cmake`

`singularity/registration-toolkit.def` exists but is not a supported channel.

CI (`.github/workflows/ci.yml`): Ubuntu 24.04, Ubuntu 22.04, and macOS, Release
builds with tests.
