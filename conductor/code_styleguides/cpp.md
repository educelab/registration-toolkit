# C++ Style Guide: registration-toolkit

Derived from `.clang-format`, `.clang-tidy`, and existing code. When in doubt,
match the surrounding file.

## Tooling

- Format with `clang-format` using the repo's `.clang-format` (Google-based,
  4-space indent, 80-column limit, left pointer alignment `T* p`, no namespace
  indentation). Use `// clang-format off/on` sparingly, e.g. for registration
  tables.
- `.clang-tidy` enables `readability-*`, `google-*`, `cppcoreguidelines-*`,
  `modernize-*`, `performance-*`, `misc-*`, `llvm-*` with some exclusions; treat
  its warnings as guidance.
- Standard: C++20. Use `std::filesystem` (aliased `namespace fs =
  std::filesystem;` in sources).

## Files and Layout

- One class per file, PascalCase filenames: `include/rt/Foo.hpp` +
  `src/Foo.cpp`. Subsystems live in `rt/io/`, `rt/types/`, `rt/util/`; graph
  nodes in `graph/include/rt/graph/`.
- Headers start with `#pragma once` then `/** @file */`.
- Include order (enforced by clang-format categories): C++ std, C std, VTK,
  ITK, OpenCV, other third-party, then project headers in quotes
  (`"rt/..."`). A source file includes its own header first.
- Sources may use `using namespace rt;` and define members out-of-line
  (`void Foo::bar() {}`); headers never use `using namespace`.

## Naming

| Kind | Style | Example |
| --- | --- | --- |
| Namespaces | lowercase | `rt`, `rt::graph` |
| Classes, structs, enums | PascalCase | `ReorderUnorganizedTexture` |
| Enum values | PascalCase, `enum class` | `OrientationMode::Canonical` |
| Free functions | PascalCase | `ReadMesh`, `QuantizeImage` |
| Public member functions | camelCase | `setMesh`, `samplingOrigin` |
| Private member functions | snake_case + `_` | `create_texture_` |
| Private data members | camelCase + `_` | `path_`, `useFirstIntersection_` |
| Public struct fields | lowercase / camelCase | `fx`, `extrinsics` |
| Constants | UPPER_SNAKE | `DEFAULT_SAMPLE_RATE` |
| Graph nodes | `<Thing>Node` | `MeshReadNode` |

## Functions and Types

- Prefer trailing return types for non-void functions:
  `auto compute() -> cv::Mat;`.
- Mark getters and pure queries `[[nodiscard]]`.
- Brace-initialize members with defaults: `double fx{1.0};`.
- Shared ownership uses the type's `::Pointer` alias (e.g.
  `Mesh::Pointer`).
- Algorithm classes follow a setter → `compute()` → getter pattern; graph nodes
  wrap them with `smgl::InputPort`/`OutputPort` members bound to private
  fields.
- Alternative logical operators (`not`, `and`, `or`) are used in existing code
  and are acceptable.

## Errors and Logging

- Throw `rt::IOException` for read/write failures, `std::runtime_error` or
  `std::invalid_argument` for bad state/arguments. Validate inputs at the top
  of `compute()`/`write()`.
- Log through `rt::logger()` (spdlog) with `{}` formatting. Some older
  registration classes still print with `std::cout`/`std::cerr`; don't add
  more, and migrate them when touching that code.

## Documentation

- Doxygen `/** ... */` on public API: `@brief` first, a short paragraph only
  when behavior is non-obvious, `@note`/`@see`/`@copydoc` as needed.
- Docstrings are descriptive but not exhaustive and written for humans.
- Inline comments are need-to-know, 2-3 lines max.

## Domain Invariants

See `CONTEXT.md` and `docs/adr/`. In particular: `rt::Mesh` and `rt::UVMap`
are the only mesh/UV types, UVs are top-left origin in memory, and `rt::UVMap`
must not be copied by value into `educelab::UVMap` (it slices off `aspect`).
