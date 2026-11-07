# Workflow: registration-toolkit

## Test-Driven Development (Strict)

Tests come before implementation. No task's implementation starts until a
failing test exists for it.

1. **Red:** write a GoogleTest case in `tests/src/` that captures the required
   behavior. New test files go in `tests/CMakeLists.txt`: in `tests` (links
   `rt::core`) or `graph_tests` (also links `rt::graph`).
2. Build and confirm the test fails for the expected reason:
   ```shell
   cmake --build build && build/bin/rt_TestFoo --gtest_filter='Suite.Case'
   ```
3. **Green:** implement the minimum needed to pass.
4. **Refactor** with the full suite passing: `ctest --test-dir build`.

Configure with `-DCMAKE_BUILD_TYPE=Release -DRT_BUILD_TESTS=ON`. For behavior
that resists exact assertions (registration quality, image appearance), test
the verifiable parts: dimensions, types, transforms on known inputs,
orientation and UV invariants on synthetic meshes, round-trip I/O.

## Commits

Match existing history:

- Imperative, descriptive summary line (e.g. "Support multiple texture images
  per mesh").
- Optional lowercase area prefix for narrow changes: `docs:`, `cmake:`, `gh:`.
- PRs are squash-merged into `develop`; the squash title carries `(#N)` and
  `closes #N` where applicable.
- Work on a feature branch off `develop`; never commit directly to `develop`
  for non-trivial changes.

## Code Review

Non-trivial changes go through a PR to `develop` and require review. Docs-only
fixes and version bumps may be committed directly. CI (Ubuntu 24.04, 22.04,
macOS) must pass before merge.

## Verification Checkpoints

Manual verification happens **at track completion**: run the affected `rt_*`
tools on real data and inspect outputs (alignment, orientation, texture
appearance) before the track is marked complete. Automated tests gate every
task and phase.

## Task Lifecycle

1. **Pending:** task defined in the track's `plan.md`.
2. **In progress:** failing test written and confirmed red.
3. **Implemented:** test green; full `ctest` suite passes.
4. **Documented:** if a domain term or architectural decision changed, update
   `CONTEXT.md` and/or add an ADR in `docs/adr/` (see
   `product-guidelines.md`).
5. **Done:** committed on the track branch; task checked off in `plan.md`.

A track is complete when all tasks are done, CI passes, manual verification is
signed off, and the PR is merged.
