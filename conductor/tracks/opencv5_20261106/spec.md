# Specification: Support OpenCV 5

**Track ID:** opencv5_20261106
**Type:** Chore
**Created:** 2026-11-06
**Status:** Stub
**Source:** [#24](https://github.com/educelab/registration-toolkit/issues/24)

## Summary

Allow RT to configure, build, and be consumed downstream against OpenCV 5 while keeping OpenCV 4 working.

## Acceptance Criteria

- [ ] `find_package`/`find_dependency` accept OpenCV 4 and 5 (`cmake/FindDependencies.cmake`, `cmake/Config.cmake.in`)
- [ ] Link targets resolve on both: `opencv_calib3d`/`opencv_features2d` (4.x) vs `opencv_features`/`opencv_geometry` or `opencv_calib` (5.x)
- [ ] Build and tests pass against OpenCV 4 and 5
- [ ] Brewfile / macOS CI no longer needs the `opencv@4` keg workaround (or keeps it deliberately)

## Dependencies

TBD

## Out of Scope

TBD

## Technical Notes

Headers (`opencv2/calib3d.hpp`, `features2d.hpp`) ship as compatibility shims on 5.x; only CMake target names change. Confirm whether `cv::estimateAffinePartial2D` lives in `opencv_geometry`.

---

_Stub generated from GitHub issue #24. See the issue for full context; refine before planning._
