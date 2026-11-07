# Specification: Docker: multi-stage build to shrink runtime image

**Track ID:** docker-multistage_20261106
**Type:** Chore
**Created:** 2026-11-06
**Status:** Stub
**Source:** [#13](https://github.com/educelab/registration-toolkit/issues/13)

## Summary

Split the Dockerfile into builder and runtime stages so the published image ships only runtime libraries and installed artifacts.

## Acceptance Criteria

- [ ] Dockerfile has `builder` and `runtime` stages; runtime has no compilers or `-dev` packages
- [ ] `ldd` on every `rt_*` binary resolves on amd64 and arm64
- [ ] Image size reduction is measured and reported

## Dependencies

TBD

## Out of Scope

TBD

## Technical Notes

Runtime package names differ across Ubuntu releases and occasionally between architectures. Requires real build + run on both arches. Follow-up deferred from #12.

---

_Stub generated from GitHub issue #13. See the issue for full context; refine before planning._
