# Specification: Remove the stopgap MeshReadNode.image scalar port

**Track ID:** mesh-image-port_20261106
**Type:** Refactor
**Created:** 2026-11-06
**Status:** Stub
**Source:** [#22](https://github.com/educelab/registration-toolkit/issues/22)

## Summary

Make `MeshReadNode`'s texture output purely plural (`images`) and migrate single-image consumers.

## Acceptance Criteria

- [ ] `MeshReadNode.image` and its getter lambda are removed; `images` is the only texture output
- [ ] `Register.cpp` and other single-image consumers are migrated (select/index adapter node or plural-accepting input ports)
- [ ] Build and tests pass

## Dependencies

TBD

## Out of Scope

TBD

## Technical Notes

`results["fixedImage"]` fans out to several scalar input ports across the 2D→2D and 2D→3D branches; a small reusable index→scalar adapter node is the likely approach. Follow-up to #21.

---

_Stub generated from GitHub issue #22. See the issue for full context; refine before planning._
