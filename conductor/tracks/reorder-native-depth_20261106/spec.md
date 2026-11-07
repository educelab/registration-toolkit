# Specification: ReorderUnorganizedTexture: support non-8bpc / non-3-channel input textures

**Track ID:** reorder-native-depth_20261106
**Type:** Feature
**Created:** 2026-11-06
**Status:** Stub
**Source:** [#19](https://github.com/educelab/registration-toolkit/issues/19)

## Summary

Preserve native bit depth and channel count through texture reordering instead of normalizing every input to 8-bit BGR.

## Acceptance Criteria

- [ ] `sample_surface_color_` handles arbitrary depth/channel counts, not only `cv::Vec3b`
- [ ] Output texture type matches the input instead of a fixed `CV_8UC3`
- [ ] `QuantizeImage`/`ColorConvertImage` normalization in `setTextureMats()` is revisited
- [ ] Doxygen note on `setTextureMats()` is updated

## Dependencies

TBD

## Out of Scope

TBD

## Technical Notes

Policy for multi-texture inputs with mismatched types is undecided.

---

_Stub generated from GitHub issue #19. See the issue for full context; refine before planning._
