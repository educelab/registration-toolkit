# Product Definition: registration-toolkit

## Description

A set of utilities written in C++ for simplifying image alignment tasks,
supporting image-to-image and image-to-mesh registration.

## Problem Statement

Images of the same object captured at different times, with different cameras,
or in different modalities do not line up, so they cannot be compared or
combined. Separately, there is no simple way to transfer a high-quality 2D image
onto a 3D scan's texture, or to flatten a scan's texture into a single usable
image.

## Target Users

- **Imaging researchers** in digital heritage who run the `rt_*` command-line
  tools on manuscripts, papyri, and other artifacts.
- **Pipeline operators** running automated acquisition and registration
  workflows built on the smgl render graph.
- **Downstream C++ developers** linking the installed `rt::core` and
  `rt::graph` libraries into their own projects.

## Key Goals

1. **Accurate, reproducible results:** registrations can be replayed exactly
   (saved transforms, graph caches) and stay comparable across captures.
2. **Full image-type support:** gray, gray+alpha, RGB, and RGBA at 8, 16, and
   32 bits per channel.
3. **Robust mesh handling:** multi-chart, multi-texture meshes are handled
   correctly, with correct orientation and UV conventions.
4. **Clean downstream package:** `rt::core` and `rt::graph` install and are
   consumable from other CMake projects.
