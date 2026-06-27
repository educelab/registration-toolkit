---
status: accepted
---

# rt::UVMap subclasses educelab::UVMap to carry an output-space aspect

`rt::UVMap` is defined as `class UVMap : public educelab::UVMap<float, 2,
educelab::traits::WithChart> { float aspect{1.0f}; }`. We adopt libcore's UVMap but
add a single per-map `aspect` field by **public inheritance**. `aspect` is the
width-to-height ratio of the output texture space and is the fallback a rasterizer
uses to size a generated image when no source texture is present to dictate
dimensions — a real domain attribute that travels with the map and survives
serialization. libcore's UVMap deliberately omits per-map metadata (its `WithChart`
trait is per-coordinate), so the attribute has no home in the library type.

## Considered Options

- **Public inheritance + `aspect` field (chosen).** libcore's mesh I/O takes the UV
  map by *template reference*, so a derived `rt::UVMap` passes straight through
  `read_mesh`/`write_mesh` and `traits::has_chart` resolves via the inherited
  `Coordinate`. No forwarding boilerplate. Mirrors libcore's own idiom
  (`Mesh::Vertex : public Vec, public VertexTraits`).
- **Composition wrapper.** Rejected — forces unwrapping `.map` at every libcore I/O
  and rasterizer call, plus method-forwarding boilerplate.
- **Drop aspect / pass as a parameter.** Rejected — `aspect` must persist on the map
  and survive the `.uvm` cache and node-to-node flow for an external/future
  rasterizer (VC `PerPixelMap`-style) to read; nothing in-repo consumes it yet.
- **Upstream a per-map metadata slot into libcore.** Rejected — widens scope to a
  libcore change and release.

## Consequences

- **Slicing risk**: copying an `rt::UVMap` into an `educelab::UVMap` *by value* drops
  `aspect`. libcore only takes templated refs, so the risk is confined to RT's own
  code (pass `rt::UVMap` by ref/value). The base destructor is non-virtual, which is
  fine since RT never deletes through an `educelab::UVMap*`.
- In-memory UV coordinates follow the top-left origin invariant; the `Origin` enum is
  removed and the flip is applied at the I/O boundary.
- `UVMapIO` (the `.uvm` graph-cache format) is rewritten for the new layout (pool +
  per-wedge + chart + `aspect`) as version 2. Legacy version-1 (per-face) caches are
  converted on load — their flat UV pool and face→3-index map become the per-wedge
  representation (default chart 0, `aspect` recovered from the old width/height), so
  existing caches keep working without regeneration.
- Contributes to the **major version bump** (the UVMap API changes substantially).
