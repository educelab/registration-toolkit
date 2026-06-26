#pragma once

/** @file */

#include <cstddef>
#include <vector>

#include <educelab/core/types/UVMap.hpp>

namespace rt
{
/**
 * @brief Per-wedge UV map with a per-map output-space aspect ratio
 *
 * `rt::UVMap` is libcore's per-wedge `educelab::UVMap` (a flat coordinate pool
 * plus a per-face, per-corner index into it, carrying `traits::WithChart`
 * per-coordinate chart indices) extended by public inheritance with a single
 * per-map `aspect` field. `aspect` is the width-to-height ratio of the output
 * texture space and is the fallback a rasterizer uses to size a generated image
 * when no source texture is present to dictate dimensions. See
 * docs/adr/0002-rt-uvmap-subclasses-educelab-uvmap.md.
 *
 * In-memory UV coordinates follow the toolkit's top-left origin invariant (see
 * CONTEXT.md); the bottom-left `vt` flip is applied only at the OBJ/PLY I/O
 * boundary (rt::io::WriteMesh / rt::io::ReadMesh).
 *
 * @warning Slicing risk: never copy an `rt::UVMap` into an `educelab::UVMap` by
 * value — that drops `aspect`. libcore I/O only takes the map by templated
 * reference, so the risk is confined to RT's own code.
 *
 * @note `map()` and `clear()` shadow the base implementations to maintain a
 * lightweight per-face corner-count index (num_faces() / face_corner_count()).
 * The base members are non-virtual, but every call path that mutates an
 * `rt::UVMap` — RT code and libcore's templated `read_obj` alike — dispatches on
 * the concrete `rt::UVMap` type, so the shadowing members are always selected.
 * This index lets WriteUVMap serialize the standalone (mesh-less) `.uvm` graph
 * cache, which the base type cannot otherwise enumerate.
 */
class UVMap : public educelab::UVMap<float, 2, educelab::traits::WithChart>
{
public:
    /** Base libcore UV map type */
    using Base = educelab::UVMap<float, 2, educelab::traits::WithChart>;

    /** @brief Output-space width-to-height aspect ratio */
    float aspect{1.0F};

    /**
     * @brief Assign pool index @p uvIdx to wedge (@p face, @p corner)
     *
     * Forwards to the base implementation and records the per-face corner
     * count so the standalone `.uvm` serializer can enumerate faces.
     */
    void map(std::size_t face, std::size_t corner, std::size_t uvIdx);

    /** @brief Reset pool, per-wedge mapping, and face index to empty */
    void clear() noexcept;

    /**
     * @brief Number of face slots tracked: the largest mapped face index + 1
     *
     * This is an upper bound on the face indices that have been mapped, not a
     * count of non-empty faces. If faces are mapped sparsely (e.g. only face 5
     * is mapped), this returns 6 and faces 0–4 report a face_corner_count() of
     * 0. Iterating `[0, num_faces())` and guarding on face_corner_count() /
     * has() therefore visits every mapped wedge and is what the `.uvm`
     * serializer relies on.
     */
    [[nodiscard]] auto num_faces() const noexcept -> std::size_t;

    /** @brief Number of corner slots seen for @p face (max mapped corner + 1) */
    [[nodiscard]] auto face_corner_count(std::size_t face) const -> std::size_t;

private:
    /** Per-face corner count (max mapped corner index + 1), for serialization */
    std::vector<std::size_t> faceCorners_;
};
}  // namespace rt
