#include "rt/types/UVMap.hpp"

#include <algorithm>

using namespace rt;

void UVMap::map(std::size_t face, std::size_t corner, std::size_t uvIdx)
{
    Base::map(face, corner, uvIdx);
    if (face >= faceCorners_.size()) {
        faceCorners_.resize(face + 1, 0);
    }
    faceCorners_[face] = std::max(faceCorners_[face], corner + 1);
}

void UVMap::clear() noexcept
{
    Base::clear();
    faceCorners_.clear();
}

auto UVMap::num_faces() const noexcept -> std::size_t
{
    return faceCorners_.size();
}

auto UVMap::face_corner_count(std::size_t face) const -> std::size_t
{
    return (face < faceCorners_.size()) ? faceCorners_[face] : 0;
}
