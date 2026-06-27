#pragma once

/** @file */

#include <filesystem>

#include "rt/types/UVMap.hpp"

namespace rt
{

/** @brief Write a UVMap to a file (.uvm) */
void WriteUVMap(const std::filesystem::path& path, const UVMap& uvMap);

/** @brief Read a UVMap from a file (.uvm) */
auto ReadUVMap(const std::filesystem::path& path) -> UVMap;

}  // namespace rt