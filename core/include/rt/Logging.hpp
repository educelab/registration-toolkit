#pragma once

#include <memory>
#include <string_view>

#include <spdlog/spdlog.h>

namespace rt
{
auto logger() -> std::shared_ptr<spdlog::logger>;

void set_log_level(std::string_view lvl);
}
