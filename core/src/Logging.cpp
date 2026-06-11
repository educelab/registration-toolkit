#include "rt/Logging.hpp"

#include <string>

#include <spdlog/sinks/stdout_color_sinks.h>

namespace
{
auto init_logger() -> std::shared_ptr<spdlog::logger>
{
    spdlog::set_pattern(
        "[%Y-%m-%d %T] [%n] %^[%l]%$ %v", spdlog::pattern_time_type::utc);
    if (auto existing = spdlog::get("rt")) {
        return existing;
    }
    return spdlog::stderr_color_mt("rt");
}
}

auto rt::logger() -> std::shared_ptr<spdlog::logger>
{
    static const auto instance = init_logger();
    return instance;
}

void rt::set_log_level(const std::string_view lvl)
{
    logger()->set_level(spdlog::level::from_str(std::string(lvl)));
}
