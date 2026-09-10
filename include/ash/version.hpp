#pragma once

#include <string_view>

namespace ash {

inline constexpr std::string_view kVersion = ASH_VERSION_STRING;

[[nodiscard]] std::string_view version() noexcept;

}  // namespace ash
