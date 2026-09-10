#pragma once

#include <optional>
#include <string_view>

namespace ash {

// USD per million tokens.
struct Price {
    double input_per_mtok = 0.0;
    double output_per_mtok = 0.0;
};

// Prices are published per model and change without notice, so they live in one
// table here rather than being spread through the code, and an unknown model
// yields nothing rather than a guess. These are DeepSeek's peak rates; their
// off-peak rates are half, so a figure derived from this table is an upper
// bound. Check the provider's pricing page before quoting a number anywhere.
[[nodiscard]] std::optional<Price> price_for(std::string_view model) noexcept;

// Usage does not record how much of the prompt hit the provider's cache, so
// every prompt token is billed at the cache-miss rate. That overestimates
// slightly, which is the direction an estimate should err in.
[[nodiscard]] std::optional<double> estimate_cost_usd(std::string_view model,
                                                      int prompt_tokens,
                                                      int completion_tokens) noexcept;

}  // namespace ash
