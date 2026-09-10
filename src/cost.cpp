#include "ash/cost.hpp"

#include <array>
#include <string_view>

namespace ash {

namespace {

struct Entry {
    std::string_view model;
    Price price;
};

// DeepSeek's published peak rates, in USD per million tokens.
constexpr std::array<Entry, 2> kPrices{{
    {"deepseek-flash", Price{0.30, 1.20}},
    {"deepseek-v4-pro", Price{1.32, 3.96}},
}};

}  // namespace

std::optional<Price> price_for(std::string_view model) noexcept {
    for (const Entry& entry : kPrices) {
        if (entry.model == model) {
            return entry.price;
        }
    }
    return std::nullopt;
}

std::optional<double> estimate_cost_usd(std::string_view model,
                                        int prompt_tokens,
                                        int completion_tokens) noexcept {
    const std::optional<Price> price = price_for(model);
    if (!price) {
        return std::nullopt;
    }
    return static_cast<double>(prompt_tokens) * price->input_per_mtok / 1'000'000.0 +
           static_cast<double>(completion_tokens) * price->output_per_mtok / 1'000'000.0;
}

}  // namespace ash
