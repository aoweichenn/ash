#pragma once

#include <memory>

#include "ash/model/provider.hpp"

namespace ash {

// Speaks the /chat/completions dialect used by OpenAI, DeepSeek, Moonshot,
// Qwen/Dashscope and most other hosted endpoints.
[[nodiscard]] std::unique_ptr<ModelProvider> make_openai_compatible(ProviderConfig config);

// Speaks the Anthropic Messages dialect.
[[nodiscard]] std::unique_ptr<ModelProvider> make_anthropic(ProviderConfig config);

}  // namespace ash
