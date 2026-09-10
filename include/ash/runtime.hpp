#pragma once

#include <stop_token>
#include <string>
#include <vector>

#include "ash/model/provider.hpp"
#include "ash/task.hpp"
#include "ash/tool/tool.hpp"

namespace ash {

struct AgentOptions {
    std::string system_prompt =
        "You are a helpful assistant. Call the provided tools when they help you answer.";
    int max_steps = 16;
};

// One model turn plus the tool results it asked for.
struct AgentStep {
    Message assistant;
    std::vector<Message> tool_results;
};

struct AgentResult {
    std::vector<Message> transcript;  // the full conversation, system message first
    std::vector<AgentStep> steps;
    Usage usage;
    std::string stop_reason;  // "completed" | "max_steps" | "cancelled"
};

// Drives the model/tool loop until the model answers without asking for a
// tool, the step budget runs out, or `stop` is requested.
[[nodiscard]] Task<AgentResult> run_agent(ModelProvider& provider,
                                          const ToolRegistry& tools,
                                          std::string task,
                                          AgentOptions options = {},
                                          std::stop_token stop = {});

}  // namespace ash
