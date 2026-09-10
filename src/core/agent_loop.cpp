#include "ash/runtime.hpp"

#include <string>
#include <utility>

namespace ash {

namespace {

Message make_message(Role role, std::string content) {
    Message message;
    message.role = role;
    message.content = std::move(content);
    return message;
}

Message make_tool_message(std::string call_id, std::string content) {
    Message message;
    message.role = Role::kTool;
    message.content = std::move(content);
    message.tool_call_id = std::move(call_id);
    return message;
}

}  // namespace

Task<AgentResult> run_agent(ModelProvider& provider,
                            const ToolRegistry& tools,
                            std::string task,
                            AgentOptions options,
                            std::stop_token stop) {
    AgentResult result;
    result.transcript.push_back(make_message(Role::kSystem, std::move(options.system_prompt)));
    result.transcript.push_back(make_message(Role::kUser, std::move(task)));

    for (int step = 0; step < options.max_steps; ++step) {
        if (stop.stop_requested()) {
            result.stop_reason = "cancelled";
            co_return result;
        }

        ChatRequest request;
        request.messages = result.transcript;
        request.tools = tools.specs();

        const ChatResponse response = co_await provider.chat(std::move(request));

        result.usage.prompt_tokens += response.usage.prompt_tokens;
        result.usage.completion_tokens += response.usage.completion_tokens;
        result.transcript.push_back(response.message);

        if (response.message.tool_calls.empty()) {
            result.stop_reason = "completed";
            co_return result;
        }

        AgentStep step_record;
        step_record.assistant = response.message;

        for (const auto& call : response.message.tool_calls) {
            ToolResult tool_result;
            const Tool* tool = tools.find(call.name);
            if (tool == nullptr) {
                tool_result.content = "unknown tool: " + call.name;
                tool_result.is_error = true;
            } else {
                tool_result = co_await tool->invoke(call.arguments, stop);
            }

            Message message = make_tool_message(call.id, std::move(tool_result.content));
            result.transcript.push_back(message);
            step_record.tool_results.push_back(std::move(message));
        }

        result.steps.push_back(std::move(step_record));
    }

    result.stop_reason = "max_steps";
    co_return result;
}

}  // namespace ash
