#include "ash/runtime.hpp"

#include <string>
#include <utility>

#include "ash/cancellation.hpp"

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

// The loop itself, handed a conversation that is already open. Both entry points
// below differ only in the messages they start from, so nothing past this line
// has to know which one ran.
Task<AgentResult> run_conversation(ModelProvider& provider,
                                   const ToolRegistry& tools,
                                   std::vector<Message> transcript,
                                   int max_steps,
                                   std::stop_token stop,
                                   StreamSink* sink) {
    AgentResult result;
    result.transcript = std::move(transcript);

    NullSink dropped;
    StreamSink& events = sink != nullptr ? *sink : dropped;

    for (int step = 0; step < max_steps; ++step) {
        if (stop.stop_requested()) {
            result.stop_reason = "cancelled";
            co_return result;
        }

        ChatRequest request;
        request.messages = result.transcript;
        request.tools = tools.specs();

        ChatResponse response;
        try {
            response = co_await provider.chat_stream(std::move(request), events, stop);
        } catch (const Cancelled&) {
            // A stop request that landed mid-call is an ending, not a failure:
            // the run keeps what it had and says why it stopped.
            result.stop_reason = "cancelled";
            co_return result;
        }

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

}  // namespace

Task<AgentResult> run_agent(ModelProvider& provider,
                            const ToolRegistry& tools,
                            std::string task,
                            AgentOptions options,
                            std::stop_token stop,
                            StreamSink* sink) {
    const int max_steps = options.max_steps;
    std::vector<Message> opening;
    opening.reserve(2);
    opening.push_back(make_message(Role::kSystem, std::move(options.system_prompt)));
    opening.push_back(make_message(Role::kUser, std::move(task)));
    return run_conversation(provider, tools, std::move(opening), max_steps, stop, sink);
}

Task<AgentResult> run_agent(ModelProvider& provider,
                            const ToolRegistry& tools,
                            std::vector<Message> history,
                            std::string task,
                            AgentOptions options,
                            std::stop_token stop,
                            StreamSink* sink) {
    const int max_steps = options.max_steps;
    std::vector<Message> opening;
    opening.reserve(history.size() + 2);
    opening.push_back(make_message(Role::kSystem, std::move(options.system_prompt)));
    for (Message& message : history) {
        opening.push_back(std::move(message));
    }
    opening.push_back(make_message(Role::kUser, std::move(task)));
    return run_conversation(provider, tools, std::move(opening), max_steps, stop, sink);
}

}  // namespace ash
