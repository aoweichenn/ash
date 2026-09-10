#include "ash/model/provider.hpp"

#include <string>

namespace ash {

std::string_view to_string(Role role) noexcept {
    switch (role) {
        case Role::kSystem:
            return "system";
        case Role::kUser:
            return "user";
        case Role::kAssistant:
            return "assistant";
        case Role::kTool:
            return "tool";
    }
    return "user";
}

Role role_from_string(std::string_view name) {
    if (name == "system") {
        return Role::kSystem;
    }
    if (name == "assistant") {
        return Role::kAssistant;
    }
    if (name == "tool") {
        return Role::kTool;
    }
    return Role::kUser;
}

void to_json(nlohmann::json& json, const ToolCall& call) {
    json = nlohmann::json{
        {"id", call.id}, {"name", call.name}, {"arguments", call.arguments}};
}

void from_json(const nlohmann::json& json, ToolCall& call) {
    json.at("id").get_to(call.id);
    json.at("name").get_to(call.name);
    call.arguments = json.value("arguments", nlohmann::json::object());
}

void to_json(nlohmann::json& json, const Message& message) {
    json = nlohmann::json{{"role", std::string{to_string(message.role)}}, {"content", message.content}};
    if (!message.tool_calls.empty()) {
        json["tool_calls"] = message.tool_calls;
    }
    if (!message.tool_call_id.empty()) {
        json["tool_call_id"] = message.tool_call_id;
    }
}

void from_json(const nlohmann::json& json, Message& message) {
    message.role = role_from_string(json.value("role", "user"));
    message.content = json.value("content", "");
    message.tool_calls = json.value("tool_calls", std::vector<ToolCall>{});
    message.tool_call_id = json.value("tool_call_id", "");
}

void to_json(nlohmann::json& json, const ToolSpec& spec) {
    json = nlohmann::json{
        {"name", spec.name}, {"description", spec.description}, {"input_schema", spec.input_schema}};
}

void from_json(const nlohmann::json& json, ToolSpec& spec) {
    json.at("name").get_to(spec.name);
    spec.description = json.value("description", "");
    spec.input_schema = json.value("input_schema", nlohmann::json::object());
}

void to_json(nlohmann::json& json, const ChatRequest& request) {
    json = nlohmann::json{
        {"model", request.model},
        {"messages", request.messages},
        {"tools", request.tools},
        {"temperature", request.temperature}};
    if (request.max_tokens) {
        json["max_tokens"] = *request.max_tokens;
    }
}

void from_json(const nlohmann::json& json, ChatRequest& request) {
    request.model = json.value("model", "");
    request.messages = json.value("messages", std::vector<Message>{});
    request.tools = json.value("tools", std::vector<ToolSpec>{});
    request.temperature = json.value("temperature", 0.0);
    if (json.contains("max_tokens") && !json["max_tokens"].is_null()) {
        request.max_tokens = json["max_tokens"].get<int>();
    }
}

void to_json(nlohmann::json& json, const Usage& usage) {
    json = nlohmann::json{
        {"prompt_tokens", usage.prompt_tokens}, {"completion_tokens", usage.completion_tokens}};
}

void from_json(const nlohmann::json& json, Usage& usage) {
    usage.prompt_tokens = json.value("prompt_tokens", 0);
    usage.completion_tokens = json.value("completion_tokens", 0);
}

void to_json(nlohmann::json& json, const ChatResponse& response) {
    json = nlohmann::json{
        {"message", response.message},
        {"usage", response.usage},
        {"finish_reason", response.finish_reason},
        {"model", response.model}};
}

void from_json(const nlohmann::json& json, ChatResponse& response) {
    response.message = json.value("message", Message{});
    response.usage = json.value("usage", Usage{});
    response.finish_reason = json.value("finish_reason", "");
    response.model = json.value("model", "");
}

}  // namespace ash
