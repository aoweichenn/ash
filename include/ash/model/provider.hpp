#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "ash/task.hpp"

namespace ash {

enum class Role { kSystem, kUser, kAssistant, kTool };

[[nodiscard]] std::string_view to_string(Role role) noexcept;
[[nodiscard]] Role role_from_string(std::string_view name);

// A model's request to run a tool, with arguments already parsed out of the
// provider's wire format.
struct ToolCall {
    std::string id;
    std::string name;
    nlohmann::json arguments = nlohmann::json::object();

    friend bool operator==(const ToolCall&, const ToolCall&) = default;
};

struct Message {
    Role role = Role::kUser;
    std::string content;
    std::vector<ToolCall> tool_calls;  // populated on assistant messages
    std::string tool_call_id;          // populated on tool messages

    friend bool operator==(const Message&, const Message&) = default;
};

// The declaration handed to the model so it knows a tool exists.
struct ToolSpec {
    std::string name;
    std::string description;
    nlohmann::json input_schema = nlohmann::json::object();

    friend bool operator==(const ToolSpec&, const ToolSpec&) = default;
};

struct ChatRequest {
    std::string model;
    std::vector<Message> messages;
    std::vector<ToolSpec> tools;
    double temperature = 0.0;
    std::optional<int> max_tokens;

    friend bool operator==(const ChatRequest&, const ChatRequest&) = default;
};

struct Usage {
    int prompt_tokens = 0;
    int completion_tokens = 0;

    [[nodiscard]] int total_tokens() const noexcept { return prompt_tokens + completion_tokens; }

    friend bool operator==(const Usage&, const Usage&) = default;
};

struct ChatResponse {
    Message message;  // always an assistant message
    Usage usage;
    std::string finish_reason;
    std::string model;

    friend bool operator==(const ChatResponse&, const ChatResponse&) = default;
};

struct ProviderConfig {
    std::string base_url;
    std::string api_key;
    std::string model;
    std::optional<int> max_tokens;
    long timeout_ms = 120000;
};

class ModelProvider {
public:
    virtual ~ModelProvider() = default;

    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual const std::string& model() const noexcept = 0;

    virtual Task<ChatResponse> chat(ChatRequest request) = 0;
};

// Serialization lives here so the recorder and the replayer agree on exactly
// one representation, and so a recorded request can be compared by value.
void to_json(nlohmann::json& json, const ToolCall& call);
void from_json(const nlohmann::json& json, ToolCall& call);

void to_json(nlohmann::json& json, const Message& message);
void from_json(const nlohmann::json& json, Message& message);

void to_json(nlohmann::json& json, const ToolSpec& spec);
void from_json(const nlohmann::json& json, ToolSpec& spec);

void to_json(nlohmann::json& json, const ChatRequest& request);
void from_json(const nlohmann::json& json, ChatRequest& request);

void to_json(nlohmann::json& json, const Usage& usage);
void from_json(const nlohmann::json& json, Usage& usage);

void to_json(nlohmann::json& json, const ChatResponse& response);
void from_json(const nlohmann::json& json, ChatResponse& response);

}  // namespace ash
