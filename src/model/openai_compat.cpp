#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

#include "ash/io/http_client.hpp"
#include "ash/model/providers.hpp"

namespace ash {

namespace {

std::string join_url(const std::string& base, std::string_view path) {
    std::string url = base;
    while (!url.empty() && url.back() == '/') {
        url.pop_back();
    }
    url.append(path);
    return url;
}

nlohmann::json encode_messages(const std::vector<Message>& messages) {
    nlohmann::json encoded = nlohmann::json::array();
    for (const auto& message : messages) {
        nlohmann::json entry{{"role", std::string{to_string(message.role)}}};
        if (message.role == Role::kAssistant) {
            entry["content"] = message.content.empty() ? nlohmann::json(nullptr) : nlohmann::json(message.content);
            if (!message.tool_calls.empty()) {
                nlohmann::json calls = nlohmann::json::array();
                for (const auto& call : message.tool_calls) {
                    calls.push_back({{"id", call.id},
                                     {"type", "function"},
                                     {"function",
                                      {{"name", call.name}, {"arguments", call.arguments.dump()}}}});
                }
                entry["tool_calls"] = std::move(calls);
            }
        } else if (message.role == Role::kTool) {
            entry["content"] = message.content;
            entry["tool_call_id"] = message.tool_call_id;
        } else {
            entry["content"] = message.content;
        }
        encoded.push_back(std::move(entry));
    }
    return encoded;
}

nlohmann::json encode_request(const ChatRequest& request) {
    nlohmann::json body{{"model", request.model},
                        {"messages", encode_messages(request.messages)},
                        {"temperature", request.temperature},
                        {"stream", false}};
    if (request.max_tokens) {
        body["max_tokens"] = *request.max_tokens;
    }
    if (!request.tools.empty()) {
        nlohmann::json tools = nlohmann::json::array();
        for (const auto& tool : request.tools) {
            tools.push_back({{"type", "function"},
                             {"function",
                              {{"name", tool.name},
                               {"description", tool.description},
                               {"parameters", tool.input_schema}}}});
        }
        body["tools"] = std::move(tools);
    }
    return body;
}

nlohmann::json parse_arguments(const std::string& raw) {
    if (raw.empty()) {
        return nlohmann::json::object();
    }
    nlohmann::json parsed = nlohmann::json::parse(raw, nullptr, false);
    return parsed.is_discarded() ? nlohmann::json::object() : parsed;
}

ChatResponse decode_response(const nlohmann::json& body, const std::string& fallback_model) {
    const auto& choice = body.at("choices").at(0);
    const auto& message = choice.at("message");

    ChatResponse response;
    response.message.role = Role::kAssistant;
    if (message.contains("content") && !message["content"].is_null()) {
        response.message.content = message["content"].get<std::string>();
    }
    if (message.contains("tool_calls") && !message["tool_calls"].is_null()) {
        for (const auto& call : message["tool_calls"]) {
            ToolCall decoded;
            decoded.id = call.value("id", "");
            const auto& function = call.at("function");
            decoded.name = function.value("name", "");
            decoded.arguments = parse_arguments(function.value("arguments", std::string{"{}"}));
            response.message.tool_calls.push_back(std::move(decoded));
        }
    }
    response.finish_reason = choice.value("finish_reason", "");
    response.model = body.value("model", fallback_model);
    if (body.contains("usage") && !body["usage"].is_null()) {
        response.usage.prompt_tokens = body["usage"].value("prompt_tokens", 0);
        response.usage.completion_tokens = body["usage"].value("completion_tokens", 0);
    }
    return response;
}

class OpenAiCompatibleProvider final : public ModelProvider {
public:
    explicit OpenAiCompatibleProvider(ProviderConfig config) : config_(std::move(config)) {}

    [[nodiscard]] std::string_view name() const noexcept override { return "openai-compatible"; }

    [[nodiscard]] const std::string& model() const noexcept override { return config_.model; }

    Task<ChatResponse> chat(ChatRequest request) override {
        if (request.model.empty()) {
            request.model = config_.model;
        }
        if (!request.max_tokens) {
            request.max_tokens = config_.max_tokens;
        }

        HttpRequest http;
        http.url = join_url(config_.base_url, "/chat/completions");
        http.timeout_ms = config_.timeout_ms;
        http.headers = {"Content-Type: application/json", "Authorization: Bearer " + config_.api_key};
        http.body = encode_request(request).dump();

        const HttpResponse response = client_.post(http);
        if (!response.error.empty()) {
            throw std::runtime_error{"transport error: " + response.error};
        }
        if (!response.ok()) {
            throw std::runtime_error{"http " + std::to_string(response.status) + ": " + response.body};
        }

        co_return decode_response(nlohmann::json::parse(response.body), config_.model);
    }

private:
    ProviderConfig config_;
    HttpClient client_;
};

}  // namespace

std::unique_ptr<ModelProvider> make_openai_compatible(ProviderConfig config) {
    return std::make_unique<OpenAiCompatibleProvider>(std::move(config));
}

}  // namespace ash
