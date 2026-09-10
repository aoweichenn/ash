#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "ash/io/http_client.hpp"
#include "ash/model/providers.hpp"

namespace ash {

namespace {

constexpr int kDefaultMaxTokens = 4096;

std::string join_url(const std::string& base, std::string_view path) {
    std::string url = base;
    while (!url.empty() && url.back() == '/') {
        url.pop_back();
    }
    url.append(path);
    return url;
}

// Anthropic has no tool role: tool results ride back as user messages holding
// tool_result blocks, and consecutive results collapse into a single message.
nlohmann::json encode_messages(const std::vector<Message>& messages, std::string& system_out) {
    nlohmann::json encoded = nlohmann::json::array();
    nlohmann::json pending_results = nlohmann::json::array();

    auto flush_results = [&encoded, &pending_results] {
        if (!pending_results.empty()) {
            encoded.push_back({{"role", "user"}, {"content", std::move(pending_results)}});
            pending_results = nlohmann::json::array();
        }
    };

    for (const auto& message : messages) {
        switch (message.role) {
            case Role::kSystem:
                if (!system_out.empty()) {
                    system_out += "\n\n";
                }
                system_out += message.content;
                break;

            case Role::kTool:
                pending_results.push_back({{"type", "tool_result"},
                                           {"tool_use_id", message.tool_call_id},
                                           {"content", message.content}});
                break;

            case Role::kUser:
                flush_results();
                encoded.push_back({{"role", "user"}, {"content", message.content}});
                break;

            case Role::kAssistant: {
                flush_results();
                nlohmann::json blocks = nlohmann::json::array();
                if (!message.content.empty()) {
                    blocks.push_back({{"type", "text"}, {"text", message.content}});
                }
                for (const auto& call : message.tool_calls) {
                    blocks.push_back({{"type", "tool_use"},
                                      {"id", call.id},
                                      {"name", call.name},
                                      {"input", call.arguments}});
                }
                if (blocks.empty()) {
                    blocks.push_back({{"type", "text"}, {"text", ""}});
                }
                encoded.push_back({{"role", "assistant"}, {"content", std::move(blocks)}});
                break;
            }
        }
    }

    flush_results();
    return encoded;
}

nlohmann::json encode_request(const ChatRequest& request) {
    std::string system;
    nlohmann::json body{{"model", request.model},
                        {"messages", encode_messages(request.messages, system)},
                        {"max_tokens", request.max_tokens.value_or(kDefaultMaxTokens)},
                        {"temperature", request.temperature}};
    if (!system.empty()) {
        body["system"] = std::move(system);
    }
    if (!request.tools.empty()) {
        nlohmann::json tools = nlohmann::json::array();
        for (const auto& tool : request.tools) {
            tools.push_back({{"name", tool.name},
                             {"description", tool.description},
                             {"input_schema", tool.input_schema}});
        }
        body["tools"] = std::move(tools);
    }
    return body;
}

ChatResponse decode_response(const nlohmann::json& body, const std::string& fallback_model) {
    ChatResponse response;
    response.message.role = Role::kAssistant;

    for (const auto& block : body.at("content")) {
        const std::string type = block.value("type", "");
        if (type == "text") {
            response.message.content += block.value("text", "");
        } else if (type == "tool_use") {
            ToolCall call;
            call.id = block.value("id", "");
            call.name = block.value("name", "");
            call.arguments = block.value("input", nlohmann::json::object());
            response.message.tool_calls.push_back(std::move(call));
        }
    }

    response.finish_reason = body.value("stop_reason", "");
    response.model = body.value("model", fallback_model);
    if (body.contains("usage") && !body["usage"].is_null()) {
        response.usage.prompt_tokens = body["usage"].value("input_tokens", 0);
        response.usage.completion_tokens = body["usage"].value("output_tokens", 0);
    }
    return response;
}

class AnthropicProvider final : public ModelProvider {
public:
    explicit AnthropicProvider(ProviderConfig config) : config_(std::move(config)) {}

    [[nodiscard]] std::string_view name() const noexcept override { return "anthropic"; }

    [[nodiscard]] const std::string& model() const noexcept override { return config_.model; }

    Task<ChatResponse> chat(ChatRequest request) override {
        if (request.model.empty()) {
            request.model = config_.model;
        }
        if (!request.max_tokens) {
            request.max_tokens = config_.max_tokens;
        }

        HttpRequest http;
        http.url = join_url(config_.base_url, "/messages");
        http.timeout_ms = config_.timeout_ms;
        http.headers = {"content-type: application/json",
                        "x-api-key: " + config_.api_key,
                        "anthropic-version: 2023-06-01"};
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

std::unique_ptr<ModelProvider> make_anthropic(ProviderConfig config) {
    return std::make_unique<AnthropicProvider>(std::move(config));
}

}  // namespace ash
