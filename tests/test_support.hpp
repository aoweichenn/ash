#pragma once

// Shared fixtures for the ash test suite. Everything is inline so the header
// can be included by several translation units.

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "ash/model/provider.hpp"
#include "ash/model/stream.hpp"
#include "ash/task.hpp"
#include "ash/tool/tool.hpp"

namespace ash::test {

namespace fs = std::filesystem;

// A scratch directory that removes itself, so tests never leak temp files.
class TempDir {
public:
    TempDir() {
        static int counter = 0;
        path_ = fs::temp_directory_path() / ("ash_test_" + std::to_string(++counter));
        std::error_code error;
        fs::remove_all(path_, error);
        fs::create_directories(path_);
    }

    ~TempDir() {
        std::error_code error;
        fs::remove_all(path_, error);
    }

    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;
    TempDir(TempDir&&) = delete;
    TempDir& operator=(TempDir&&) = delete;

    [[nodiscard]] std::string str() const { return path_.string(); }
    [[nodiscard]] const fs::path& path() const noexcept { return path_; }
    [[nodiscard]] fs::path operator/(std::string_view name) const { return path_ / name; }

private:
    fs::path path_;
};

inline void write_text_file(const fs::path& path, const std::string& content) {
    std::ofstream output{path, std::ios::binary | std::ios::trunc};
    output << content;
}

inline ash::Message assistant_text(std::string content) {
    ash::Message message;
    message.role = ash::Role::kAssistant;
    message.content = std::move(content);
    return message;
}

inline ash::Message assistant_tool_call(std::string id, std::string name, nlohmann::json arguments) {
    ash::Message message;
    message.role = ash::Role::kAssistant;
    ash::ToolCall call;
    call.id = std::move(id);
    call.name = std::move(name);
    call.arguments = std::move(arguments);
    message.tool_calls.push_back(std::move(call));
    return message;
}

inline ash::ChatResponse reply(ash::Message message, ash::Usage usage = {}) {
    ash::ChatResponse response;
    response.message = std::move(message);
    response.usage = usage;
    response.finish_reason = "stop";
    response.model = "scripted-model";
    return response;
}

// Replays a fixed script. This is the test double the whole loop is judged
// against: no network, no API key, fully deterministic.
class ScriptedProvider final : public ash::ModelProvider {
public:
    explicit ScriptedProvider(std::vector<ash::ChatResponse> script) : script_(std::move(script)) {}

    [[nodiscard]] std::string_view name() const noexcept override { return "scripted"; }
    [[nodiscard]] const std::string& model() const noexcept override { return model_; }

    ash::Task<ash::ChatResponse> chat(ash::ChatRequest request) override {
        requests.push_back(std::move(request));
        if (next_ >= script_.size()) {
            throw std::runtime_error{"scripted provider ran out of responses"};
        }
        co_return script_[next_++];
    }

    std::vector<ash::ChatRequest> requests;

private:
    std::vector<ash::ChatResponse> script_;
    std::size_t next_ = 0;
    std::string model_ = "scripted-model";
};

// Replays a fixed script, but answers a streaming call the way an endpoint
// does: in pieces, then a closing event.
//
// The text is split one character at a time on purpose. A test that only
// compared the joined text could not tell this apart from a provider that sends
// the whole answer in a single event at the end, and the whole point of the
// seam is that the pieces arrive while the call is still running.
class StreamingScriptedProvider final : public ash::ModelProvider {
public:
    explicit StreamingScriptedProvider(std::vector<ash::ChatResponse> script)
        : script_(std::move(script)) {}

    [[nodiscard]] std::string_view name() const noexcept override { return "scripted"; }
    [[nodiscard]] const std::string& model() const noexcept override { return model_; }

    ash::Task<ash::ChatResponse> chat(ash::ChatRequest request) override {
        requests.push_back(std::move(request));
        if (next_ >= script_.size()) {
            throw std::runtime_error{"scripted provider ran out of responses"};
        }
        co_return script_[next_++];
    }

    ash::Task<ash::ChatResponse> chat_stream(ash::ChatRequest request,
                                             ash::StreamSink& sink,
                                             std::stop_token) override {
        requests.push_back(std::move(request));
        if (next_ >= script_.size()) {
            throw std::runtime_error{"scripted provider ran out of responses"};
        }
        // Copied rather than moved out of the script: the caller may replay the
        // same script, and a response that had been hollowed out would make the
        // second run disagree with the first for no visible reason.
        const ash::ChatResponse response = script_[next_++];

        for (const char piece : response.message.content) {
            sink.on_event(ash::TextDelta{std::string(1, piece)});
        }
        for (std::size_t index = 0; index < response.message.tool_calls.size(); ++index) {
            const ash::ToolCall& call = response.message.tool_calls[index];
            sink.on_event(ash::ToolCallDelta{static_cast<int>(index), call.id, call.name,
                                             call.arguments.dump()});
        }
        if (response.usage.prompt_tokens != 0 || response.usage.completion_tokens != 0) {
            sink.on_event(ash::UsageDelta{response.usage});
        }
        sink.on_event(ash::StreamDone{response.finish_reason, response.model});
        co_return response;
    }

    std::vector<ash::ChatRequest> requests;

private:
    std::vector<ash::ChatResponse> script_;
    std::size_t next_ = 0;
    std::string model_ = "scripted-model";
};

// Records every event a streaming call handed it.
//
// The count matters as much as the text: a provider that produces "Hello" in
// one event and a provider that produces it as "Hel" then "lo" both end with
// the same text, and only the second is streaming. Tests that mean the second
// assert on `texts`.
class CollectingSink final : public ash::StreamSink {
public:
    void on_event(const ash::StreamEvent& event) override {
        if (const auto* delta = std::get_if<ash::TextDelta>(&event)) {
            texts.push_back(delta->text);
        }
        if (const auto* done = std::get_if<ash::StreamDone>(&event)) {
            finish_reason = done->finish_reason;
            saw_done = true;
        }
        events.push_back(event);
    }

    [[nodiscard]] std::string joined() const {
        std::string all;
        for (const std::string& piece : texts) {
            all += piece;
        }
        return all;
    }

    std::vector<ash::StreamEvent> events;
    std::vector<std::string> texts;
    std::string finish_reason;
    bool saw_done = false;
};

// A tool with no side effects, so tests stay hermetic.
inline std::shared_ptr<ash::Tool> echo_tool() {
    return ash::make_tool("echo", "Echo the given text back.", nlohmann::json{{"type", "object"}},
                          [](const nlohmann::json& arguments, std::stop_token) -> ash::Task<ash::ToolResult> {
                              ash::ToolResult result;
                              result.content = arguments.value("text", "");
                              co_return result;
                          });
}

inline ash::ToolResult call_tool(const ash::Tool& tool, nlohmann::json arguments) {
    return tool.invoke(arguments, std::stop_token{}).sync_wait();
}

}  // namespace ash::test
