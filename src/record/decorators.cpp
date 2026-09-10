#include "ash/record/decorators.hpp"

#include <memory>
#include <stop_token>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace ash {

RecordingProvider::RecordingProvider(std::unique_ptr<ModelProvider> inner, Journal& journal, std::string actor)
    : inner_(std::move(inner)), journal_(journal), actor_(std::move(actor)) {}

std::string_view RecordingProvider::name() const noexcept { return inner_->name(); }

const std::string& RecordingProvider::model() const noexcept { return inner_->model(); }

Task<ChatResponse> RecordingProvider::chat(ChatRequest request) {
    // Recorded only once the call succeeds, so a journal never holds a request
    // without its answer. The API key lives in the provider's config, which is
    // never part of a ChatRequest, so it cannot reach the journal.
    ChatResponse response = co_await inner_->chat(request);
    journal_.append(actor_, ModelCallRecord{request, response});
    co_return std::move(response);
}

ReplayingProvider::ReplayingProvider(ReplayCursor& cursor, std::string name, std::string model)
    : cursor_(cursor), name_(std::move(name)), model_(std::move(model)) {}

std::string_view ReplayingProvider::name() const noexcept { return name_; }

const std::string& ReplayingProvider::model() const noexcept { return model_; }

Task<ChatResponse> ReplayingProvider::chat(ChatRequest request) {
    co_return cursor_.next_model_call(request);
}

RecordingTool::RecordingTool(std::shared_ptr<Tool> inner, Journal& journal, std::string actor)
    : inner_(std::move(inner)), journal_(journal), actor_(std::move(actor)) {}

std::string_view RecordingTool::name() const noexcept { return inner_->name(); }

std::string_view RecordingTool::description() const noexcept { return inner_->description(); }

const nlohmann::json& RecordingTool::input_schema() const noexcept { return inner_->input_schema(); }

Task<ToolResult> RecordingTool::invoke(const nlohmann::json& arguments, std::stop_token stop) const {
    ToolResult result = co_await inner_->invoke(arguments, stop);
    journal_.append(actor_, ToolCallRecord{std::string{name()}, arguments, result});
    co_return result;
}

ReplayingTool::ReplayingTool(ReplayCursor& cursor, ToolSpec spec) : cursor_(cursor), spec_(std::move(spec)) {}

std::string_view ReplayingTool::name() const noexcept { return spec_.name; }

std::string_view ReplayingTool::description() const noexcept { return spec_.description; }

const nlohmann::json& ReplayingTool::input_schema() const noexcept { return spec_.input_schema; }

Task<ToolResult> ReplayingTool::invoke(const nlohmann::json& arguments, std::stop_token) const {
    co_return cursor_.next_tool_call(spec_.name, arguments);
}

ToolRegistry make_recording_registry(const ToolRegistry& tools, Journal& journal, std::string actor) {
    ToolRegistry recording;
    for (const auto& tool : tools.tools()) {
        // The wrapper co-owns the real tool, so the recording registry stays
        // valid for as long as it needs to.
        recording.add(std::make_shared<RecordingTool>(tool, journal, actor));
    }
    return recording;
}

ToolRegistry make_replaying_registry(const Journal& journal, ReplayCursor& cursor, std::string_view actor) {
    ToolRegistry replaying;
    for (const auto& spec : recorded_tool_specs(journal, actor)) {
        replaying.add(std::make_shared<ReplayingTool>(cursor, spec));
    }
    return replaying;
}

}  // namespace ash
