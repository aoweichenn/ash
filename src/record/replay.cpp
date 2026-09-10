#include "ash/record/replay.hpp"

#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "ash/record/journal.hpp"

namespace ash {

namespace {

std::string kind_of(const Event& event) {
    return std::holds_alternative<ModelCallRecord>(event.payload) ? "model_call" : "tool_call";
}

}  // namespace

ReplayCursor::ReplayCursor(const Journal& journal, std::string actor) : actor_(std::move(actor)) {
    for (const auto& event : journal.events()) {
        if (event.actor == actor_) {
            events_.push_back(&event);
        }
    }
}

const Event& ReplayCursor::next_event(std::string_view expected_kind) const {
    if (consumed_ >= events_.size()) {
        throw ReplayError{"replay ran past the end of the journal: actor '" + actor_ + "' made a " +
                          std::string{expected_kind} + " call after all " + std::to_string(events_.size()) +
                          " recorded events were consumed"};
    }

    const Event& event = *events_[consumed_];
    const std::string actual_kind = kind_of(event);
    if (actual_kind != expected_kind) {
        throw ReplayError{"replay diverged: actor '" + actor_ + "' seq " + std::to_string(event.seq) +
                          " recorded a " + actual_kind + " but the run asked for a " +
                          std::string{expected_kind}};
    }
    return event;
}

const ChatResponse& ReplayCursor::next_model_call(const ChatRequest& request) {
    const Event& event = next_event("model_call");
    const auto& record = std::get<ModelCallRecord>(event.payload);

    if (!(record.request == request)) {
        throw ReplayError{"replay diverged at actor '" + actor_ + "' seq " + std::to_string(event.seq) +
                          ": the run asked a different question than the recording\n  recorded req_hash " +
                          request_hash(record.request) + "\n  replayed req_hash " + request_hash(request)};
    }

    ++consumed_;
    return record.response;
}

const ToolResult& ReplayCursor::next_tool_call(std::string_view name, const nlohmann::json& arguments) {
    const Event& event = next_event("tool_call");
    const auto& record = std::get<ToolCallRecord>(event.payload);

    if (record.name != name || !(record.arguments == arguments)) {
        throw ReplayError{"replay diverged at actor '" + actor_ + "' seq " + std::to_string(event.seq) +
                          ": the run called a different tool than the recording\n  recorded tool_hash " +
                          tool_call_hash(record.name, record.arguments) +
                          "\n  replayed tool_hash " + tool_call_hash(name, arguments)};
    }

    ++consumed_;
    return record.result;
}

void ReplayCursor::verify_consumed() const {
    if (consumed_ != events_.size()) {
        throw ReplayError{"replay finished after " + std::to_string(consumed_) + " of " +
                          std::to_string(events_.size()) + " recorded events for actor '" + actor_ +
                          "': the run took a different path than the recording"};
    }
}

std::vector<ToolSpec> recorded_tool_specs(const Journal& journal, std::string_view actor) {
    for (const auto& event : journal.events()) {
        if (event.actor != actor) {
            continue;
        }
        if (const auto* model_call = std::get_if<ModelCallRecord>(&event.payload)) {
            return model_call->request.tools;
        }
    }
    return {};
}

}  // namespace ash
