#include "ash/model/stream.hpp"

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace ash {

namespace {

// Read-only access to an object's members that does not throw on a shape the
// endpoint was not supposed to send. This code sits on a network boundary: a
// truncated frame, a null where a string was expected, or a field that only
// exists in a newer version of the API must not be able to take the process
// down, and a stream that has already delivered half its answer is worth more
// than an exception.
[[nodiscard]] const nlohmann::json* member(const nlohmann::json& object, std::string_view key) {
    if (!object.is_object()) {
        return nullptr;
    }
    const auto found = object.find(key);
    return found == object.end() ? nullptr : &*found;
}

[[nodiscard]] std::string string_member(const nlohmann::json& object, std::string_view key) {
    const nlohmann::json* value = member(object, key);
    return value != nullptr && value->is_string() ? value->get<std::string>() : std::string{};
}

[[nodiscard]] int int_member(const nlohmann::json& object, std::string_view key) {
    const nlohmann::json* value = member(object, key);
    return value != nullptr && value->is_number_integer() ? value->get<int>() : 0;
}

// The usage block is spelled differently by the two dialects and is absent
// altogether on a compatible endpoint that does not implement it, so an
// unreported count stays zero rather than becoming an error.
[[nodiscard]] std::vector<StreamEvent> usage_event(const nlohmann::json& object,
                                                   std::string_view prompt_key,
                                                   std::string_view completion_key) {
    const nlohmann::json* usage = member(object, "usage");
    if (usage == nullptr || !usage->is_object()) {
        return {};
    }
    Usage counts;
    counts.prompt_tokens = int_member(*usage, prompt_key);
    counts.completion_tokens = int_member(*usage, completion_key);
    if (counts.prompt_tokens == 0 && counts.completion_tokens == 0) {
        return {};
    }
    return {UsageDelta{counts}};
}

constexpr std::string_view kOpenAiSentinel = "[DONE]";

void append(std::vector<StreamEvent>& into, std::vector<StreamEvent>&& extra) {
    for (StreamEvent& event : extra) {
        into.push_back(std::move(event));
    }
}

}  // namespace

namespace detail {

nlohmann::json parse_tool_arguments(const std::string& text) {
    if (text.empty()) {
        return nlohmann::json::object();
    }
    nlohmann::json parsed = nlohmann::json::parse(text, nullptr, false);
    return parsed.is_discarded() ? nlohmann::json::object() : parsed;
}

}  // namespace detail

void NullSink::on_event(const StreamEvent& /*event*/) {}

StreamAssembler::PartialCall& StreamAssembler::call_at(int index) {
    for (PartialCall& call : calls_) {
        if (call.index == index) {
            return call;
        }
    }
    // Assigned on first sight rather than placed at its index, because the
    // index is not always a position: Anthropic numbers content blocks, so a
    // response with text before a tool call starts that call at index 1.
    PartialCall fresh;
    fresh.index = index;
    calls_.push_back(std::move(fresh));
    return calls_.back();
}

void StreamAssembler::feed(const StreamEvent& event) {
    std::visit(
        [this](const auto& typed) {
            using Event = std::decay_t<decltype(typed)>;

            if constexpr (std::is_same_v<Event, TextDelta>) {
                text_ += typed.text;
            } else if constexpr (std::is_same_v<Event, ToolCallDelta>) {
                PartialCall& call = call_at(typed.index);
                if (!typed.id.empty()) {
                    call.id = typed.id;
                }
                if (!typed.name.empty()) {
                    call.name = typed.name;
                }
                call.arguments += typed.arguments_fragment;
            } else if constexpr (std::is_same_v<Event, UsageDelta>) {
                if (typed.usage.prompt_tokens > 0) {
                    usage_.prompt_tokens = typed.usage.prompt_tokens;
                }
                if (typed.usage.completion_tokens > 0) {
                    usage_.completion_tokens = typed.usage.completion_tokens;
                }
            } else if constexpr (std::is_same_v<Event, StreamDone>) {
                finish_reason_ = typed.finish_reason;
                model_ = typed.model;
            }
        },
        event);
}

ChatResponse StreamAssembler::take() {
    ChatResponse response;
    response.message.role = Role::kAssistant;
    response.message.content = std::move(text_);

    response.message.tool_calls.reserve(calls_.size());
    for (const PartialCall& partial : calls_) {
        ToolCall call;
        call.id = partial.id;
        call.name = partial.name;
        call.arguments = detail::parse_tool_arguments(partial.arguments);
        response.message.tool_calls.push_back(std::move(call));
    }

    response.usage = usage_;
    response.finish_reason = std::move(finish_reason_);
    response.model = std::move(model_);

    calls_.clear();
    text_.clear();
    usage_ = Usage{};
    return response;
}

std::vector<StreamEvent> events_for(const ChatResponse& response) {
    std::vector<StreamEvent> events;

    if (!response.message.content.empty()) {
        events.push_back(TextDelta{response.message.content});
    }

    for (std::size_t index = 0; index < response.message.tool_calls.size(); ++index) {
        const ToolCall& call = response.message.tool_calls[index];
        events.push_back(ToolCallDelta{static_cast<int>(index), call.id, call.name, call.arguments.dump()});
    }

    if (response.usage.prompt_tokens != 0 || response.usage.completion_tokens != 0) {
        events.push_back(UsageDelta{response.usage});
    }

    // Always last, and always present: a consumer can treat it as the end of the
    // response rather than as another thing that might not arrive.
    events.push_back(StreamDone{response.finish_reason, response.model});
    return events;
}

std::vector<StreamEvent> OpenAiStreamDecoder::feed(const SseFrame& frame) {
    std::vector<StreamEvent> events;

    if (frame.data == kOpenAiSentinel) {
        finished_ = true;
        return events;
    }

    const nlohmann::json chunk = nlohmann::json::parse(frame.data, nullptr, false);
    if (chunk.is_discarded()) {
        return events;
    }

    const std::string reported = string_member(chunk, "model");
    if (!reported.empty()) {
        model_ = reported;
    }

    // Only sent when the request asked for it with stream_options, and only on
    // the frame that carries the finish reason, which is why it is read here
    // rather than from a frame of its own.
    append(events, usage_event(chunk, "prompt_tokens", "completion_tokens"));

    const nlohmann::json* choices = member(chunk, "choices");
    if (choices == nullptr || !choices->is_array() || choices->empty()) {
        return events;
    }

    const nlohmann::json& choice = choices->front();
    const std::string reason = string_member(choice, "finish_reason");
    if (!reason.empty()) {
        finish_reason_ = reason;
    }

    const nlohmann::json* delta = member(choice, "delta");
    if (delta == nullptr) {
        return events;
    }

    const std::string text = string_member(*delta, "content");
    if (!text.empty()) {
        events.push_back(TextDelta{text});
    }

    const nlohmann::json* calls = member(*delta, "tool_calls");
    if (calls == nullptr || !calls->is_array()) {
        return events;
    }

    for (const nlohmann::json& call : *calls) {
        ToolCallDelta fragment;
        fragment.index = int_member(call, "index");
        fragment.id = string_member(call, "id");
        if (const nlohmann::json* function = member(call, "function"); function != nullptr) {
            fragment.name = string_member(*function, "name");
            fragment.arguments_fragment = string_member(*function, "arguments");
        }
        events.push_back(std::move(fragment));
    }
    return events;
}

std::vector<StreamEvent> OpenAiStreamDecoder::finish() {
    return {StreamDone{finish_reason_, model_.empty() ? fallback_model_ : model_}};
}

std::vector<StreamEvent> AnthropicStreamDecoder::feed(const SseFrame& frame) {
    std::vector<StreamEvent> events;

    const nlohmann::json payload = nlohmann::json::parse(frame.data, nullptr, false);
    if (payload.is_discarded()) {
        return events;
    }

    // The event name is repeated inside the payload. Preferring the framing is
    // deliberate: it is what the stream said the frame was, and a payload that
    // disagrees is a payload we should not be guessing at.
    const std::string type =
        frame.event.empty() ? string_member(payload, "type") : frame.event;

    if (type == "message_start") {
        const nlohmann::json* message = member(payload, "message");
        if (message == nullptr) {
            return events;
        }
        const std::string reported = string_member(*message, "model");
        if (!reported.empty()) {
            model_ = reported;
        }
        append(events, usage_event(*message, "input_tokens", "output_tokens"));
    } else if (type == "content_block_start") {
        const nlohmann::json* block = member(payload, "content_block");
        if (block == nullptr) {
            return events;
        }
        if (string_member(*block, "type") == "tool_use") {
            // The name and the id arrive here, before the arguments do, because
            // the endpoint has to open the block before it can stream into it.
            ToolCallDelta fragment;
            fragment.index = int_member(payload, "index");
            fragment.id = string_member(*block, "id");
            fragment.name = string_member(*block, "name");
            events.push_back(std::move(fragment));
        }
    } else if (type == "content_block_delta") {
        const nlohmann::json* delta = member(payload, "delta");
        if (delta == nullptr) {
            return events;
        }
        const std::string kind = string_member(*delta, "type");
        if (kind == "text_delta") {
            const std::string text = string_member(*delta, "text");
            if (!text.empty()) {
                events.push_back(TextDelta{text});
            }
        } else if (kind == "input_json_delta") {
            ToolCallDelta fragment;
            fragment.index = int_member(payload, "index");
            fragment.arguments_fragment = string_member(*delta, "partial_json");
            events.push_back(std::move(fragment));
        }
    } else if (type == "message_delta") {
        const nlohmann::json* delta = member(payload, "delta");
        if (delta != nullptr) {
            const std::string reason = string_member(*delta, "stop_reason");
            if (!reason.empty()) {
                finish_reason_ = reason;
            }
        }
        // The completion count is reported here and the prompt count was
        // reported at the start, which is why a usage delta is partial.
        append(events, usage_event(payload, "", "output_tokens"));
    } else if (type == "message_stop") {
        finished_ = true;
    }
    return events;
}

std::vector<StreamEvent> AnthropicStreamDecoder::finish() {
    return {StreamDone{finish_reason_, model_.empty() ? fallback_model_ : model_}};
}

}  // namespace ash
