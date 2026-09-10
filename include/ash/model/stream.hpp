#pragma once

#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "ash/model/provider.hpp"
#include "ash/model/sse.hpp"

namespace ash {

// What a model call reports while it is still in flight.
//
// The vocabulary is the intersection of the two dialects, not their union.
// Everything a streamed response can say that an assembled one cannot -- which
// frame it arrived in, how long a pause preceded it, how the text was split --
// is deliberately absent, because none of it survives into the journal and a
// type that carried it would invite code that depended on it.
struct TextDelta {
    std::string text;

    friend bool operator==(const TextDelta&, const TextDelta&) = default;
};

// A fragment of a tool call. The endpoint decides where to split the arguments,
// so the pieces are only meaningful once concatenated in index order.
struct ToolCallDelta {
    int index = 0;                     // which call of the response this belongs to
    std::string id;                    // sent on the first fragment, empty afterwards
    std::string name;                  // sent on the first fragment, empty afterwards
    std::string arguments_fragment;    // JSON text, split wherever the endpoint split it

    friend bool operator==(const ToolCallDelta&, const ToolCallDelta&) = default;
};

// A partial usage report. A field left at zero means the endpoint has not
// reported it yet rather than that it is zero, because the two dialects report
// the two counts in different frames: one sends both at the end, the other sends
// the prompt count at the start and the completion count at the end.
struct UsageDelta {
    Usage usage;

    friend bool operator==(const UsageDelta&, const UsageDelta&) = default;
};

// The last event of a stream, sent exactly once.
struct StreamDone {
    std::string finish_reason;
    std::string model;

    friend bool operator==(const StreamDone&, const StreamDone&) = default;
};

using StreamEvent = std::variant<TextDelta, ToolCallDelta, UsageDelta, StreamDone>;

// Where a streaming call delivers its events.
//
// on_event is called on the thread performing the transfer, from inside
// libcurl's write callback, and it is called synchronously: libcurl will not
// read another byte of the body until it returns. That is the entire
// backpressure story. A slow sink stops draining the socket, the socket stops
// accepting, and the endpoint's own flow control slows the model down -- so
// there is no queue here to bound, because nothing is queued. The cost is the
// rule that follows from being inside a C callback: on_event must not let an
// exception escape, and must not block indefinitely.
struct StreamSink {
    virtual ~StreamSink() = default;

    virtual void on_event(const StreamEvent& event) = 0;
};

// A sink that drops everything, for a caller that wants the answer and not the
// play-by-play.
class NullSink final : public StreamSink {
public:
    void on_event(const StreamEvent& event) override;
};

// Folds a stream back into the single response the non-streaming call returns.
//
// This is what keeps streaming from being a second kind of call. The journal
// records the assembled response, so a run that streamed and a run that did not
// write byte-identical entries, and a replay cannot tell which produced it --
// which is only true if the fold lands on exactly the same value the decoder
// lands on, so the two are checked against each other rather than assumed equal.
class StreamAssembler {
public:
    void feed(const StreamEvent& event);

    // Takes the response. Call once: the pieces are moved out, and an assembler
    // that has given its answer away has none left.
    [[nodiscard]] ChatResponse take();

private:
    struct PartialCall {
        int index = 0;
        std::string id;
        std::string name;
        std::string arguments;
    };

    [[nodiscard]] PartialCall& call_at(int index);

    std::string text_;
    std::vector<PartialCall> calls_;
    Usage usage_;
    std::string finish_reason_;
    std::string model_;
};

// Projects an assembled response back onto the event stream.
//
// A replay has no timing to reproduce -- it has the answer already -- so the
// events arrive in one burst at the end. They are still the same events, in the
// same order, which is what lets one sink render a live run and a replay without
// knowing which it is looking at.
[[nodiscard]] std::vector<StreamEvent> events_for(const ChatResponse& response);

// Translates one dialect's frames into events.
//
// The dialects differ in ways the events do not: OpenAI puts everything in one
// chunk shape and terminates with a sentinel, while Anthropic names its frames
// with `event:` lines, opens a content block before sending fragments into it,
// and reports usage across two separate messages. Confining that difference to
// these two classes is what keeps the agent loop and the journal free of it.
class OpenAiStreamDecoder {
public:
    explicit OpenAiStreamDecoder(std::string fallback_model) : fallback_model_(std::move(fallback_model)) {}

    [[nodiscard]] std::vector<StreamEvent> feed(const SseFrame& frame);

    // True once the stream said it was over. The caller stops reading when it
    // is, so a server that keeps the connection alive after the sentinel does
    // not hold the run open.
    [[nodiscard]] bool finished() const noexcept { return finished_; }

    // The closing event, exactly once, whether or not the stream reached its
    // sentinel: a truncated answer still has an end.
    [[nodiscard]] std::vector<StreamEvent> finish();

private:
    std::string fallback_model_;
    std::string finish_reason_;
    std::string model_;
    bool finished_ = false;
};

class AnthropicStreamDecoder {
public:
    explicit AnthropicStreamDecoder(std::string fallback_model) : fallback_model_(std::move(fallback_model)) {}

    [[nodiscard]] std::vector<StreamEvent> feed(const SseFrame& frame);

    [[nodiscard]] bool finished() const noexcept { return finished_; }

    [[nodiscard]] std::vector<StreamEvent> finish();

private:
    std::string fallback_model_;
    std::string finish_reason_;
    std::string model_;
    bool finished_ = false;
};

namespace detail {

// Shared by the streaming assembler and the OpenAI-compatible non-streaming
// decode, so the two cannot drift on how they read a tool-call argument dump.
// A dump that does not parse is the model's mistake, not a transport failure,
// and the tool gets an empty object to complain about.
[[nodiscard]] nlohmann::json parse_tool_arguments(const std::string& text);

}  // namespace detail

}  // namespace ash
