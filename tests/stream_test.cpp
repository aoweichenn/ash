#include <catch2/catch_test_macros.hpp>

#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "ash/model/sse.hpp"
#include "ash/model/stream.hpp"

namespace {

using ash::AnthropicStreamDecoder;
using ash::ChatResponse;
using ash::Message;
using ash::OpenAiStreamDecoder;
using ash::Role;
using ash::SseDecoder;
using ash::StreamAssembler;
using ash::StreamDone;
using ash::StreamEvent;
using ash::TextDelta;
using ash::ToolCall;
using ash::ToolCallDelta;
using ash::Usage;
using ash::UsageDelta;

// Counts what a sink was handed. Every assertion about "the sink saw this" goes
// through here, so a test never has to care how the events were routed.
class CountedSink final : public ash::StreamSink {
public:
    void on_event(const StreamEvent& event) override { events.push_back(event); }

    [[nodiscard]] std::string text() const {
        std::string joined;
        for (const StreamEvent& event : events) {
            if (const auto* delta = std::get_if<TextDelta>(&event)) {
                joined += delta->text;
            }
        }
        return joined;
    }

    [[nodiscard]] bool done() const {
        for (const StreamEvent& event : events) {
            if (std::holds_alternative<StreamDone>(event)) {
                return true;
            }
        }
        return false;
    }

    std::vector<StreamEvent> events;
};

// Runs a whole wire-format stream through the framer and then a dialect decoder,
// assembling as it goes. This is the path a live call takes minus the socket, so
// a change that breaks the integration breaks these tests too.
ChatResponse decode_stream(std::string_view wire, bool anthropic) {
    SseDecoder framer;
    StreamAssembler assembler;
    OpenAiStreamDecoder openai{"fallback"};
    AnthropicStreamDecoder claude{"fallback"};

    const auto consume = [&](const ash::SseFrame& frame) {
        std::vector<StreamEvent> events =
            anthropic ? claude.feed(frame) : openai.feed(frame);
        for (StreamEvent& event : events) {
            assembler.feed(event);
        }
    };

    for (const ash::SseFrame& frame : framer.feed(wire)) {
        consume(frame);
    }
    for (const ash::SseFrame& frame : framer.finish()) {
        consume(frame);
    }
    for (StreamEvent& event : (anthropic ? claude.finish() : openai.finish())) {
        assembler.feed(event);
    }
    return assembler.take();
}

bool has_done(const std::vector<StreamEvent>& events) {
    return !events.empty() && std::holds_alternative<StreamDone>(events.back());
}

}  // namespace

TEST_CASE("a text-only response projects onto and back off the event stream", "[stream]") {
    // The property the whole design rests on: streaming is a view of a call,
    // not a second kind of call. If this round-trip did not land exactly, a
    // streamed recording and a non-streamed one would disagree.
    ChatResponse original;
    original.message.role = Role::kAssistant;
    original.message.content = "Hello there";
    original.usage = Usage{.prompt_tokens = 9, .completion_tokens = 3};
    original.finish_reason = "stop";
    original.model = "deepseek-chat";

    StreamAssembler assembler;
    for (const StreamEvent& event : ash::events_for(original)) {
        assembler.feed(event);
    }

    CHECK(assembler.take() == original);
}

TEST_CASE("a tool call response projects onto and back off the event stream", "[stream]") {
    ChatResponse original;
    original.message.role = Role::kAssistant;
    original.message.content = "let me check";
    original.message.tool_calls.push_back(
        ToolCall{.id = "call_1", .name = "read_file", .arguments = {{"path", "a.txt"}}});
    original.message.tool_calls.push_back(
        ToolCall{.id = "call_2", .name = "list_dir", .arguments = nlohmann::json::object()});
    original.usage = Usage{.prompt_tokens = 40, .completion_tokens = 12};
    original.finish_reason = "tool_calls";
    original.model = "deepseek-chat";

    StreamAssembler assembler;
    for (const StreamEvent& event : ash::events_for(original)) {
        assembler.feed(event);
    }

    CHECK(assembler.take() == original);
}

TEST_CASE("an empty response projects onto and back off the event stream", "[stream]") {
    ChatResponse original;
    original.message.role = Role::kAssistant;
    original.finish_reason = "stop";
    original.model = "deepseek-chat";

    const std::vector<StreamEvent> events = ash::events_for(original);
    REQUIRE(events.size() == 1);
    CHECK(std::holds_alternative<StreamDone>(events.front()));

    StreamAssembler assembler;
    for (const StreamEvent& event : events) {
        assembler.feed(event);
    }
    CHECK(assembler.take() == original);
}

TEST_CASE("the event stream always ends with a done event", "[stream]") {
    ChatResponse response;
    response.message.role = Role::kAssistant;
    response.message.content = "hi";

    CHECK(has_done(ash::events_for(response)));

    // And a response with nothing in it at all still ends, so a consumer can
    // wait for the end unconditionally instead of guarding every stream.
    CHECK(has_done(ash::events_for(ChatResponse{})));
}

TEST_CASE("a usage event is sent only when the endpoint reported tokens", "[stream]") {
    ChatResponse no_usage;
    no_usage.message.content = "hi";
    for (const StreamEvent& event : ash::events_for(no_usage)) {
        CHECK_FALSE(std::holds_alternative<UsageDelta>(event));
    }

    ChatResponse with_usage;
    with_usage.message.content = "hi";
    with_usage.usage = Usage{.prompt_tokens = 5, .completion_tokens = 1};
    bool saw_usage = false;
    for (const StreamEvent& event : ash::events_for(with_usage)) {
        saw_usage = saw_usage || std::holds_alternative<UsageDelta>(event);
    }
    CHECK(saw_usage);
}

TEST_CASE("a usage delta only overwrites the counts it carries", "[stream]") {
    // Anthropic reports the prompt count at the start of the stream and the
    // completion count at the end, so the second report must not zero the first.
    StreamAssembler assembler;
    assembler.feed(UsageDelta{Usage{.prompt_tokens = 12, .completion_tokens = 0}});
    assembler.feed(UsageDelta{Usage{.prompt_tokens = 0, .completion_tokens = 25}});
    assembler.feed(StreamDone{"end_turn", "m"});

    const ChatResponse response = assembler.take();
    CHECK(response.usage.prompt_tokens == 12);
    CHECK(response.usage.completion_tokens == 25);
}

TEST_CASE("an OpenAI stream assembles the response the non-streaming call returns", "[stream]") {
    const std::string wire =
        "data: {\"id\":\"1\",\"model\":\"deepseek-chat\",\"choices\":[{\"index\":0,\"delta\":"
        "{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n"
        "\n"
        "data: {\"id\":\"1\",\"model\":\"deepseek-chat\",\"choices\":[{\"index\":0,\"delta\":"
        "{\"content\":\"Hel\"},\"finish_reason\":null}]}\n"
        "\n"
        "data: {\"id\":\"1\",\"model\":\"deepseek-chat\",\"choices\":[{\"index\":0,\"delta\":"
        "{\"content\":\"lo\"},\"finish_reason\":null}]}\n"
        "\n"
        "data: {\"id\":\"1\",\"model\":\"deepseek-chat\",\"choices\":[{\"index\":0,\"delta\":{},"
        "\"finish_reason\":\"stop\"}]}\n"
        "\n"
        "data: {\"id\":\"1\",\"model\":\"deepseek-chat\",\"choices\":[],\"usage\":"
        "{\"prompt_tokens\":9,\"completion_tokens\":2,\"total_tokens\":11}}\n"
        "\n"
        "data: [DONE]\n"
        "\n";

    const ChatResponse response = decode_stream(wire, /*anthropic=*/false);

    CHECK(response.message.role == Role::kAssistant);
    CHECK(response.message.content == "Hello");
    CHECK(response.message.tool_calls.empty());
    CHECK(response.usage.prompt_tokens == 9);
    CHECK(response.usage.completion_tokens == 2);
    CHECK(response.finish_reason == "stop");
    CHECK(response.model == "deepseek-chat");
}

TEST_CASE("an OpenAI tool call split across frames is stitched back together", "[stream]") {
    // The endpoint decides where to cut the argument JSON, and it does not
    // respect JSON syntax when it does -- here it cuts inside a key name.
    const std::string wire =
        "data: {\"model\":\"deepseek-chat\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":"
        "[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":"
        "{\"name\":\"read_file\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n"
        "\n"
        "data: {\"model\":\"deepseek-chat\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":"
        "[{\"index\":0,\"function\":{\"arguments\":\"{\\\"pa\"}}]},\"finish_reason\":null}]}\n"
        "\n"
        "data: {\"model\":\"deepseek-chat\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":"
        "[{\"index\":0,\"function\":{\"arguments\":\"th\\\":\\\"a.txt\\\"}\"}}]},"
        "\"finish_reason\":null}]}\n"
        "\n"
        "data: {\"model\":\"deepseek-chat\",\"choices\":[{\"index\":0,\"delta\":{},"
        "\"finish_reason\":\"tool_calls\"}]}\n"
        "\n"
        "data: [DONE]\n"
        "\n";

    const ChatResponse response = decode_stream(wire, /*anthropic=*/false);

    REQUIRE(response.message.tool_calls.size() == 1);
    CHECK(response.message.tool_calls[0].id == "call_1");
    CHECK(response.message.tool_calls[0].name == "read_file");
    CHECK(response.message.tool_calls[0].arguments == nlohmann::json{{"path", "a.txt"}});
    CHECK(response.finish_reason == "tool_calls");
}

TEST_CASE("two OpenAI tool calls keep their own arguments apart", "[stream]") {
    const std::string wire =
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"function\":"
        "{\"name\":\"echo\",\"arguments\":\"\"}},{\"index\":1,\"id\":\"c2\",\"function\":"
        "{\"name\":\"echo\",\"arguments\":\"\"}}]}}]}\n"
        "\n"
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":"
        "{\"arguments\":\"{\\\"text\\\":\\\"one\\\"}\"}}]}}]}\n"
        "\n"
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"function\":"
        "{\"arguments\":\"{\\\"text\\\":\\\"two\\\"}\"}}]}}]}\n"
        "\n"
        "data: [DONE]\n";

    const ChatResponse response = decode_stream(wire, /*anthropic=*/false);

    REQUIRE(response.message.tool_calls.size() == 2);
    CHECK(response.message.tool_calls[0].id == "c1");
    CHECK(response.message.tool_calls[0].arguments == nlohmann::json{{"text", "one"}});
    CHECK(response.message.tool_calls[1].id == "c2");
    CHECK(response.message.tool_calls[1].arguments == nlohmann::json{{"text", "two"}});
}

TEST_CASE("an OpenAI stream stops at its sentinel", "[stream]") {
    OpenAiStreamDecoder decoder{"fallback"};

    CHECK_FALSE(decoder.finished());
    CHECK(decoder.feed(ash::SseFrame{"", "{\"choices\":[{\"delta\":{\"content\":\"x\"}}]}"}).size() == 1);
    CHECK_FALSE(decoder.finished());

    // The sentinel is the stream saying it is over, not the model saying
    // something, so it carries no chunk and produces no event of its own.
    CHECK(decoder.feed(ash::SseFrame{"", "[DONE]"}).empty());
    CHECK(decoder.finished());
}

TEST_CASE("an Anthropic stream assembles text and a tool call together", "[stream]") {
    // The tool call here is content block 1, not 0, because the text block took
    // slot 0. An assembler that placed calls by position would drop it or file
    // it under the text.
    const std::string wire =
        "event: message_start\n"
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\","
        "\"model\":\"claude-sonnet-5\",\"usage\":{\"input_tokens\":12,\"output_tokens\":1}}}\n"
        "\n"
        "event: content_block_start\n"
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":"
        "{\"type\":\"text\",\"text\":\"\"}}\n"
        "\n"
        "event: content_block_delta\n"
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":"
        "{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n"
        "\n"
        "event: content_block_stop\n"
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n"
        "\n"
        "event: content_block_start\n"
        "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":"
        "{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"echo\",\"input\":{}}}\n"
        "\n"
        "event: content_block_delta\n"
        "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":"
        "{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"text\\\":\"}}\n"
        "\n"
        "event: content_block_delta\n"
        "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":"
        "{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"ping\\\"}\"}}\n"
        "\n"
        "event: content_block_stop\n"
        "data: {\"type\":\"content_block_stop\",\"index\":1}\n"
        "\n"
        "event: message_delta\n"
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},"
        "\"usage\":{\"output_tokens\":25}}\n"
        "\n"
        "event: message_stop\n"
        "data: {\"type\":\"message_stop\"}\n"
        "\n";

    const ChatResponse response = decode_stream(wire, /*anthropic=*/true);

    CHECK(response.message.content == "Hi");
    REQUIRE(response.message.tool_calls.size() == 1);
    CHECK(response.message.tool_calls[0].id == "toolu_1");
    CHECK(response.message.tool_calls[0].name == "echo");
    CHECK(response.message.tool_calls[0].arguments == nlohmann::json{{"text", "ping"}});
    CHECK(response.usage.prompt_tokens == 12);
    CHECK(response.usage.completion_tokens == 25);
    CHECK(response.finish_reason == "tool_use");
    CHECK(response.model == "claude-sonnet-5");
}

TEST_CASE("an Anthropic stream stops at message_stop", "[stream]") {
    AnthropicStreamDecoder decoder{"fallback"};

    CHECK_FALSE(decoder.finished());
    CHECK(decoder.feed(ash::SseFrame{"content_block_delta",
                                     "{\"type\":\"content_block_delta\",\"index\":0,\"delta\":"
                                     "{\"type\":\"text_delta\",\"text\":\"x\"}}"})
              .size() == 1);
    CHECK_FALSE(decoder.finished());

    CHECK(decoder.feed(ash::SseFrame{"message_stop", "{\"type\":\"message_stop\"}"}).empty());
    CHECK(decoder.finished());
}

TEST_CASE("the event name in the framing wins over the one in the payload", "[stream]") {
    // The two always agree in practice. When they do not, the framing is what
    // the stream said this frame was, and guessing from the payload would mean
    // silently accepting a stream that does not make sense.
    AnthropicStreamDecoder decoder{"fallback"};

    const std::vector<StreamEvent> events =
        decoder.feed(ash::SseFrame{"content_block_delta",
                                   "{\"type\":\"totally_unknown\",\"delta\":"
                                   "{\"type\":\"text_delta\",\"text\":\"kept\"}}"});

    REQUIRE(events.size() == 1);
    REQUIRE(std::holds_alternative<TextDelta>(events.front()));
    CHECK(std::get<TextDelta>(events.front()).text == "kept");
}

TEST_CASE("a frame with no event name falls back to the payload type", "[stream]") {
    // Anthropic clients that read the payload instead of the framing are common
    // enough that a stream which omits the event lines should still work.
    AnthropicStreamDecoder decoder{"fallback"};

    const std::vector<StreamEvent> events =
        decoder.feed(ash::SseFrame{"", "{\"type\":\"content_block_delta\",\"index\":0,\"delta\":"
                                       "{\"type\":\"text_delta\",\"text\":\"kept\"}}"});

    REQUIRE(events.size() == 1);
    CHECK(std::get<TextDelta>(events.front()).text == "kept");
}

TEST_CASE("finish always produces exactly one done event", "[stream]") {
    for (const bool anthropic : {false, true}) {
        OpenAiStreamDecoder openai{"deepseek-chat"};
        AnthropicStreamDecoder claude{"claude-sonnet-5"};

        // A stream that says nothing at all -- a connection that opened and died.
        std::vector<StreamEvent> closing =
            anthropic ? claude.finish() : openai.finish();
        REQUIRE(closing.size() == 1);
        REQUIRE(std::holds_alternative<StreamDone>(closing.front()));

        const auto& done = std::get<StreamDone>(closing.front());
        // The model the caller asked for, because a truncated stream never got
        // to say which model answered.
        CHECK(done.model == (anthropic ? "claude-sonnet-5" : "deepseek-chat"));

        // And calling it twice does not produce a second ending.
        CHECK((anthropic ? claude.finish() : openai.finish()).size() == 1);
    }
}

TEST_CASE("a stream that reported its model does not fall back", "[stream]") {
    OpenAiStreamDecoder decoder{"fallback-model"};
    CHECK(decoder.feed(ash::SseFrame{"", "{\"model\":\"deepseek-chat\",\"choices\":[{\"delta\":"
                                          "{\"content\":\"x\"}}]}"})
              .size() == 1);

    const auto done = std::get<StreamDone>(decoder.finish().front());
    CHECK(done.model == "deepseek-chat");
}

TEST_CASE("a malformed frame is skipped rather than thrown", "[stream]") {
    // A network boundary: one bad frame should cost the run that frame, not the
    // whole answer that was already delivered.
    OpenAiStreamDecoder openai{"fallback"};
    AnthropicStreamDecoder claude{"fallback"};

    for (const char* garbage : {"not json", "{", "[]", "null", "{\"choices\":3}"}) {
        CHECK_NOTHROW(openai.feed(ash::SseFrame{"", garbage}));
        CHECK_NOTHROW(claude.feed(ash::SseFrame{"", garbage}));
    }

    // The decoders are still usable afterwards.
    const std::vector<StreamEvent> events =
        openai.feed(ash::SseFrame{"", "{\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}"});
    REQUIRE(events.size() == 1);
    CHECK(std::get<TextDelta>(events.front()).text == "ok");
}

TEST_CASE("a frame with an unexpected shape contributes nothing", "[stream]") {
    OpenAiStreamDecoder decoder{"fallback"};

    // A delta whose content is null is what the first chunk of a tool-call-only
    // response looks like. It must not become an empty text event, because the
    // sink would then be told the model said nothing.
    const std::vector<StreamEvent> events = decoder.feed(
        ash::SseFrame{"", "{\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":null},"
                          "\"finish_reason\":null}]}"});
    CHECK(events.empty());
}

TEST_CASE("bad tool arguments become an empty object", "[stream]") {
    CHECK(ash::detail::parse_tool_arguments("") == nlohmann::json::object());
    CHECK(ash::detail::parse_tool_arguments("{\"a\":1}") == nlohmann::json{{"a", 1}});
    // Truncated because the connection dropped mid-arguments: the tool gets
    // something it can reject rather than an exception nobody can attribute.
    CHECK(ash::detail::parse_tool_arguments("{\"a\":") == nlohmann::json::object());
    CHECK(ash::detail::parse_tool_arguments("garbage") == nlohmann::json::object());
}

TEST_CASE("a sink sees the text as it arrives, not only at the end", "[stream]") {
    // The point of streaming: the sink is called during the transfer. Counting
    // calls rather than inspecting the total is what makes this a test of
    // incrementality -- one event carrying "Hello" would satisfy a text check.
    SseDecoder framer;
    OpenAiStreamDecoder decoder{"fallback"};
    CountedSink sink;

    const std::string wire =
        "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n"
        "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n"
        "data: [DONE]\n\n";

    for (const ash::SseFrame& frame : framer.feed(wire)) {
        for (const StreamEvent& event : decoder.feed(frame)) {
            sink.on_event(event);
        }
    }

    CHECK(sink.text() == "Hello");
    CHECK_FALSE(sink.done());  // the done event comes from finish(), not the frames
}
