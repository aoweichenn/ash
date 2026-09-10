#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <stop_token>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "ash/cancellation.hpp"
#include "ash/runtime.hpp"
#include "ash/tool/builtin.hpp"
#include "ash/tool/tool.hpp"
#include "test_support.hpp"

using namespace ash::test;

namespace {

// A model call cut off partway through, which is what a stop request looks like
// to the loop once the transport has noticed it.
class InterruptedProvider final : public ash::ModelProvider {
public:
    [[nodiscard]] std::string_view name() const noexcept override { return "interrupted"; }
    [[nodiscard]] const std::string& model() const noexcept override { return model_; }

    ash::Task<ash::ChatResponse> chat(ash::ChatRequest) override { throw ash::Cancelled{}; }

    ash::Task<ash::ChatResponse> chat_stream(ash::ChatRequest,
                                             ash::StreamSink& sink,
                                             std::stop_token) override {
        sink.on_event(ash::TextDelta{"par"});
        sink.on_event(ash::TextDelta{"tial"});
        throw ash::Cancelled{};
    }

private:
    std::string model_ = "scripted-model";
};

}  // namespace

TEST_CASE("agent loop returns as soon as the model stops calling tools") {
    ScriptedProvider provider{{reply(assistant_text("all done"))}};
    const ash::ToolRegistry tools;

    const auto result = ash::run_agent(provider, tools, "say hello").sync_wait();

    CHECK(result.stop_reason == "completed");
    CHECK(result.steps.empty());
    REQUIRE(result.transcript.size() == 3);
    CHECK(result.transcript[0].role == ash::Role::kSystem);
    CHECK(result.transcript[1].role == ash::Role::kUser);
    CHECK(result.transcript[1].content == "say hello");
    CHECK(result.transcript[2].content == "all done");
    CHECK(provider.requests.size() == 1);
}

TEST_CASE("agent loop advertises registered tools to the model") {
    ScriptedProvider provider{{reply(assistant_text("ok"))}};
    ash::ToolRegistry tools;
    tools.add(echo_tool());

    const auto result = ash::run_agent(provider, tools, "hi").sync_wait();

    REQUIRE(result.stop_reason == "completed");
    REQUIRE(provider.requests.size() == 1);
    REQUIRE(provider.requests[0].tools.size() == 1);
    CHECK(provider.requests[0].tools[0].name == "echo");
}

TEST_CASE("agent loop runs a tool and feeds the result back") {
    ScriptedProvider provider{{reply(assistant_tool_call("call-1", "echo", {{"text", "ping"}})),
                               reply(assistant_text("got it"))}};
    ash::ToolRegistry tools;
    tools.add(echo_tool());

    const auto result = ash::run_agent(provider, tools, "echo ping").sync_wait();

    CHECK(result.stop_reason == "completed");
    REQUIRE(result.steps.size() == 1);
    CHECK(result.steps[0].assistant.tool_calls.size() == 1);
    REQUIRE(result.steps[0].tool_results.size() == 1);
    CHECK(result.steps[0].tool_results[0].content == "ping");

    REQUIRE(result.transcript.size() == 5);
    CHECK(result.transcript[3].role == ash::Role::kTool);
    CHECK(result.transcript[3].content == "ping");
    CHECK(result.transcript[3].tool_call_id == "call-1");
    CHECK(result.transcript[4].content == "got it");

    // The second request must carry the assistant tool call and its result.
    REQUIRE(provider.requests.size() == 2);
    REQUIRE(provider.requests[1].messages.size() == 4);
    CHECK(provider.requests[1].messages[2].tool_calls[0].name == "echo");
    CHECK(provider.requests[1].messages[3].tool_call_id == "call-1");
}

TEST_CASE("an unknown tool becomes an error result instead of an exception") {
    ScriptedProvider provider{{reply(assistant_tool_call("call-1", "nope", nlohmann::json::object())),
                               reply(assistant_text("recovered"))}};
    const ash::ToolRegistry tools;

    const auto result = ash::run_agent(provider, tools, "call nope").sync_wait();

    CHECK(result.stop_reason == "completed");
    REQUIRE(result.transcript.size() == 5);
    CHECK(result.transcript[3].content == "unknown tool: nope");
}

TEST_CASE("a failing tool is handed back to the model, not thrown") {
    ScriptedProvider provider{{reply(assistant_tool_call("call-1", "read_file", {{"path", "/no/such/file"}})),
                               reply(assistant_text("could not read it"))}};
    ash::ToolRegistry tools;
    tools.add(ash::make_read_file_tool());

    const auto result = ash::run_agent(provider, tools, "read it").sync_wait();

    CHECK(result.stop_reason == "completed");
    REQUIRE(result.transcript.size() == 5);
    CHECK(result.transcript[3].content.find("read_file:") == 0);
}

TEST_CASE("agent loop stops at the step budget") {
    ScriptedProvider provider{{reply(assistant_tool_call("c1", "echo", {{"text", "1"}})),
                               reply(assistant_tool_call("c2", "echo", {{"text", "2"}})),
                               reply(assistant_tool_call("c3", "echo", {{"text", "3"}}))}};
    ash::ToolRegistry tools;
    tools.add(echo_tool());

    ash::AgentOptions options;
    options.max_steps = 2;

    const auto result = ash::run_agent(provider, tools, "loop forever", options).sync_wait();

    CHECK(result.stop_reason == "max_steps");
    CHECK(result.steps.size() == 2);
    CHECK(provider.requests.size() == 2);
}

TEST_CASE("usage accumulates across steps") {
    ash::Usage first;
    first.prompt_tokens = 10;
    first.completion_tokens = 5;
    ash::Usage second;
    second.prompt_tokens = 20;
    second.completion_tokens = 7;

    ScriptedProvider provider{{reply(assistant_tool_call("c1", "echo", {{"text", "1"}}), first),
                               reply(assistant_text("done"), second)}};
    ash::ToolRegistry tools;
    tools.add(echo_tool());

    const auto result = ash::run_agent(provider, tools, "count tokens").sync_wait();

    CHECK(result.usage.prompt_tokens == 30);
    CHECK(result.usage.completion_tokens == 12);
    CHECK(result.usage.total_tokens() == 42);
}

TEST_CASE("an already-cancelled run never calls the model") {
    ScriptedProvider provider{{reply(assistant_text("unreachable"))}};
    const ash::ToolRegistry tools;

    std::stop_source source;
    source.request_stop();

    const auto result = ash::run_agent(provider, tools, "do work", {}, source.get_token()).sync_wait();

    CHECK(result.stop_reason == "cancelled");
    CHECK(provider.requests.empty());
    // The prompt is still recorded, so a trace shows what was asked for.
    CHECK(result.transcript.size() == 2);
}

TEST_CASE("the loop hands the model's output to the sink as it arrives") {
    StreamingScriptedProvider provider{{reply(assistant_text("hello"))}};
    const ash::ToolRegistry tools;
    CollectingSink sink;

    const auto result =
        ash::run_agent(provider, tools, "greet", {}, std::stop_token{}, &sink).sync_wait();

    CHECK(result.stop_reason == "completed");
    // Five characters, five events: the loop is not collecting the answer and
    // replaying it at the end.
    CHECK(sink.texts.size() == 5);
    CHECK(sink.joined() == "hello");
    CHECK(sink.saw_done);
    CHECK(sink.finish_reason == "stop");
}

TEST_CASE("every step of a run streams through the same sink") {
    StreamingScriptedProvider provider{
        {reply(assistant_tool_call("c1", "echo", {{"text", "ping"}})), reply(assistant_text("ok"))}};
    ash::ToolRegistry tools;
    tools.add(echo_tool());
    CollectingSink sink;

    const auto result =
        ash::run_agent(provider, tools, "echo ping", {}, std::stop_token{}, &sink).sync_wait();

    CHECK(result.stop_reason == "completed");
    CHECK(result.steps.size() == 1);

    // Two model calls, so two endings. The first step asked for a tool and said
    // nothing, so its only content is the call fragment; the second said "ok".
    std::size_t endings = 0;
    std::size_t call_fragments = 0;
    for (const ash::StreamEvent& event : sink.events) {
        endings += std::holds_alternative<ash::StreamDone>(event) ? 1 : 0;
        call_fragments += std::holds_alternative<ash::ToolCallDelta>(event) ? 1 : 0;
    }

    CHECK(sink.joined() == "ok");
    CHECK(sink.texts.size() == 2);
    CHECK(call_fragments == 1);
    CHECK(endings == 2);
    CHECK(sink.events.size() == 5);
}

TEST_CASE("a run with no sink still completes") {
    // The sink is optional, and passing none must be the same run, not a
    // different code path. A null sink is not a mode switch.
    StreamingScriptedProvider provider{{reply(assistant_text("quiet"))}};
    const ash::ToolRegistry tools;

    const auto result = ash::run_agent(provider, tools, "greet").sync_wait();

    CHECK(result.stop_reason == "completed");
    REQUIRE(!result.transcript.empty());
    CHECK(result.transcript.back().content == "quiet");
}

TEST_CASE("a model call cut short ends the run as cancelled, keeping what arrived") {
    InterruptedProvider provider;
    const ash::ToolRegistry tools;
    CollectingSink sink;

    const auto result =
        ash::run_agent(provider, tools, "do work", {}, std::stop_token{}, &sink).sync_wait();

    CHECK(result.stop_reason == "cancelled");
    // The partial answer is not thrown away and not passed off as the model's
    // final word: it reached the sink, and the transcript holds only the prompt.
    CHECK(sink.joined() == "partial");
    CHECK_FALSE(sink.saw_done);
    REQUIRE(result.transcript.size() == 2);
    CHECK(result.transcript[1].content == "do work");
}

TEST_CASE("a cancelled streaming call is not reported as a failure") {
    // The distinction the Cancelled type exists for: a caller that only saw
    // std::runtime_error could not tell a stopped run from a crashed one.
    InterruptedProvider provider;
    const ash::ToolRegistry tools;

    ash::AgentOptions options;
    options.max_steps = 3;

    CHECK_NOTHROW(ash::run_agent(provider, tools, "do work", options).sync_wait());
}
