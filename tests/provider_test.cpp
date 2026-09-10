#include <catch2/catch_test_macros.hpp>

#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "ash/model/provider.hpp"
#include "test_support.hpp"

namespace {

const std::vector<ash::Role> kAllRoles{
    ash::Role::kSystem, ash::Role::kUser, ash::Role::kAssistant, ash::Role::kTool};

ash::Message text_message(ash::Role role, std::string content) {
    ash::Message message;
    message.role = role;
    message.content = std::move(content);
    return message;
}

}  // namespace

TEST_CASE("role names round-trip through their wire spelling", "[provider]") {
    for (const auto role : kAllRoles) {
        REQUIRE(ash::role_from_string(ash::to_string(role)) == role);
    }
}

TEST_CASE("an unknown role name falls back to user", "[provider]") {
    REQUIRE(ash::role_from_string("robot") == ash::Role::kUser);
}

TEST_CASE("an assistant message with tool calls survives a round-trip", "[provider]") {
    ash::Message original;
    original.role = ash::Role::kAssistant;
    original.content = "let me look that up";
    original.tool_calls.push_back(
        ash::ToolCall{.id = "call_1", .name = "read_file", .arguments = {{"path", "a.txt"}}});
    original.tool_calls.push_back(
        ash::ToolCall{.id = "call_2", .name = "list_dir", .arguments = nlohmann::json::object()});

    const nlohmann::json encoded = original;
    REQUIRE(encoded.get<ash::Message>() == original);
}

TEST_CASE("a tool result message survives a round-trip", "[provider]") {
    ash::Message original;
    original.role = ash::Role::kTool;
    original.content = "file contents";
    original.tool_call_id = "call_1";

    const nlohmann::json encoded = original;
    const auto decoded = encoded.get<ash::Message>();
    REQUIRE(decoded == original);
    REQUIRE(decoded.role == ash::Role::kTool);
    REQUIRE(decoded.tool_call_id == "call_1");
}

TEST_CASE("a chat request survives a round-trip", "[provider]") {
    ash::ChatRequest original;
    original.model = "deepseek-chat";
    original.temperature = 0.25;
    original.max_tokens = 1024;
    original.messages.push_back(text_message(ash::Role::kSystem, "be terse"));
    original.messages.push_back(text_message(ash::Role::kUser, "hi"));
    original.tools.push_back(ash::ToolSpec{.name = "read_file",
                                           .description = "read a file",
                                           .input_schema = {{"type", "object"}}});

    const nlohmann::json encoded = original;
    const auto decoded = encoded.get<ash::ChatRequest>();

    REQUIRE(decoded == original);
    REQUIRE(decoded.max_tokens.has_value());
    REQUIRE(*decoded.max_tokens == 1024);
}

TEST_CASE("an omitted max_tokens stays empty through a round-trip", "[provider]") {
    ash::ChatRequest original;
    original.model = "deepseek-chat";

    const nlohmann::json encoded = original;
    const auto decoded = encoded.get<ash::ChatRequest>();

    REQUIRE_FALSE(decoded.max_tokens.has_value());
    REQUIRE(decoded == original);
}

TEST_CASE("a chat response survives a round-trip", "[provider]") {
    ash::ChatResponse original;
    original.message.role = ash::Role::kAssistant;
    original.message.content = "done";
    original.usage.prompt_tokens = 11;
    original.usage.completion_tokens = 7;
    original.finish_reason = "stop";
    original.model = "deepseek-chat";

    const nlohmann::json encoded = original;
    REQUIRE(encoded.get<ash::ChatResponse>() == original);
    REQUIRE(original.usage.total_tokens() == 18);
}

TEST_CASE("a provider that cannot stream still answers a streaming call", "[provider]") {
    // The base class folds: it makes the ordinary call and then reports the
    // whole answer through the sink. That is what lets every existing provider
    // -- and every test double -- serve a streaming caller without knowing
    // streaming exists, and what makes "no streaming support" a degradation
    // rather than a missing feature.
    ash::test::ScriptedProvider provider{{ash::test::reply(ash::test::assistant_text("all at once"))}};
    ash::test::CollectingSink sink;

    const ash::ChatResponse response = provider.chat_stream(ash::ChatRequest{}, sink).sync_wait();

    CHECK(response.message.content == "all at once");
    CHECK(provider.requests.size() == 1);

    // One event carrying the whole answer, not one per character: the fold is
    // honest about being a fold, and a sink that wanted to render text as it
    // arrived gets it all at the end rather than in fabricated pieces.
    REQUIRE(sink.texts.size() == 1);
    CHECK(sink.texts[0] == "all at once");
    CHECK(sink.saw_done);
    CHECK(sink.finish_reason == "stop");
}

TEST_CASE("the folding default reports an empty answer without inventing text", "[provider]") {
    ash::ChatResponse tool_call;
    tool_call.message.role = ash::Role::kAssistant;
    tool_call.message.tool_calls.push_back(
        ash::ToolCall{.id = "c1", .name = "echo", .arguments = {{"text", "x"}}});
    tool_call.finish_reason = "tool_calls";
    tool_call.model = "scripted-model";

    ash::test::ScriptedProvider provider{{tool_call}};
    ash::test::CollectingSink sink;

    const ash::ChatResponse response = provider.chat_stream(ash::ChatRequest{}, sink).sync_wait();

    // A response that is all tool call has no text, and the sink must not be
    // handed an empty text event for it -- a renderer would print nothing and a
    // test counting events would see one where there was no content.
    CHECK(sink.texts.empty());
    CHECK(sink.saw_done);
    REQUIRE(response.message.tool_calls.size() == 1);
}
