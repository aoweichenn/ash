#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

#include "ash/model/provider.hpp"
#include "ash/runtime.hpp"
#include "ash/tool/builtin.hpp"
#include "ash/tool/tool.hpp"
#include "logic.hpp"
#include "test_support.hpp"

using namespace ash::test;
using ash::cli::PermissionMode;

namespace {

ash::Message user_text(std::string content) {
    ash::Message message;
    message.role = ash::Role::kUser;
    message.content = std::move(content);
    return message;
}

ash::Message tool_result(std::string id, std::string content) {
    ash::Message message;
    message.role = ash::Role::kTool;
    message.content = std::move(content);
    message.tool_call_id = std::move(id);
    return message;
}

const ash::Message kSystem = [] {
    ash::Message message;
    message.role = ash::Role::kSystem;
    message.content = "you are ash";
    return message;
}();

}  // namespace

TEST_CASE("a mode name is read the way it is written, and junk is refused") {
    CHECK(ash::cli::parse_mode("ask") == PermissionMode::kAsk);
    CHECK(ash::cli::parse_mode("edits") == PermissionMode::kEdits);
    CHECK(ash::cli::parse_mode("yolo") == PermissionMode::kYolo);
    // "auto" is the word people reach for; it means yolo.
    CHECK(ash::cli::parse_mode("auto") == PermissionMode::kYolo);

    CHECK_FALSE(ash::cli::parse_mode("").has_value());
    CHECK_FALSE(ash::cli::parse_mode("Ask").has_value());
    CHECK_FALSE(ash::cli::parse_mode("yolo ").has_value());
    CHECK_FALSE(ash::cli::parse_mode("yes").has_value());
}

TEST_CASE("every mode name parses back to the mode it came from") {
    for (const PermissionMode mode :
         {PermissionMode::kAsk, PermissionMode::kEdits, PermissionMode::kYolo}) {
        const auto parsed = ash::cli::parse_mode(ash::cli::to_string(mode));
        REQUIRE(parsed.has_value());
        CHECK(*parsed == mode);
        // /mode with no argument has to be able to reach every mode and come
        // back, or one of them is unreachable by typing.
        CHECK(ash::cli::mode_description(mode) != std::string_view{});
    }
}

TEST_CASE("cycling walks all three modes and returns to the first") {
    CHECK(ash::cli::cycle_mode(PermissionMode::kAsk) == PermissionMode::kEdits);
    CHECK(ash::cli::cycle_mode(PermissionMode::kEdits) == PermissionMode::kYolo);
    CHECK(ash::cli::cycle_mode(PermissionMode::kYolo) == PermissionMode::kAsk);

    PermissionMode mode = PermissionMode::kAsk;
    for (int i = 0; i < 3; ++i) {
        mode = ash::cli::cycle_mode(mode);
    }
    CHECK(mode == PermissionMode::kAsk);
}

TEST_CASE("the approval matrix is the one the modes promise") {
    using ash::cli::should_require_approval;

    // Reads are never worth a prompt in any mode.
    for (const PermissionMode mode :
         {PermissionMode::kAsk, PermissionMode::kEdits, PermissionMode::kYolo}) {
        CHECK_FALSE(should_require_approval(mode, "read_file"));
        CHECK_FALSE(should_require_approval(mode, "list_dir"));
    }

    // ask confirms both, edits lets the write through and still confirms the
    // command, yolo confirms nothing.
    CHECK(should_require_approval(PermissionMode::kAsk, "write_file"));
    CHECK(should_require_approval(PermissionMode::kAsk, "run_shell"));
    CHECK_FALSE(should_require_approval(PermissionMode::kEdits, "write_file"));
    CHECK(should_require_approval(PermissionMode::kEdits, "run_shell"));
    CHECK_FALSE(should_require_approval(PermissionMode::kYolo, "write_file"));
    CHECK_FALSE(should_require_approval(PermissionMode::kYolo, "run_shell"));
}

TEST_CASE("a tool nobody has heard of is confirmed rather than run") {
    // The direction that fails safely. A tool added later must not inherit
    // permission because someone forgot to add it to a list here.
    CHECK(ash::cli::should_require_approval(PermissionMode::kAsk, "some_new_tool"));
    CHECK(ash::cli::should_require_approval(PermissionMode::kEdits, "some_new_tool"));
    CHECK(ash::cli::should_require_approval(PermissionMode::kAsk, ""));
    // yolo is the one mode that means "do not ask me".
    CHECK_FALSE(ash::cli::should_require_approval(PermissionMode::kYolo, "some_new_tool"));
}

TEST_CASE("the history drops the system message and keeps the rest in order") {
    ash::AgentResult result;
    result.transcript = {kSystem, user_text("hi"), assistant_text("hello"), user_text("more"),
                         assistant_text("more back")};

    const std::vector<ash::Message> body = ash::cli::carry_forward_history(result);

    REQUIRE(body.size() == 4);
    CHECK(body[0] == user_text("hi"));
    CHECK(body[1] == assistant_text("hello"));
    CHECK(body[2] == user_text("more"));
    CHECK(body[3] == assistant_text("more back"));
}

TEST_CASE("the history keeps tool calls and tool results") {
    ash::AgentResult result;
    result.transcript = {kSystem, user_text("list it"), assistant_tool_call("c1", "list_dir", {{"path", "."}}),
                         tool_result("c1", "README.md"), assistant_text("just a README")};

    const std::vector<ash::Message> body = ash::cli::carry_forward_history(result);

    // Index 0 is the user turn, because the system message is already gone.
    REQUIRE(body.size() == 4);
    CHECK(body[1].tool_calls.size() == 1);
    CHECK(body[1].tool_calls[0].name == "list_dir");
    // The tool message has no content of its own to lose, but dropping it would
    // leave the assistant's tool_calls with nothing answering them, which the
    // API rejects.
    CHECK(body[2].role == ash::Role::kTool);
    CHECK(body[2].tool_call_id == "c1");
}

TEST_CASE("a trailing assistant message with nothing in it is dropped") {
    // This is what a step that ends without saying anything leaves behind, and
    // the Anthropic encoder turns it into {"role":"assistant","content":[]},
    // which the API refuses. One of these would poison every later turn.
    ash::AgentResult result;
    result.transcript = {kSystem, user_text("hi"), assistant_text("an answer"), assistant_text(""),
                         assistant_text("")};

    const std::vector<ash::Message> body = ash::cli::carry_forward_history(result);

    REQUIRE(body.size() == 2);
    CHECK(body[0] == user_text("hi"));
    CHECK(body[1] == assistant_text("an answer"));
}

TEST_CASE("a user message the model never answered is dropped") {
    // A turn interrupted before the model replied leaves the question last. Kept,
    // it would open the next turn as two user messages in a row, which the API
    // rejects -- and it would reject every turn after that too, because the pair
    // stays in the history. Dropping it is the same choice the reader already
    // makes for a half-typed line: the interruption means forget it.
    ash::AgentResult result;
    result.transcript = {kSystem, user_text("hi")};

    CHECK(ash::cli::carry_forward_history(result).empty());
}

TEST_CASE("an interrupted turn keeps the work it finished") {
    // Only the unanswered tail goes. A turn cut short after a tool ran leaves
    // results the next request can carry, and dropping them would make the
    // session ask for the same work a second time.
    ash::AgentResult result;
    result.transcript = {kSystem, user_text("list it"),
                         assistant_tool_call("c1", "list_dir", {{"path", "."}}),
                         tool_result("c1", "README.md")};

    const std::vector<ash::Message> body = ash::cli::carry_forward_history(result);

    REQUIRE(body.size() == 3);
    CHECK(body.back().role == ash::Role::kTool);
}

TEST_CASE("a reply with nothing in it leaves nothing to continue from") {
    // The model answered with nothing at all. Once the empty message is dropped
    // the question is last, and it goes with it -- keeping it would make the next
    // request unbuildable, and the model has already declined to answer it.
    ash::AgentResult result;
    result.transcript = {kSystem, user_text("hi"), assistant_text("")};

    CHECK(ash::cli::carry_forward_history(result).empty());
}

TEST_CASE("an assistant message with tool calls is not empty, even with no text") {
    ash::AgentResult result;
    result.transcript = {kSystem, user_text("list it"), assistant_tool_call("c1", "list_dir", {{"path", "."}})};

    const std::vector<ash::Message> body = ash::cli::carry_forward_history(result);

    REQUIRE(body.size() == 2);
    CHECK(body.back().tool_calls.size() == 1);
}

TEST_CASE("an empty assistant message in the middle is left alone") {
    // Only the tail is trimmed. A blank turn in the middle is part of what was
    // said, and rewriting the middle of a conversation is not this function's
    // business. The fixture ends on an answer for the same reason: a transcript
    // that stops at a user message is the one shape the trim is for, so leaving
    // it there would test the trim instead of what this is about.
    ash::AgentResult result;
    result.transcript = {kSystem, user_text("hi"), assistant_text(""), user_text("still there"),
                         assistant_text("and answered")};

    const std::vector<ash::Message> body = ash::cli::carry_forward_history(result);

    REQUIRE(body.size() == 4);
    CHECK(body[1] == assistant_text(""));
    CHECK(body[2] == user_text("still there"));
    CHECK(body[3] == assistant_text("and answered"));
}

TEST_CASE("a transcript that does not open with a system message loses nothing") {
    // The system message is dropped so the caller can change the prompt between
    // turns. If the first message is not one, dropping it would throw away
    // something the user said, so nothing is dropped.
    ash::AgentResult result;
    result.transcript = {user_text("first thing said"), assistant_text("ok")};

    const std::vector<ash::Message> body = ash::cli::carry_forward_history(result);

    REQUIRE(body.size() == 2);
    CHECK(body[0] == user_text("first thing said"));
}

TEST_CASE("an empty transcript gives an empty history") {
    CHECK(ash::cli::carry_forward_history(ash::AgentResult{}).empty());
}

TEST_CASE("the system prompt names the directory and the tools that are installed") {
    const ash::ToolRegistry tools = ash::make_builtin_tools();
    const std::string prompt = ash::cli::build_system_prompt("/home/someone/project", tools);

    CHECK(prompt.find("/home/someone/project") != std::string::npos);
    for (const auto& tool : tools.tools()) {
        CHECK(prompt.find(tool->name()) != std::string::npos);
    }
}

TEST_CASE("the system prompt is built from the registry, not from a list of its own") {
    // A prompt that named tools it does not have would have the model calling
    // things that are not there. Reading the names off the registry is what
    // makes that impossible rather than unlikely.
    ash::ToolRegistry tools;
    tools.add(echo_tool());

    const std::string prompt = ash::cli::build_system_prompt("/tmp", tools);

    CHECK(prompt.find("echo") != std::string::npos);
    CHECK(prompt.find("read_file") == std::string::npos);
}
