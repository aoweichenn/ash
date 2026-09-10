#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "ash/tool/permission.hpp"
#include "ash/tool/tool.hpp"
#include "test_support.hpp"

using namespace ash::test;

namespace {

// Records that it was called, so a test can tell "the call was refused" apart
// from "the call happened and happened to return an error".
class CountingTool final : public ash::Tool {
public:
    explicit CountingTool(std::string name) : name_(std::move(name)) {}

    [[nodiscard]] std::string_view name() const noexcept override { return name_; }
    [[nodiscard]] std::string_view description() const noexcept override { return "a tool that counts"; }
    [[nodiscard]] const nlohmann::json& input_schema() const noexcept override { return schema_; }

    ash::Task<ash::ToolResult> invoke(const nlohmann::json&, std::stop_token) const override {
        ++calls;
        co_return ash::ToolResult{"the real work happened", false};
    }

    mutable int calls = 0;

private:
    std::string name_;
    nlohmann::json schema_ = nlohmann::json{{"type", "object"}};
};

// What the gate asked, in order, so the question itself can be checked and not
// only its answer.
struct Asked {
    std::string tool;
    nlohmann::json arguments;
};

}  // namespace

TEST_CASE("an approved call reaches the tool") {
    auto inner = std::make_shared<CountingTool>("write_file");
    ash::ToolRegistry tools;
    tools.add(inner);

    std::vector<Asked> asked;
    const ash::ToolRegistry gated = ash::make_permission_registry(
        tools, [&asked](std::string_view tool, const nlohmann::json& arguments) {
            asked.push_back(Asked{std::string{tool}, arguments});
            return true;
        });

    const ash::ToolResult result = call_tool(*gated.find("write_file"), {{"path", "a.txt"}});

    CHECK_FALSE(result.is_error);
    CHECK(result.content == "the real work happened");
    CHECK(inner->calls == 1);

    REQUIRE(asked.size() == 1);
    // The question has to carry what is being asked about, or the answer is not
    // an answer to anything.
    CHECK(asked[0].tool == "write_file");
    CHECK(asked[0].arguments == nlohmann::json{{"path", "a.txt"}});
}

TEST_CASE("a refused call never reaches the tool") {
    auto inner = std::make_shared<CountingTool>("write_file");
    ash::ToolRegistry tools;
    tools.add(inner);

    const ash::ToolRegistry gated =
        ash::make_permission_registry(tools, [](std::string_view, const nlohmann::json&) { return false; });

    const ash::ToolResult result = call_tool(*gated.find("write_file"), {{"path", "a.txt"}});

    // The whole point: nothing happened.
    CHECK(inner->calls == 0);
    CHECK(result.is_error);
    CHECK(result.content == "the user denied this tool call");
}

TEST_CASE("a refusal is an error result rather than an exception") {
    auto inner = std::make_shared<CountingTool>("run_shell");
    ash::ToolRegistry tools;
    tools.add(inner);

    const ash::ToolRegistry gated =
        ash::make_permission_registry(tools, [](std::string_view, const nlohmann::json&) { return false; });

    // agent_loop.cpp awaits a tool outside the try that catches Cancelled, so a
    // tool that threw here would unwind straight out of run_agent and end the
    // session instead of ending the call.
    ash::ToolResult result;
    REQUIRE_NOTHROW(result = call_tool(*gated.find("run_shell"), {{"command", "rm -rf /"}}));
    CHECK(result.is_error);
}

TEST_CASE("the refusal message is the one the caller asked for") {
    auto inner = std::make_shared<CountingTool>("run_shell");
    ash::ToolRegistry tools;
    tools.add(inner);

    const ash::ToolRegistry gated = ash::make_permission_registry(
        tools, [](std::string_view, const nlohmann::json&) { return false; }, "not this time");

    CHECK(call_tool(*gated.find("run_shell"), {}).content == "not this time");
}

TEST_CASE("the model cannot tell a gated tool from the tool itself") {
    auto inner = std::make_shared<CountingTool>("write_file");
    ash::ToolRegistry tools;
    tools.add(inner);

    const ash::ToolRegistry gated =
        ash::make_permission_registry(tools, [](std::string_view, const nlohmann::json&) { return true; });

    const ash::Tool* wrapped = gated.find("write_file");
    REQUIRE(wrapped != nullptr);
    // Same name, same description, same schema: a gate that changed what the
    // model was told about a tool would change how the model uses it.
    CHECK(wrapped->name() == inner->name());
    CHECK(wrapped->description() == inner->description());
    CHECK(wrapped->input_schema() == inner->input_schema());
    CHECK(wrapped->spec() == inner->spec());
}

TEST_CASE("every tool is gated, and the tools that were not asked about are gone") {
    ash::ToolRegistry tools;
    tools.add(std::make_shared<CountingTool>("one"));
    tools.add(std::make_shared<CountingTool>("two"));
    tools.add(std::make_shared<CountingTool>("three"));

    int questions = 0;
    const ash::ToolRegistry gated = ash::make_permission_registry(
        tools, [&questions](std::string_view, const nlohmann::json&) {
            ++questions;
            return false;
        });

    CHECK(gated.size() == tools.size());
    for (const auto& tool : tools.tools()) {
        CHECK(gated.find(tool->name()) != nullptr);
    }

    for (const char* name : {"one", "two", "three"}) {
        call_tool(*gated.find(name), {});
    }
    // One question per tool, not one per registry: the gate wraps tools, it
    // does not replace them with a single switch.
    CHECK(questions == 3);
}

TEST_CASE("wrapping leaves the original registry alone") {
    auto inner = std::make_shared<CountingTool>("write_file");
    ash::ToolRegistry tools;
    tools.add(inner);

    const ash::ToolRegistry gated =
        ash::make_permission_registry(tools, [](std::string_view, const nlohmann::json&) { return false; });

    // `run` and `eval` still get the ungated set, so the recording path is not
    // quietly routed through a gate that answers no.
    CHECK(tools.find("write_file") == inner.get());
    CHECK(gated.find("write_file") != inner.get());
    CHECK_FALSE(call_tool(*tools.find("write_file"), {}).is_error);
    CHECK(inner->calls == 1);
}

TEST_CASE("an empty registry stays empty") {
    ash::ToolRegistry tools;

    const ash::ToolRegistry gated =
        ash::make_permission_registry(tools, [](std::string_view, const nlohmann::json&) { return true; });

    CHECK(gated.empty());
}

TEST_CASE("the question is asked again on every call") {
    // Nothing is remembered here on purpose. "Always allow this one" is state
    // that belongs to the session, and putting it in the tool would break the
    // property that lets one registry be shared across threads without locking.
    auto inner = std::make_shared<CountingTool>("write_file");
    ash::ToolRegistry tools;
    tools.add(inner);

    int questions = 0;
    bool allow = false;
    const ash::ToolRegistry gated = ash::make_permission_registry(
        tools, [&questions, &allow](std::string_view, const nlohmann::json&) {
            ++questions;
            return allow;
        });

    const ash::Tool* tool = gated.find("write_file");
    CHECK(call_tool(*tool, {}).is_error);
    allow = true;
    CHECK_FALSE(call_tool(*tool, {}).is_error);
    allow = false;
    CHECK(call_tool(*tool, {}).is_error);

    CHECK(questions == 3);
    CHECK(inner->calls == 1);
}
