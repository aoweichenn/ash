#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <stop_token>
#include <string>
#include <thread>

#include <nlohmann/json.hpp>

#include "ash/tool/shell.hpp"
#include "ash/tool/tool.hpp"
#include "test_support.hpp"

using namespace ash::test;

namespace {

using Clock = std::chrono::steady_clock;

std::chrono::milliseconds since(Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start);
}

// Every command here that could hang would otherwise hang until the tool's own
// two-minute timeout, so the tests that care also bound the wall clock. The
// bound is generous on purpose: it is there to tell "wrong" apart from "slow",
// not to measure anything.
constexpr auto kPromptly = std::chrono::seconds{10};

}  // namespace

TEST_CASE("run_shell returns the command's output") {
    const auto tool = ash::make_shell_tool();
    const auto result = call_tool(*tool, {{"command", "echo hello"}});

    CHECK_FALSE(result.is_error);
    CHECK(result.content == "hello\n");
}

TEST_CASE("run_shell captures stderr alongside stdout") {
    const auto tool = ash::make_shell_tool();
    const auto result = call_tool(*tool, {{"command", "echo to-stdout; echo to-stderr 1>&2"}});

    CHECK_FALSE(result.is_error);
    CHECK(result.content.find("to-stdout") != std::string::npos);
    CHECK(result.content.find("to-stderr") != std::string::npos);
}

TEST_CASE("a non-zero exit is an error result rather than an exception") {
    const auto tool = ash::make_shell_tool();

    ash::ToolResult result;
    REQUIRE_NOTHROW(result = call_tool(*tool, {{"command", "echo failing; exit 3"}}));

    CHECK(result.is_error);
    // The output the command managed to produce is kept: it is usually the
    // explanation for the status.
    CHECK(result.content.find("failing") != std::string::npos);
    CHECK(result.content.find("[exit status 3]") != std::string::npos);
}

TEST_CASE("output past the limit is dropped, and the command still finishes") {
    ash::ShellOptions options;
    options.max_output_bytes = 64;
    const auto tool = ash::make_shell_tool(options);

    // A megabyte of output into a pipe nobody is draining would block the
    // writer forever, so this test is the drain check as much as the truncation
    // one: an implementation that stops reading at the limit hangs here.
    const auto start = Clock::now();
    const auto result = call_tool(*tool, {{"command", "head -c 1000000 /dev/zero | tr '\\0' a"}});
    const auto elapsed = since(start);

    CHECK_FALSE(result.is_error);
    CHECK(elapsed < kPromptly);
    CHECK(result.content.find("[output truncated at 64 bytes; 1000000 bytes total]") !=
          std::string::npos);
    // What was kept is the limit plus the note, not the megabyte.
    CHECK(result.content.size() < 200);
}

TEST_CASE("a command that outlives its timeout is killed and reported") {
    ash::ShellOptions options;
    options.timeout = std::chrono::milliseconds{300};
    const auto tool = ash::make_shell_tool(options);

    const auto start = Clock::now();
    const auto result = call_tool(*tool, {{"command", "echo reached; sleep 30; echo unreached"}});
    const auto elapsed = since(start);

    CHECK(result.is_error);
    CHECK(result.content.find("reached") != std::string::npos);
    // The words have to be ones the marker itself does not use: the note says
    // "timed out after N ms", so asserting on "after" would only be testing the
    // note against itself.
    CHECK(result.content.find("unreached") == std::string::npos);
    CHECK(result.content.find("[command timed out after 300 ms and was killed]") !=
          std::string::npos);
    CHECK(elapsed < kPromptly);
}

TEST_CASE("a stop request ends a command that is still running") {
    const auto tool = ash::make_shell_tool();

    std::stop_source source;
    std::jthread stopper{[&source] {
        std::this_thread::sleep_for(std::chrono::milliseconds{300});
        source.request_stop();
    }};

    const auto start = Clock::now();
    const auto result = tool->invoke({{"command", "sleep 30"}}, source.get_token()).sync_wait();
    const auto elapsed = since(start);

    CHECK(result.is_error);
    CHECK(result.content.find("[command cancelled]") != std::string::npos);
    CHECK(elapsed < kPromptly);
}

TEST_CASE("a cancelled run never starts the command") {
    const auto tool = ash::make_shell_tool();

    std::stop_source source;
    source.request_stop();

    const auto result = tool->invoke({{"command", "echo hi"}}, source.get_token()).sync_wait();

    CHECK(result.is_error);
    CHECK(result.content.find("cancelled before the command started") != std::string::npos);
}

TEST_CASE("the command's stdin is empty, so a command that reads still finishes") {
    const auto tool = ash::make_shell_tool();

    // Without an empty stdin this reads whatever the caller has, which in a
    // session is the user's next line -- or, when there is nothing, waits
    // forever.
    const auto start = Clock::now();
    const auto result = call_tool(*tool, {{"command", "read line; echo got-$line"}});
    const auto elapsed = since(start);

    CHECK(elapsed < kPromptly);
    CHECK_FALSE(result.is_error);
    CHECK(result.content.find("got-") != std::string::npos);
}

TEST_CASE("a child the command leaves behind does not hold the tool open") {
    const auto tool = ash::make_shell_tool();

    // The shell exits at once, but the sleep it started inherited the pipe, so
    // the read end never sees EOF. Waiting for EOF before returning would block
    // here for the full two-minute timeout.
    const auto start = Clock::now();
    const auto result = call_tool(*tool, {{"command", "sleep 30 & echo started"}});
    const auto elapsed = since(start);

    CHECK(elapsed < kPromptly);
    CHECK_FALSE(result.is_error);
    CHECK(result.content.find("started") != std::string::npos);
}

TEST_CASE("bad arguments are reported rather than thrown") {
    const auto tool = ash::make_shell_tool();

    ash::ToolResult missing;
    ash::ToolResult empty;
    ash::ToolResult wrong_type;
    ash::ToolResult not_an_object;
    REQUIRE_NOTHROW(missing = call_tool(*tool, nlohmann::json::object()));
    REQUIRE_NOTHROW(empty = call_tool(*tool, {{"command", ""}}));
    REQUIRE_NOTHROW(wrong_type = call_tool(*tool, {{"command", 7}}));
    REQUIRE_NOTHROW(not_an_object = call_tool(*tool, nlohmann::json::array()));

    CHECK(missing.is_error);
    CHECK(missing.content == "run_shell: 'command' is required and must be a string");
    CHECK(empty.is_error);
    CHECK(wrong_type.is_error);
    CHECK(not_an_object.is_error);
}

TEST_CASE("a shell that cannot be run is an error result") {
    ash::ShellOptions options;
    options.shell = "/no/such/shell";
    const auto tool = ash::make_shell_tool(options);

    ash::ToolResult result;
    REQUIRE_NOTHROW(result = call_tool(*tool, {{"command", "echo hi"}}));

    CHECK(result.is_error);
    CHECK(result.content.find("cannot run '/no/such/shell'") != std::string::npos);
}

TEST_CASE("run_shell advertises the one argument it takes") {
    const auto tool = ash::make_shell_tool();

    CHECK(tool->name() == "run_shell");
    CHECK(tool->input_schema()["required"] == nlohmann::json::array({"command"}));
    CHECK(tool->input_schema()["additionalProperties"] == false);
    // No per-call timeout: a model that can choose how long to hang the session
    // is a model that eventually will.
    CHECK_FALSE(tool->input_schema()["properties"].contains("timeout"));
}
