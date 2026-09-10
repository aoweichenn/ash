#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "ash/tool/builtin.hpp"
#include "ash/tool/tool.hpp"
#include "test_support.hpp"

using namespace ash::test;

TEST_CASE("registry rejects null and duplicate tools") {
    ash::ToolRegistry registry;
    CHECK(registry.empty());

    CHECK_THROWS_AS(registry.add(nullptr), std::invalid_argument);

    registry.add(echo_tool());
    CHECK(registry.size() == 1);

    CHECK_THROWS_AS(registry.add(echo_tool()), std::invalid_argument);
    CHECK(registry.size() == 1);
}

TEST_CASE("registry finds tools by name and exposes their specs") {
    ash::ToolRegistry registry;
    registry.add(echo_tool());
    registry.add(ash::make_read_file_tool());

    REQUIRE(registry.find("echo") != nullptr);
    CHECK(registry.find("echo")->description() == "Echo the given text back.");
    CHECK(registry.find("read_file") != nullptr);
    CHECK(registry.find("missing") == nullptr);

    const auto specs = registry.specs();
    REQUIRE(specs.size() == 2);
    CHECK(specs[0].name == "echo");
    CHECK(specs[1].name == "read_file");
    CHECK(specs[1].input_schema["properties"].contains("path"));
}

TEST_CASE("read_file returns file contents") {
    const TempDir dir;
    const auto file = dir / "hello.txt";
    write_text_file(file, "hello ash\n");

    const auto tool = ash::make_read_file_tool();
    const auto result = call_tool(*tool, {{"path", file.string()}});

    CHECK_FALSE(result.is_error);
    CHECK(result.content == "hello ash\n");
}

TEST_CASE("read_file refuses files over the size limit") {
    const TempDir dir;
    const auto file = dir / "big.txt";
    write_text_file(file, "0123456789");

    const auto tool = ash::make_read_file_tool(4);
    const auto result = call_tool(*tool, {{"path", file.string()}});

    CHECK(result.is_error);
    CHECK(result.content.find("over the 4 byte limit") != std::string::npos);
}

TEST_CASE("read_file reports a missing path rather than throwing") {
    const TempDir dir;

    const auto tool = ash::make_read_file_tool();
    const auto result = call_tool(*tool, {{"path", (dir / "nope.txt").string()}});

    CHECK(result.is_error);
    CHECK(result.content.find("cannot stat") != std::string::npos);
}

TEST_CASE("read_file requires a path") {
    const auto tool = ash::make_read_file_tool();
    CHECK(call_tool(*tool, nlohmann::json::object()).content == "read_file: 'path' is required");
}

TEST_CASE("write_file creates parent directories") {
    const TempDir dir;
    const auto nested = dir / "a" / "b" / "c.txt";

    const auto tool = ash::make_write_file_tool();
    const auto result = call_tool(*tool, {{"path", nested.string()}, {"content", "written"}});

    REQUIRE_FALSE(result.is_error);
    CHECK(std::filesystem::exists(nested));
    CHECK(result.content.find("7 bytes") != std::string::npos);
}

TEST_CASE("write then read round-trips through the filesystem") {
    const TempDir dir;
    const auto file = dir / "round.txt";
    const std::string payload = "line one\nline two\n";

    const auto writer = ash::make_write_file_tool();
    REQUIRE_FALSE(call_tool(*writer, {{"path", file.string()}, {"content", payload}}).is_error);

    const auto reader = ash::make_read_file_tool();
    CHECK(call_tool(*reader, {{"path", file.string()}}).content == payload);
}

TEST_CASE("list_dir sorts entries and marks directories") {
    const TempDir dir;
    write_text_file(dir / "b.txt", "b");
    write_text_file(dir / "a.txt", "a");
    std::filesystem::create_directories(dir / "sub");

    const auto tool = ash::make_list_dir_tool();
    const auto result = call_tool(*tool, {{"path", dir.str()}});

    CHECK_FALSE(result.is_error);
    CHECK(result.content == "a.txt\nb.txt\nsub/\n");
}

TEST_CASE("list_dir reports a missing directory") {
    const TempDir dir;

    const auto tool = ash::make_list_dir_tool();
    const auto result = call_tool(*tool, {{"path", (dir / "nowhere").string()}});

    CHECK(result.is_error);
    CHECK(result.content.find("does not exist") != std::string::npos);
}
