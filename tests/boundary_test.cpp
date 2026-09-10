#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>

namespace {

namespace fs = std::filesystem;

// Things that belong to a consumer of the runtime, never to the runtime itself.
// The Python bindings and the CLI are downstream of include/ash; if either
// starts leaking in, the core has stopped being embeddable.
struct Banned {
    std::string_view pattern;
    std::string_view why;
};

constexpr Banned kBanned[] = {
    {"pybind11", "the Python bindings must consume the core, not live inside it"},
    {"py::", "the Python bindings must consume the core, not live inside it"},
    {"<iostream>", "the core returns data; deciding what to print belongs to the CLI"},
    {"std::cout", "the core returns data; deciding what to print belongs to the CLI"},
    {"std::cerr", "the core returns data; deciding what to print belongs to the CLI"},
    {"int main(", "a library header must not define an entry point"},
};

}  // namespace

TEST_CASE("the public headers stay free of consumer-layer dependencies") {
    const fs::path include_root = fs::path{ASH_SOURCE_DIR} / "include" / "ash";
    REQUIRE(fs::exists(include_root));

    std::size_t scanned = 0;
    for (const auto& entry : fs::recursive_directory_iterator{include_root}) {
        if (!entry.is_regular_file() || entry.path().extension() != ".hpp") {
            continue;
        }
        ++scanned;

        std::ifstream input{entry.path()};
        std::ostringstream buffer;
        buffer << input.rdbuf();
        const std::string text = buffer.str();

        for (const Banned& banned : kBanned) {
            INFO("header: " << entry.path().string());
            INFO("reason: " << banned.why);
            CHECK(text.find(banned.pattern) == std::string::npos);
        }
    }

    // A boundary check that silently scanned nothing would always pass, so the
    // scan itself is asserted.
    CHECK(scanned >= 10);
}
