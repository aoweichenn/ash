// The recordings under examples/journals/ were made against real endpoints.
// Replaying them here turns them into a regression test for request assembly:
// if the system prompt, the tool declarations, or the order of the messages
// ever changes, the computed request stops matching the recording and these
// fail with the two hashes that disagree. It costs nothing to run.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "ash/record/decorators.hpp"
#include "ash/record/journal.hpp"
#include "ash/record/replay.hpp"
#include "ash/runtime.hpp"

namespace {

std::vector<std::filesystem::path> fixture_journals() {
    const std::filesystem::path dir = std::filesystem::path{ASH_SOURCE_DIR} / "examples" / "journals";

    std::vector<std::filesystem::path> paths;
    for (const auto& entry : std::filesystem::directory_iterator{dir}) {
        if (entry.path().extension() == ".jsonl") {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

std::string read_text(const std::filesystem::path& path) {
    std::ifstream input{path, std::ios::binary};
    return std::string{std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
}

std::size_t count_model_calls(const ash::Journal& journal) {
    return static_cast<std::size_t>(
        std::count_if(journal.events().begin(), journal.events().end(), [](const ash::Event& event) {
            return std::holds_alternative<ash::ModelCallRecord>(event.payload);
        }));
}

}  // namespace

TEST_CASE("a committed recording still replays against the current code") {
    const std::vector<std::filesystem::path> journals = fixture_journals();

    // Without this the test would pass by iterating over nothing, which is the
    // one way a self-check like this can lie.
    REQUIRE(journals.size() >= 2);

    for (const auto& path : journals) {
        INFO("journal: " << path.string());

        const ash::Journal journal = ash::Journal::load(path);
        REQUIRE_FALSE(journal.events().empty());

        const std::string actor = journal.events().front().actor;
        const ash::JournalHeader& header = journal.header();
        CHECK_FALSE(header.provider.empty());
        CHECK_FALSE(header.model.empty());
        CHECK_FALSE(header.task.empty());

        ash::ReplayCursor cursor{journal, actor};
        ash::ReplayingProvider provider{cursor, header.provider, header.model};
        ash::ToolRegistry tools = ash::make_replaying_registry(journal, cursor, actor);

        ash::AgentOptions options;
        options.system_prompt = header.system_prompt;
        options.max_steps = header.max_steps > 0 ? header.max_steps : 16;

        const ash::AgentResult result =
            ash::run_agent(provider, tools, header.task, options).sync_wait();

        CHECK(result.stop_reason == "completed");
        // Consuming the whole journal means the run took exactly the recorded
        // path. A mismatch throws with the offending request hash.
        CHECK_NOTHROW(cursor.verify_consumed());

        // The replayed loop must arrive at the answer the model actually gave.
        REQUIRE_FALSE(result.transcript.empty());
        CHECK_FALSE(result.transcript.back().content.empty());
        CHECK(result.transcript.back().role == ash::Role::kAssistant);
        // Every step but the last one ended in a tool call, so the recorded run
        // made exactly one more model call than it has steps.
        CHECK(result.steps.size() + 1 == count_model_calls(journal));
    }
}

TEST_CASE("no committed recording contains credential-shaped text") {
    const std::vector<std::filesystem::path> journals = fixture_journals();
    REQUIRE(journals.size() >= 2);

    for (const auto& path : journals) {
        INFO("journal: " << path.string());
        const std::string text = read_text(path);

        // These recordings were made with a real key in the environment, so this
        // is the place a leak would actually show up rather than in theory.
        CHECK(text.find("sk-") == std::string::npos);
        CHECK(text.find("api_key") == std::string::npos);
        CHECK(text.find("Authorization") == std::string::npos);
        CHECK(text.find("Bearer ") == std::string::npos);
    }
}
