#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "ash/record/decorators.hpp"
#include "ash/record/journal.hpp"
#include "ash/record/replay.hpp"
#include "ash/runtime.hpp"
#include "ash/tool/tool.hpp"
#include "test_support.hpp"

using namespace ash::test;

namespace {

constexpr std::string_view kActor = "root";
constexpr std::string_view kTask = "echo the word ping";

std::vector<ash::ChatResponse> two_step_script() {
    return {reply(assistant_tool_call("call-1", "echo", {{"text", "ping"}})),
            reply(assistant_text("the tool said ping"))};
}

ash::AgentOptions test_options() {
    ash::AgentOptions options;
    options.max_steps = 8;
    return options;
}

// Records a two-step run -- one tool call, two model calls -- to `journal_path`
// and returns what the run produced. The journal is closed before this returns,
// so callers can load it.
ash::AgentResult record_two_step_run(const std::filesystem::path& journal_path) {
    const std::string actor{kActor};
    const ash::AgentOptions options = test_options();

    ash::ToolRegistry live_tools;
    live_tools.add(echo_tool());

    ash::Journal writer = ash::Journal::create(journal_path);

    ash::JournalHeader header;
    header.provider = "scripted";
    header.model = "scripted-model";
    header.task = std::string{kTask};
    header.system_prompt = options.system_prompt;
    header.max_steps = options.max_steps;
    writer.set_header(std::move(header));

    ash::RecordingProvider provider{std::make_unique<ScriptedProvider>(two_step_script()), writer, actor};
    ash::ToolRegistry recording_tools = ash::make_recording_registry(live_tools, writer, actor);

    const ash::AgentResult result =
        ash::run_agent(provider, recording_tools, std::string{kTask}, options).sync_wait();
    REQUIRE(result.stop_reason == "completed");
    return result;
}

// Replays a journal with nothing but the journal itself.
ash::ReplayCursor make_cursor(const ash::Journal& journal) {
    return ash::ReplayCursor{journal, std::string{kActor}};
}

void collect_keys(const nlohmann::json& value, std::vector<std::string>& out) {
    if (value.is_object()) {
        for (auto it = value.begin(); it != value.end(); ++it) {
            out.push_back(it.key());
            collect_keys(it.value(), out);
        }
    } else if (value.is_array()) {
        for (const auto& element : value) {
            collect_keys(element, out);
        }
    }
}

std::string lowercase(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return text;
}

}  // namespace

TEST_CASE("a recorded run replays to an identical transcript") {
    const TempDir dir;
    const auto journal_path = dir / "run.jsonl";
    const ash::AgentResult recorded = record_two_step_run(journal_path);
    REQUIRE(recorded.steps.size() == 1);

    const ash::Journal loaded = ash::Journal::load(journal_path);
    ash::ReplayCursor cursor = make_cursor(loaded);
    ash::ReplayingProvider provider{cursor, loaded.header().provider, loaded.header().model};
    const ash::ToolRegistry replaying_tools = ash::make_replaying_registry(loaded, cursor, kActor);

    const auto replayed =
        ash::run_agent(provider, replaying_tools, loaded.header().task, test_options()).sync_wait();

    // The loop itself re-ran; every side effect came from the journal.
    CHECK(replayed.transcript == recorded.transcript);
    CHECK(replayed.stop_reason == recorded.stop_reason);
    CHECK(replayed.usage == recorded.usage);
    CHECK(replayed.steps.size() == recorded.steps.size());

    CHECK_NOTHROW(cursor.verify_consumed());
    CHECK(cursor.consumed() == 3);  // two model calls plus one tool call
}

TEST_CASE("the journal keeps a run's configuration so a replay can rebuild it") {
    const TempDir dir;
    const auto journal_path = dir / "run.jsonl";
    record_two_step_run(journal_path);

    const ash::Journal loaded = ash::Journal::load(journal_path);
    CHECK(loaded.header().version == ash::kJournalVersion);
    CHECK(loaded.header().provider == "scripted");
    CHECK(loaded.header().model == "scripted-model");
    CHECK(loaded.header().task == kTask);
    CHECK(loaded.header().max_steps == 8);
    CHECK_FALSE(loaded.header().created_at.empty());
}

TEST_CASE("events load ordered by actor and sequence") {
    const TempDir dir;
    const auto path = dir / "order.jsonl";
    const ash::ChatRequest request;

    {
        ash::Journal writer = ash::Journal::create(path);
        writer.set_header(ash::JournalHeader{});
        writer.append("alpha", ash::ModelCallRecord{request, ash::ChatResponse{}});
        writer.append("beta", ash::ModelCallRecord{request, ash::ChatResponse{}});
        writer.append("alpha", ash::ModelCallRecord{request, ash::ChatResponse{}});
    }

    const ash::Journal loaded = ash::Journal::load(path);
    REQUIRE(loaded.events().size() == 3);
    // Grouped by actor, each sequence starting at zero, so the order in which
    // concurrent tasks happened to interleave is not part of the format.
    CHECK(loaded.events()[0].actor == "alpha");
    CHECK(loaded.events()[0].seq == 0);
    CHECK(loaded.events()[1].actor == "alpha");
    CHECK(loaded.events()[1].seq == 1);
    CHECK(loaded.events()[2].actor == "beta");
    CHECK(loaded.events()[2].seq == 0);
}

TEST_CASE("the tool declarations are recovered from the recording") {
    const TempDir dir;
    const auto journal_path = dir / "run.jsonl";
    record_two_step_run(journal_path);

    const ash::Journal loaded = ash::Journal::load(journal_path);
    const auto specs = ash::recorded_tool_specs(loaded, kActor);
    REQUIRE(specs.size() == 1);
    CHECK(specs[0].name == "echo");
    CHECK(specs[0].description == "Echo the given text back.");

    // This is what lets a replay rebuild the tool set from the journal alone,
    // instead of asking the caller to register the same tools again.
    ash::ReplayCursor cursor = make_cursor(loaded);
    const ash::ToolRegistry replaying = ash::make_replaying_registry(loaded, cursor, kActor);
    REQUIRE(replaying.size() == 1);
    CHECK(replaying.find("echo") != nullptr);
    CHECK(replaying.find("read_file") == nullptr);
}

TEST_CASE("replay reports a request that diverges from the recording") {
    const TempDir dir;
    const auto journal_path = dir / "run.jsonl";
    record_two_step_run(journal_path);

    const ash::Journal loaded = ash::Journal::load(journal_path);
    ash::ReplayCursor cursor = make_cursor(loaded);
    ash::ReplayingProvider provider{cursor, loaded.header().provider, loaded.header().model};
    const ash::ToolRegistry tools = ash::make_replaying_registry(loaded, cursor, kActor);

    CHECK_THROWS_AS(
        ash::run_agent(provider, tools, "a completely different task", test_options()).sync_wait(),
        ash::ReplayError);
}

TEST_CASE("replay reports a journal that runs out of events") {
    const TempDir dir;
    const auto journal_path = dir / "run.jsonl";
    const auto short_path = dir / "short.jsonl";
    record_two_step_run(journal_path);

    // Drop the final event, as if the recording had been cut short.
    {
        std::ifstream input{journal_path};
        std::ofstream output{short_path};
        std::vector<std::string> lines;
        std::string line;
        while (std::getline(input, line)) {
            if (!line.empty()) {
                lines.push_back(line);
            }
        }
        REQUIRE(lines.size() == 4);  // a header plus three events
        for (std::size_t i = 0; i + 1 < lines.size(); ++i) {
            output << lines[i] << "\n";
        }
    }

    const ash::Journal loaded = ash::Journal::load(short_path);
    ash::ReplayCursor cursor = make_cursor(loaded);
    ash::ReplayingProvider provider{cursor, loaded.header().provider, loaded.header().model};
    const ash::ToolRegistry tools = ash::make_replaying_registry(loaded, cursor, kActor);

    CHECK_THROWS_WITH(
        ash::run_agent(provider, tools, loaded.header().task, test_options()).sync_wait(),
        Catch::Matchers::ContainsSubstring("ran past the end"));
}

TEST_CASE("verify_consumed reports a journal that was not fully replayed") {
    const TempDir dir;
    const auto path = dir / "two.jsonl";
    const ash::ChatRequest request;

    {
        ash::Journal writer = ash::Journal::create(path);
        writer.set_header(ash::JournalHeader{});
        writer.append(kActor, ash::ModelCallRecord{request, ash::ChatResponse{}});
        writer.append(kActor, ash::ModelCallRecord{request, ash::ChatResponse{}});
    }

    const ash::Journal loaded = ash::Journal::load(path);
    ash::ReplayCursor cursor = make_cursor(loaded);
    REQUIRE(cursor.available() == 2);

    // Both recorded responses are the default-constructed ones.
    CHECK(cursor.next_model_call(request).message.content.empty());
    CHECK(cursor.consumed() == 1);
    CHECK_THROWS_AS(cursor.verify_consumed(), ash::ReplayError);

    CHECK(cursor.next_model_call(request).message.content.empty());
    CHECK_NOTHROW(cursor.verify_consumed());
}

TEST_CASE("no credential-shaped field can reach the journal") {
    const TempDir dir;
    const auto journal_path = dir / "run.jsonl";
    record_two_step_run(journal_path);

    // The API key lives in ProviderConfig, which is not part of any request, so
    // it has no path into the journal. This walks every key that was written to
    // keep it that way.
    static constexpr std::string_view kBanned[] = {"api_key", "apikey",  "api-key",
                                                  "authorization", "bearer", "secret"};

    std::ifstream input{journal_path};
    std::string line;
    std::size_t records = 0;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        ++records;

        const nlohmann::json record = nlohmann::json::parse(line);
        std::vector<std::string> keys;
        collect_keys(record, keys);
        REQUIRE_FALSE(keys.empty());

        for (const std::string& key : keys) {
            const std::string lowered = lowercase(key);
            for (const std::string_view banned : kBanned) {
                INFO("field: " << key);
                CHECK(lowered.find(banned) == std::string::npos);
            }
        }
    }
    CHECK(records == 4);
}

TEST_CASE("an unknown journal version is rejected") {
    const TempDir dir;
    const auto path = dir / "version.jsonl";
    {
        std::ofstream output{path};
        output << R"({"kind":"header","version":99,"provider":"p","model":"m"})" << "\n";
    }

    CHECK_THROWS_AS(ash::Journal::load(path), std::runtime_error);
}

TEST_CASE("a corrupt journal line is reported with its line number") {
    const TempDir dir;
    const auto path = dir / "corrupt.jsonl";
    {
        std::ofstream output{path};
        output << R"({"kind":"header","version":1})" << "\n";
        output << "this is not json\n";
    }

    CHECK_THROWS_WITH(ash::Journal::load(path), Catch::Matchers::ContainsSubstring("line 2"));
}
