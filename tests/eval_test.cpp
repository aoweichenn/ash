#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "ash/record/journal.hpp"
#include "ash_eval.hpp"

namespace {

namespace fs = std::filesystem;

fs::path core_suite() { return fs::path{ASH_SOURCE_DIR} / "examples" / "suites" / "core.json"; }

// A scratch directory that cleans up after itself, so the tests can write
// journals and suites without leaving anything behind on failure.
class TempDir {
public:
    explicit TempDir(std::string name) : path_{fs::temp_directory_path() / std::move(name)} {
        fs::remove_all(path_);
        fs::create_directories(path_);
    }
    ~TempDir() { fs::remove_all(path_); }

    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    [[nodiscard]] const fs::path& path() const noexcept { return path_; }

private:
    fs::path path_;
};

std::string read_text(const fs::path& path) {
    std::ifstream input{path, std::ios::binary};
    REQUIRE(input);
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

void write_text(const fs::path& path, const std::string& text) {
    std::ofstream output{path, std::ios::binary | std::ios::trunc};
    REQUIRE(output);
    output << text;
}

std::string join_failures(const std::vector<std::string>& failures) {
    std::string joined;
    for (const std::string& failure : failures) {
        joined += (joined.empty() ? "" : "; ") + failure;
    }
    return joined;
}

// Rewrites the journal header's model, and nothing else. The model belongs to
// the header rather than to the recorded requests: the loop composes a request
// without one and the provider adapter fills its own in, so a recorded request
// carries an empty model by design. Rewriting the requests too would make the
// replayed run diverge on the very first comparison.
void rewrite_header_model(const fs::path& source, const fs::path& destination, const std::string& model) {
    std::istringstream input{read_text(source)};
    std::ofstream output{destination, std::ios::binary | std::ios::trunc};
    REQUIRE(output);

    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        nlohmann::json parsed = nlohmann::json::parse(line);
        if (parsed.value("kind", "") == "header") {
            parsed["model"] = model;
        }
        output << parsed.dump() << '\n';
    }
}

// A step is a model turn that asked for tools. The turn that closes the loop
// asks for none and is not recorded as a step, so a completed run has exactly
// one more model call than it has steps.
int model_calls(const fs::path& journal_path) {
    const ash::Journal journal = ash::Journal::load(journal_path);
    int count = 0;
    for (const ash::Event& event : journal.events()) {
        if (std::holds_alternative<ash::ModelCallRecord>(event.payload)) {
            ++count;
        }
    }
    return count;
}

}  // namespace

TEST_CASE("the committed suite replays offline and passes every check") {
    const ash::eval::Suite suite = ash::eval::load_suite(core_suite());
    const ash::eval::SuiteReport report = ash::eval::run_suite(suite);

    REQUIRE(report.jobs.size() == 2);
    for (std::size_t i = 0; i < report.jobs.size(); ++i) {
        const ash::eval::JobResult& job = report.jobs[i];
        INFO("job " << job.id << ": " << join_failures(job.failures));
        CHECK(job.passed);
        CHECK(job.stop_reason == "completed");
        CHECK(job.steps + 1 == model_calls(suite.jobs[i].journal));
        CHECK(job.wall_us > 0);
    }

    CHECK(report.passed() == 2);
    CHECK(report.failed() == 0);
    CHECK(report.total_prompt_tokens() > 0);
    CHECK(report.total_completion_tokens() > 0);

    // Both fixtures use a model the price table knows, so the cost column is
    // populated rather than quietly zero.
    CHECK(report.every_cost_known());
    CHECK(report.total_cost_usd() > 0.0);
    CHECK(report.total_cost_usd() < 0.01);

    // Three recorded calls in the openai journal and four in the anthropic one.
    CHECK(report.model_call_latencies_us.size() == 7);
    CHECK(report.latency_percentile_us(50) > 0);
    CHECK(report.latency_percentile_us(50) <= report.latency_percentile_us(95));
    CHECK(report.latency_percentile_us(95) <= report.latency_percentile_us(100));

    for (const ash::eval::JobResult& job : report.jobs) {
        CHECK(job.model_latency_us > 0);
        CHECK(job.cost_usd.has_value());
    }
}

TEST_CASE("a check that does not hold fails the job and explains why") {
    ash::eval::Suite suite;
    suite.name = "tampered";

    ash::eval::Job job;
    job.id = "wrong-tools";
    job.journal = fs::path{ASH_SOURCE_DIR} / "examples" / "journals" / "openai.jsonl";
    job.checks.tools_called = {"list_dir", "write_file"};
    suite.jobs.push_back(std::move(job));

    const ash::eval::SuiteReport report = ash::eval::run_suite(suite);

    REQUIRE(report.jobs.size() == 1);
    const ash::eval::JobResult& result = report.jobs.front();
    CHECK_FALSE(result.passed);
    CHECK(report.failed() == 1);

    const std::string failures = join_failures(result.failures);
    INFO(failures);
    // The message has to name both sides, or it is not actionable.
    CHECK(failures.find("read_file") != std::string::npos);
    CHECK(failures.find("list_dir, read_file, write_file") != std::string::npos);
}

TEST_CASE("a cost budget on an unpriced model fails rather than passing quietly") {
    const TempDir dir{"ash_eval_unpriced"};

    const fs::path journal = dir.path() / "unpriced.jsonl";
    rewrite_header_model(fs::path{ASH_SOURCE_DIR} / "examples" / "journals" / "openai.jsonl", journal,
                         "a-model-with-no-published-price");

    ash::eval::Suite suite;
    suite.name = "unpriced";
    ash::eval::Job job;
    job.id = "no-price";
    job.journal = journal;
    // A budget no real run could exceed, so the only thing that can fail this
    // job is the missing price.
    job.checks.max_cost_usd = 1000.0;
    suite.jobs.push_back(std::move(job));

    const ash::eval::SuiteReport report = ash::eval::run_suite(suite);

    REQUIRE(report.jobs.size() == 1);
    const ash::eval::JobResult& result = report.jobs.front();
    CHECK_FALSE(result.passed);
    CHECK_FALSE(result.cost_usd.has_value());
    CHECK_FALSE(report.every_cost_known());
    CHECK(join_failures(result.failures).find("no price") != std::string::npos);
}

TEST_CASE("a journal that will not replay fails its job instead of throwing") {
    ash::eval::Suite suite;
    suite.name = "missing";
    ash::eval::Job job;
    job.id = "absent";
    job.journal = fs::path{ASH_SOURCE_DIR} / "examples" / "journals" / "does-not-exist.jsonl";
    suite.jobs.push_back(std::move(job));

    ash::eval::SuiteReport report;
    REQUIRE_NOTHROW(report = ash::eval::run_suite(suite));

    REQUIRE(report.jobs.size() == 1);
    CHECK_FALSE(report.jobs.front().passed);
    CHECK(join_failures(report.jobs.front().failures).find("replay failed") != std::string::npos);
    CHECK(report.model_call_latencies_us.empty());
}

TEST_CASE("latency percentiles use nearest rank") {
    ash::eval::SuiteReport report;
    report.model_call_latencies_us = {5, 1, 4, 2, 3};

    CHECK(report.latency_percentile_us(0) == 1);
    CHECK(report.latency_percentile_us(50) == 3);
    CHECK(report.latency_percentile_us(80) == 4);
    CHECK(report.latency_percentile_us(95) == 5);
    CHECK(report.latency_percentile_us(100) == 5);

    // Percentiles of nothing are zero, not a crash on an empty vector.
    report.model_call_latencies_us.clear();
    CHECK(report.latency_percentile_us(50) == 0);
}

TEST_CASE("a malformed suite is rejected with the offending field named") {
    const TempDir dir{"ash_eval_bad_suite"};

    SECTION("an unknown check key") {
        const fs::path path = dir.path() / "typo.json";
        write_text(path, R"({"jobs": [{"id": "a", "task": "t", "journal": "j.jsonl",
                                        "check": {"final_contians": ["x"]}}]})");
        CHECK_THROWS_WITH(ash::eval::load_suite(path), Catch::Matchers::ContainsSubstring("final_contians"));
    }

    SECTION("a suite with no jobs") {
        const fs::path path = dir.path() / "empty.json";
        write_text(path, R"({"jobs": []})");
        CHECK_THROWS_WITH(ash::eval::load_suite(path), Catch::Matchers::ContainsSubstring("cannot tell you anything"));
    }

    SECTION("a job with no id") {
        const fs::path path = dir.path() / "noid.json";
        write_text(path, R"({"jobs": [{"id": "", "task": "t", "journal": "j.jsonl"}]})");
        CHECK_THROWS_WITH(ash::eval::load_suite(path), Catch::Matchers::ContainsSubstring("id cannot be empty"));
    }

    SECTION("malformed JSON") {
        const fs::path path = dir.path() / "broken.json";
        write_text(path, "{ this is not json");
        CHECK_THROWS_WITH(ash::eval::load_suite(path), Catch::Matchers::ContainsSubstring("not valid JSON"));
    }

    SECTION("a missing file") {
        CHECK_THROWS_WITH(ash::eval::load_suite(dir.path() / "absent.json"),
                          Catch::Matchers::ContainsSubstring("cannot open suite"));
    }
}
