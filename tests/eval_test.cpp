#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
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
#include "report.hpp"

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

ash::eval::JobResult make_job(std::string id, bool passed, int steps = 1, int tokens = 10) {
    ash::eval::JobResult result;
    result.id = std::move(id);
    result.passed = passed;
    result.stop_reason = "completed";
    result.steps = steps;
    result.prompt_tokens = tokens;
    result.completion_tokens = 0;
    if (!passed) {
        result.failures.push_back("expected something else");
    }
    return result;
}

ash::eval::SuiteReport report_of(std::vector<ash::eval::JobResult> jobs) {
    ash::eval::SuiteReport report;
    report.suite = "scripted";
    report.jobs = std::move(jobs);
    return report;
}

ash::eval::JobDelta delta_for(const ash::eval::ReportDiff& diff, const std::string& id) {
    const auto found = std::find_if(diff.jobs.begin(), diff.jobs.end(),
                                    [&](const ash::eval::JobDelta& delta) { return delta.id == id; });
    REQUIRE(found != diff.jobs.end());
    return *found;
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

TEST_CASE("a suite replayed in parallel produces the same report as a serial one") {
    ash::eval::Suite suite;
    suite.name = "ordering";

    // Six jobs over the same journal with distinct ids. Replayed at once they
    // finish in some other order, so an id list that comes back in suite order
    // is the ordering guarantee doing its job rather than the scheduler's luck.
    for (int i = 0; i < 6; ++i) {
        ash::eval::Job job;
        job.id = "job-" + std::to_string(i);
        job.journal = fs::path{ASH_SOURCE_DIR} / "examples" / "journals" / "openai.jsonl";
        job.checks.stop_reason = "completed";
        suite.jobs.push_back(std::move(job));
    }

    const nlohmann::json serial = nlohmann::json(ash::eval::run_suite(suite, 1));
    const nlohmann::json parallel = nlohmann::json(ash::eval::run_suite(suite, 6));
    const nlohmann::json per_core = nlohmann::json(ash::eval::run_suite(suite, 0));

    // Byte for byte, not field by field: the claim is that --jobs is invisible
    // in the report, and nothing that legitimately differs is serialized --
    // wall_us is measured and deliberately left out of the file.
    CHECK(serial.dump() == parallel.dump());
    CHECK(serial.dump() == per_core.dump());

    for (std::size_t i = 0; i < suite.jobs.size(); ++i) {
        CHECK(parallel.at("jobs").at(i).at("id").get<std::string>() == suite.jobs[i].id);
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

TEST_CASE("a report round-trips through the file it is written to") {
    const TempDir dir{"ash_eval_report"};
    const fs::path path = dir.path() / "report.json";

    const ash::eval::SuiteReport original = ash::eval::run_suite(ash::eval::load_suite(core_suite()));
    ash::eval::write_report(original, path);

    // A cost has to be a JSON number. nlohmann routes a braced scalar through
    // its initializer_list constructor, so `json{0.001}` is the array [0.001],
    // which comes back as a different report rather than as an error.
    const nlohmann::json raw = nlohmann::json::parse(read_text(path));
    REQUIRE(raw.at("jobs").size() == 2);
    CHECK(raw.at("jobs").at(0).at("cost_usd").is_number());

    const ash::eval::SuiteReport loaded = ash::eval::read_report(path);
    CHECK(loaded.suite == original.suite);
    REQUIRE(loaded.jobs.size() == original.jobs.size());
    for (std::size_t i = 0; i < loaded.jobs.size(); ++i) {
        CHECK(loaded.jobs[i].id == original.jobs[i].id);
        CHECK(loaded.jobs[i].passed == original.jobs[i].passed);
        CHECK(loaded.jobs[i].stop_reason == original.jobs[i].stop_reason);
        CHECK(loaded.jobs[i].steps == original.jobs[i].steps);
        CHECK(loaded.jobs[i].prompt_tokens == original.jobs[i].prompt_tokens);
        CHECK(loaded.jobs[i].completion_tokens == original.jobs[i].completion_tokens);
        CHECK(loaded.jobs[i].cost_usd == original.jobs[i].cost_usd);
        CHECK(loaded.jobs[i].model_latency_us == original.jobs[i].model_latency_us);
    }
    CHECK(loaded.model_call_latencies_us == original.model_call_latencies_us);
    CHECK(loaded.total_cost_usd() == original.total_cost_usd());
    CHECK(loaded.latency_percentile_us(95) == original.latency_percentile_us(95));
}

TEST_CASE("a job with no known price round-trips as an explicit null") {
    const TempDir dir{"ash_eval_report_null"};
    const fs::path path = dir.path() / "report.json";

    ash::eval::SuiteReport report = report_of({make_job("unpriced", true)});
    // make_job leaves cost_usd empty, which is the case being tested.
    ash::eval::write_report(report, path);

    const nlohmann::json raw = nlohmann::json::parse(read_text(path));
    CHECK(raw.at("jobs").at(0).at("cost_usd").is_null());

    const ash::eval::SuiteReport loaded = ash::eval::read_report(path);
    REQUIRE(loaded.jobs.size() == 1);
    CHECK_FALSE(loaded.jobs.front().cost_usd.has_value());
    CHECK_FALSE(loaded.every_cost_known());
}

TEST_CASE("a report from an unknown format version is refused") {
    const TempDir dir{"ash_eval_report_version"};
    const fs::path path = dir.path() / "future.json";
    write_text(path, R"({"version": 99, "suite": "x", "jobs": []})");

    CHECK_THROWS_WITH(ash::eval::read_report(path), Catch::Matchers::ContainsSubstring("version 99"));
}

TEST_CASE("the baseline diff names what moved and what broke") {
    SECTION("identical reports are unchanged, and that is not a regression") {
        const ash::eval::SuiteReport report = report_of({make_job("a", true), make_job("b", true)});
        const ash::eval::ReportDiff diff = ash::eval::diff_reports(report, report);

        CHECK(delta_for(diff, "a").kind == ash::eval::ChangeKind::kUnchanged);
        CHECK_FALSE(diff.regressed());
        CHECK(diff.notes.empty());
    }

    SECTION("a job that passed and now fails is a regression") {
        const ash::eval::ReportDiff diff = ash::eval::diff_reports(report_of({make_job("a", true)}),
                                                                   report_of({make_job("a", false)}));

        const ash::eval::JobDelta delta = delta_for(diff, "a");
        CHECK(delta.kind == ash::eval::ChangeKind::kRegressed);
        CHECK(diff.regressed());
        CHECK(join_failures(delta.notes).find("was passing, now fails") != std::string::npos);
    }

    SECTION("a job that failed and now passes is an improvement") {
        const ash::eval::ReportDiff diff = ash::eval::diff_reports(report_of({make_job("a", false)}),
                                                                   report_of({make_job("a", true)}));

        CHECK(delta_for(diff, "a").kind == ash::eval::ChangeKind::kImproved);
        CHECK_FALSE(diff.regressed());
    }

    SECTION("a job that still passes but costs more is changed, not a regression") {
        const ash::eval::ReportDiff diff = ash::eval::diff_reports(
            report_of({make_job("a", true, 2, 100)}), report_of({make_job("a", true, 5, 250)}));

        const ash::eval::JobDelta delta = delta_for(diff, "a");
        CHECK(delta.kind == ash::eval::ChangeKind::kChanged);
        CHECK_FALSE(diff.regressed());
        const std::string notes = join_failures(delta.notes);
        CHECK(notes.find("steps 2 -> 5") != std::string::npos);
        CHECK(notes.find("tokens 100 -> 250") != std::string::npos);
    }

    SECTION("a new job is reported and is not a regression") {
        const ash::eval::ReportDiff diff = ash::eval::diff_reports(
            report_of({make_job("a", true)}), report_of({make_job("a", true), make_job("b", true)}));

        CHECK(delta_for(diff, "b").kind == ash::eval::ChangeKind::kAdded);
        CHECK_FALSE(diff.regressed());
        CHECK(join_failures(diff.notes).find("passing 1/1 -> 2/2") != std::string::npos);
    }

    SECTION("deleting a failing job is still a regression") {
        // Removing the job that fails would otherwise make the suite look
        // better, which is the hole a baseline exists to close.
        const ash::eval::ReportDiff diff = ash::eval::diff_reports(
            report_of({make_job("a", true), make_job("b", false)}), report_of({make_job("a", true)}));

        CHECK(delta_for(diff, "b").kind == ash::eval::ChangeKind::kRemoved);
        CHECK(diff.regressed());
    }
}
