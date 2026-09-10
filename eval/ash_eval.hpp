#pragma once

// The eval harness is a consumer of the runtime, not part of it. Nothing here
// lives under include/ash, and a test enforces that: a program embedding the
// runtime should not have to compile a suite parser or a price table.

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ash::eval {

// What a job asserts about a run. Every check is decided from the transcript,
// the tool calls and the recorded usage, so all of them hold when the job is
// replayed offline -- the harness never has to inspect the world to grade.
struct FileWrite {
    std::string path;              // matched against a write_file call's path
    std::string content_contains;  // empty means "any content"
};

struct Checks {
    std::optional<std::string> stop_reason;
    std::vector<std::string> final_contains;  // every one must appear
    std::vector<std::string> tools_called;    // exact, in order
    std::vector<std::string> tools_forbidden; // none may appear
    std::optional<FileWrite> wrote_file;
    std::optional<int> max_steps;
    std::optional<double> max_cost_usd;
};

struct Job {
    std::string id;
    std::string task;
    std::filesystem::path journal;  // resolved relative to the suite file
    std::string actor = "root";
    Checks checks;
};

struct Suite {
    std::string name;
    std::vector<Job> jobs;
};

// Throws std::runtime_error with the offending field named.
[[nodiscard]] Suite load_suite(const std::filesystem::path& path);

struct JobResult {
    std::string id;
    bool passed = false;
    std::vector<std::string> failures;  // empty when passed
    std::string stop_reason;
    int steps = 0;
    int prompt_tokens = 0;
    int completion_tokens = 0;
    std::optional<double> cost_usd;  // absent when the model has no known price
    std::int64_t model_latency_us = 0;  // summed over the run's model calls
    std::int64_t wall_us = 0;           // this replay, not the original run
};

struct SuiteReport {
    std::string suite;
    std::vector<JobResult> jobs;
    std::vector<std::int64_t> model_call_latencies_us;  // one per recorded call

    [[nodiscard]] int passed() const noexcept;
    [[nodiscard]] int failed() const noexcept;
    [[nodiscard]] int total_prompt_tokens() const noexcept;
    [[nodiscard]] int total_completion_tokens() const noexcept;
    [[nodiscard]] double total_cost_usd() const noexcept;  // priced jobs only
    [[nodiscard]] bool every_cost_known() const noexcept;
    // Nearest-rank, which is the definition that does not interpolate a value
    // no run actually had.
    [[nodiscard]] std::int64_t latency_percentile_us(double percentile) const;
};

// Replays every job from its journal. No network, no filesystem writes, no key.
[[nodiscard]] SuiteReport run_suite(const Suite& suite);

}  // namespace ash::eval
