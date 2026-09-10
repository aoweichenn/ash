#include "ash_eval.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <optional>
#include <string>
#include <vector>

#include "ash/cost.hpp"
#include "ash/model/provider.hpp"
#include "ash/record/journal.hpp"
#include "ash/record/replay_run.hpp"
#include "ash/runtime.hpp"

namespace ash::eval {

namespace {

using WallClock = std::chrono::steady_clock;

std::int64_t elapsed_us(WallClock::time_point start) noexcept {
    return std::chrono::duration_cast<std::chrono::microseconds>(WallClock::now() - start).count();
}

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

// The tool calls the loop actually made, in order, taken from the run rather
// than from the journal so that the check grades the run and not the recording.
std::vector<const ToolCall*> tool_calls_of(const AgentResult& result) {
    std::vector<const ToolCall*> calls;
    for (const AgentStep& step : result.steps) {
        for (const ToolCall& call : step.assistant.tool_calls) {
            calls.push_back(&call);
        }
    }
    return calls;
}

const ToolCall* find_write(const AgentResult& result, const std::string& path) {
    for (const ToolCall* call : tool_calls_of(result)) {
        if (call->name != "write_file") {
            continue;
        }
        if (call->arguments.value("path", "") == path) {
            return call;
        }
    }
    return nullptr;
}

void check_tools_called(const Checks& checks, const AgentResult& result, std::vector<std::string>& failures) {
    if (checks.tools_called.empty()) {
        return;
    }
    std::vector<std::string> actual;
    for (const ToolCall* call : tool_calls_of(result)) {
        actual.push_back(call->name);
    }
    if (actual != checks.tools_called) {
        std::string expected;
        for (const std::string& name : checks.tools_called) {
            expected += (expected.empty() ? "" : ", ") + name;
        }
        std::string seen;
        for (const std::string& name : actual) {
            seen += (seen.empty() ? "" : ", ") + name;
        }
        failures.push_back("expected tools [" + expected + "] but the run called [" + seen + "]");
    }
}

void check_forbidden_tools(const Checks& checks, const AgentResult& result, std::vector<std::string>& failures) {
    for (const ToolCall* call : tool_calls_of(result)) {
        if (std::find(checks.tools_forbidden.begin(), checks.tools_forbidden.end(), call->name) !=
            checks.tools_forbidden.end()) {
            failures.push_back("the run called '" + call->name + "', which the suite forbids");
        }
    }
}

void check_wrote_file(const Checks& checks, const AgentResult& result, std::vector<std::string>& failures) {
    if (!checks.wrote_file) {
        return;
    }
    const ToolCall* write = find_write(result, checks.wrote_file->path);
    if (write == nullptr) {
        failures.push_back("no write_file call for '" + checks.wrote_file->path + "'");
        return;
    }
    if (!checks.wrote_file->content_contains.empty()) {
        const std::string content = write->arguments.value("content", "");
        if (!contains(content, checks.wrote_file->content_contains)) {
            failures.push_back("the write to '" + checks.wrote_file->path + "' does not contain '" +
                               checks.wrote_file->content_contains + "'");
        }
    }
}

void check_final_message(const Checks& checks, const AgentResult& result, std::vector<std::string>& failures) {
    if (checks.final_contains.empty()) {
        return;
    }
    if (result.transcript.empty()) {
        failures.push_back("the run produced no transcript");
        return;
    }
    const std::string& final_text = result.transcript.back().content;
    for (const std::string& needle : checks.final_contains) {
        if (!contains(final_text, needle)) {
            failures.push_back("the final message does not mention '" + needle + "'");
        }
    }
}

}  // namespace

int SuiteReport::passed() const noexcept {
    return static_cast<int>(std::count_if(jobs.begin(), jobs.end(), [](const JobResult& job) { return job.passed; }));
}

int SuiteReport::failed() const noexcept { return static_cast<int>(jobs.size()) - passed(); }

int SuiteReport::total_prompt_tokens() const noexcept {
    int total = 0;
    for (const JobResult& job : jobs) {
        total += job.prompt_tokens;
    }
    return total;
}

int SuiteReport::total_completion_tokens() const noexcept {
    int total = 0;
    for (const JobResult& job : jobs) {
        total += job.completion_tokens;
    }
    return total;
}

bool SuiteReport::every_cost_known() const noexcept {
    return std::all_of(jobs.begin(), jobs.end(), [](const JobResult& job) { return job.cost_usd.has_value(); });
}

double SuiteReport::total_cost_usd() const noexcept {
    double total = 0.0;
    for (const JobResult& job : jobs) {
        total += job.cost_usd.value_or(0.0);
    }
    return total;
}

std::int64_t SuiteReport::latency_percentile_us(double percentile) const {
    if (model_call_latencies_us.empty()) {
        return 0;
    }
    std::vector<std::int64_t> sorted = model_call_latencies_us;
    std::sort(sorted.begin(), sorted.end());

    // Nearest rank: the smallest value at or above the requested proportion,
    // so the answer is always a latency some call actually had.
    const double rank = percentile / 100.0 * static_cast<double>(sorted.size());
    auto index = static_cast<std::size_t>(rank);
    if (static_cast<double>(index) < rank) {
        ++index;
    }
    index = std::max<std::size_t>(1, std::min(index, sorted.size()));
    return sorted[index - 1];
}

SuiteReport run_suite(const Suite& suite) {
    SuiteReport report;
    report.suite = suite.name;

    for (const Job& job : suite.jobs) {
        JobResult result;
        result.id = job.id;

        // Latencies only reach the report once the journal has replayed, so a
        // job that diverged cannot contribute timings from a run that did not
        // happen.
        std::vector<std::int64_t> latencies;
        bool replayed = false;

        const WallClock::time_point start = WallClock::now();
        try {
            const Journal journal = Journal::load(job.journal);

            for (const Event& event : journal.events()) {
                if (event.actor != job.actor) {
                    continue;
                }
                if (const auto* model_call = std::get_if<ModelCallRecord>(&event.payload)) {
                    latencies.push_back(model_call->duration_us);
                }
            }

            const ReplayOutcome outcome = replay_run(journal, job.actor);
            const AgentResult& run = outcome.result;
            replayed = true;

            result.stop_reason = run.stop_reason;
            result.steps = static_cast<int>(run.steps.size());
            result.prompt_tokens = run.usage.prompt_tokens;
            result.completion_tokens = run.usage.completion_tokens;
            result.cost_usd = estimate_cost_usd(journal.header().model, run.usage.prompt_tokens,
                                                run.usage.completion_tokens);
            for (const std::int64_t latency : latencies) {
                result.model_latency_us += latency;
            }

            const Checks& checks = job.checks;
            if (checks.stop_reason && run.stop_reason != *checks.stop_reason) {
                result.failures.push_back("expected stop_reason '" + *checks.stop_reason + "' but got '" +
                                          run.stop_reason + "'");
            }
            if (checks.max_steps && result.steps > *checks.max_steps) {
                result.failures.push_back("took " + std::to_string(result.steps) + " steps, over the budget of " +
                                          std::to_string(*checks.max_steps));
            }
            check_final_message(checks, run, result.failures);
            check_tools_called(checks, run, result.failures);
            check_forbidden_tools(checks, run, result.failures);
            check_wrote_file(checks, run, result.failures);

            if (checks.max_cost_usd) {
                if (result.cost_usd) {
                    if (*result.cost_usd > *checks.max_cost_usd) {
                        result.failures.push_back("cost $" + std::to_string(*result.cost_usd) + " is over the budget");
                    }
                } else {
                    // Saying nothing would read as a pass.
                    result.failures.push_back("no price is known for model '" + journal.header().model +
                                              "', so the cost budget could not be checked");
                }
            }
        } catch (const std::exception& error) {
            // A journal that will not replay is a failed job, not a crashed
            // harness: the suite should report it and carry on to the others.
            result.failures.push_back(std::string{"replay failed: "} + error.what());
        }

        if (replayed) {
            report.model_call_latencies_us.insert(report.model_call_latencies_us.end(), latencies.begin(),
                                                  latencies.end());
        }

        result.wall_us = elapsed_us(start);
        result.passed = result.failures.empty();
        report.jobs.push_back(std::move(result));
    }

    return report;
}

}  // namespace ash::eval
