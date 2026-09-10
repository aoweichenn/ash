#include "report.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ash::eval {

namespace {

std::string format_delta(std::int64_t before, std::int64_t after) {
    std::ostringstream out;
    out << before << " -> " << after;
    if (before != 0 && after != before) {
        const double percent = 100.0 * static_cast<double>(after - before) / static_cast<double>(before);
        out << " (" << (percent > 0 ? "+" : "") << std::fixed << std::setprecision(1) << percent << "%)";
    }
    return out.str();
}

std::string first_failure(const JobResult& result) {
    return result.failures.empty() ? std::string{"failed"} : result.failures.front();
}

JobDelta diff_job(const JobResult& before, const JobResult& after) {
    JobDelta delta;
    delta.id = after.id;

    if (before.passed != after.passed) {
        delta.kind = after.passed ? ChangeKind::kImproved : ChangeKind::kRegressed;
        delta.notes.push_back(after.passed ? "was failing, now passes"
                                           : "was passing, now fails: " + first_failure(after));
        // The rest of the numbers still moved, so they are still worth showing.
    } else {
        delta.kind = ChangeKind::kUnchanged;
    }

    if (before.stop_reason != after.stop_reason) {
        delta.kind = delta.kind == ChangeKind::kRegressed ? delta.kind : ChangeKind::kChanged;
        delta.notes.push_back("stop_reason " + before.stop_reason + " -> " + after.stop_reason);
    }
    if (before.steps != after.steps) {
        delta.kind = delta.kind == ChangeKind::kRegressed ? delta.kind : ChangeKind::kChanged;
        delta.notes.push_back("steps " + format_delta(before.steps, after.steps));
    }

    const int before_tokens = before.prompt_tokens + before.completion_tokens;
    const int after_tokens = after.prompt_tokens + after.completion_tokens;
    if (before_tokens != after_tokens) {
        delta.kind = delta.kind == ChangeKind::kRegressed ? delta.kind : ChangeKind::kChanged;
        delta.notes.push_back("tokens " + format_delta(before_tokens, after_tokens));
    }
    if (before.cost_usd != after.cost_usd) {
        delta.kind = delta.kind == ChangeKind::kRegressed ? delta.kind : ChangeKind::kChanged;
        delta.notes.push_back("cost " + format_cost(before.cost_usd) + " -> " + format_cost(after.cost_usd));
    }
    if (before.model_latency_us != after.model_latency_us) {
        delta.kind = delta.kind == ChangeKind::kRegressed ? delta.kind : ChangeKind::kChanged;
        delta.notes.push_back("recorded latency " + format_us(before.model_latency_us) + " -> " +
                              format_us(after.model_latency_us));
    }
    return delta;
}

}  // namespace

std::string format_us(std::int64_t microseconds) {
    std::ostringstream out;
    if (microseconds < 1000) {
        out << microseconds << "us";
        return out.str();
    }
    out << std::fixed << std::setprecision(microseconds < 10'000'000 ? 3 : 2)
        << static_cast<double>(microseconds) / 1'000'000.0 << "s";
    return out.str();
}

std::string format_cost(const std::optional<double>& usd) {
    if (!usd.has_value()) {
        return "unpriced";
    }
    if (*usd != 0.0 && *usd < 0.0001) {
        std::ostringstream out;
        out << "$" << std::scientific << std::setprecision(2) << *usd;
        return out.str();
    }
    std::ostringstream out;
    out << "$" << std::fixed << std::setprecision(6) << *usd;
    return out.str();
}

std::string_view to_string(ChangeKind kind) noexcept {
    switch (kind) {
        case ChangeKind::kUnchanged:
            return "unchanged";
        case ChangeKind::kChanged:
            return "changed";
        case ChangeKind::kImproved:
            return "improved";
        case ChangeKind::kRegressed:
            return "REGRESSED";
        case ChangeKind::kAdded:
            return "added";
        case ChangeKind::kRemoved:
            return "REMOVED";
    }
    return "unknown";
}

void to_json(nlohmann::json& json, const JobResult& result) {
    // Parentheses, not braces: nlohmann routes braced scalars through its
    // initializer_list constructor, so `json{0.001}` is the one-element array
    // [0.001] rather than a number. And the optional is spelled out because the
    // library would otherwise serialize it as a zero-or-one element array too.
    const nlohmann::json cost = result.cost_usd ? nlohmann::json(*result.cost_usd) : nlohmann::json(nullptr);

    json = nlohmann::json{{"id", result.id},
                          {"passed", result.passed},
                          {"failures", result.failures},
                          {"stop_reason", result.stop_reason},
                          {"steps", result.steps},
                          {"prompt_tokens", result.prompt_tokens},
                          {"completion_tokens", result.completion_tokens},
                          {"cost_usd", cost},
                          {"model_latency_us", result.model_latency_us}};
}

void from_json(const nlohmann::json& json, JobResult& result) {
    result.id = json.at("id").get<std::string>();
    result.passed = json.at("passed").get<bool>();
    result.failures = json.value("failures", std::vector<std::string>{});
    result.stop_reason = json.value("stop_reason", "");
    result.steps = json.value("steps", 0);
    result.prompt_tokens = json.value("prompt_tokens", 0);
    result.completion_tokens = json.value("completion_tokens", 0);
    result.cost_usd = json.contains("cost_usd") && !json.at("cost_usd").is_null()
                          ? std::optional<double>{json.at("cost_usd").get<double>()}
                          : std::nullopt;
    result.model_latency_us = json.value("model_latency_us", std::int64_t{0});
}

void to_json(nlohmann::json& json, const SuiteReport& report) {
    // Totals are recomputed from the jobs rather than stored, so a report can
    // never disagree with itself.
    json = nlohmann::json{{"version", kReportVersion},
                          {"suite", report.suite},
                          {"jobs", report.jobs},
                          {"model_call_latencies_us", report.model_call_latencies_us}};
}

void from_json(const nlohmann::json& json, SuiteReport& report) {
    const int version = json.value("version", 0);
    if (version != kReportVersion) {
        throw std::runtime_error{"report version " + std::to_string(version) + " is not version " +
                                 std::to_string(kReportVersion)};
    }
    report.suite = json.value("suite", "");
    report.jobs = json.at("jobs").get<std::vector<JobResult>>();
    report.model_call_latencies_us = json.value("model_call_latencies_us", std::vector<std::int64_t>{});
}

void write_report(const SuiteReport& report, const std::filesystem::path& path) {
    std::ofstream output{path, std::ios::binary | std::ios::trunc};
    if (!output) {
        throw std::runtime_error{"cannot write report: " + path.string()};
    }
    output << nlohmann::json(report).dump(2) << '\n';
    if (!output) {
        throw std::runtime_error{"failed while writing report: " + path.string()};
    }
}

SuiteReport read_report(const std::filesystem::path& path) {
    std::ifstream input{path, std::ios::binary};
    if (!input) {
        throw std::runtime_error{"cannot read report: " + path.string()};
    }
    const nlohmann::json parsed = nlohmann::json::parse(input, nullptr, false);
    if (parsed.is_discarded() || !parsed.is_object()) {
        throw std::runtime_error{"report is not a JSON object: " + path.string()};
    }
    try {
        return parsed.get<SuiteReport>();
    } catch (const std::exception& error) {
        throw std::runtime_error{"report " + path.string() + ": " + error.what()};
    }
}

bool ReportDiff::regressed() const noexcept {
    return std::any_of(jobs.begin(), jobs.end(), [](const JobDelta& delta) {
        return delta.kind == ChangeKind::kRegressed || delta.kind == ChangeKind::kRemoved;
    });
}

ReportDiff diff_reports(const SuiteReport& baseline, const SuiteReport& current) {
    ReportDiff diff;

    for (const JobResult& job : current.jobs) {
        const auto before = std::find_if(baseline.jobs.begin(), baseline.jobs.end(),
                                         [&](const JobResult& candidate) { return candidate.id == job.id; });
        if (before == baseline.jobs.end()) {
            JobDelta delta;
            delta.id = job.id;
            delta.kind = ChangeKind::kAdded;
            delta.notes.push_back(job.passed ? "new job, passes" : "new job, fails: " + first_failure(job));
            diff.jobs.push_back(std::move(delta));
            continue;
        }
        diff.jobs.push_back(diff_job(*before, job));
    }

    for (const JobResult& job : baseline.jobs) {
        const auto present = std::find_if(current.jobs.begin(), current.jobs.end(),
                                          [&](const JobResult& candidate) { return candidate.id == job.id; });
        if (present == current.jobs.end()) {
            JobDelta delta;
            delta.id = job.id;
            delta.kind = ChangeKind::kRemoved;
            delta.notes.push_back(job.passed ? "job is gone" : "failing job is gone");
            diff.jobs.push_back(std::move(delta));
        }
    }

    if (baseline.passed() != current.passed() || baseline.jobs.size() != current.jobs.size()) {
        diff.notes.push_back("passing " + std::to_string(baseline.passed()) + "/" +
                             std::to_string(baseline.jobs.size()) + " -> " + std::to_string(current.passed()) +
                             "/" + std::to_string(current.jobs.size()));
    }

    const auto before_tokens = baseline.total_prompt_tokens() + baseline.total_completion_tokens();
    const auto after_tokens = current.total_prompt_tokens() + current.total_completion_tokens();
    if (before_tokens != after_tokens) {
        diff.notes.push_back("tokens " + format_delta(before_tokens, after_tokens));
    }

    // Both sides are recordings, so these only move when the journal itself
    // changed. Showing them anyway is what makes a re-recorded fixture visible.
    if (baseline.every_cost_known() && current.every_cost_known() &&
        baseline.total_cost_usd() != current.total_cost_usd()) {
        diff.notes.push_back("cost " + format_cost(baseline.total_cost_usd()) + " -> " +
                             format_cost(current.total_cost_usd()));
    }
    for (const double percentile : {50.0, 95.0}) {
        const std::int64_t before = baseline.latency_percentile_us(percentile);
        const std::int64_t after = current.latency_percentile_us(percentile);
        if (before != after) {
            diff.notes.push_back("latency p" + std::to_string(static_cast<int>(percentile)) + " " +
                                 format_us(before) + " -> " + format_us(after));
        }
    }

    return diff;
}

}  // namespace ash::eval
