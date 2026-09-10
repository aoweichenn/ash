#pragma once

// Reporting on top of the harness: the file a run writes, and the comparison
// between two of them. `--json` writes a report and `--baseline` reads one, so
// a report has to stay stable enough to diff across a commit -- which is why it
// carries a version rather than being compared blind.

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "ash_eval.hpp"

namespace ash::eval {

inline constexpr int kReportVersion = 1;

void to_json(nlohmann::json& json, const JobResult& result);
void from_json(const nlohmann::json& json, JobResult& result);
void to_json(nlohmann::json& json, const SuiteReport& report);
void from_json(const nlohmann::json& json, SuiteReport& report);

void write_report(const SuiteReport& report, const std::filesystem::path& path);
[[nodiscard]] SuiteReport read_report(const std::filesystem::path& path);

// Shared so that the table and the diff cannot disagree about what a number
// looks like.
[[nodiscard]] std::string format_us(std::int64_t microseconds);
[[nodiscard]] std::string format_cost(const std::optional<double>& usd);

enum class ChangeKind { kUnchanged, kChanged, kImproved, kRegressed, kAdded, kRemoved };

[[nodiscard]] std::string_view to_string(ChangeKind kind) noexcept;

struct JobDelta {
    std::string id;
    ChangeKind kind = ChangeKind::kUnchanged;
    std::vector<std::string> notes;  // empty when unchanged
};

struct ReportDiff {
    std::vector<JobDelta> jobs;
    std::vector<std::string> notes;  // whole-suite movements

    // A disappearing job counts as a regression. Deleting the job that fails is
    // the cheapest way to make a suite look better, and catching that is much
    // of the point of keeping a baseline.
    [[nodiscard]] bool regressed() const noexcept;
};

[[nodiscard]] ReportDiff diff_reports(const SuiteReport& baseline, const SuiteReport& current);

}  // namespace ash::eval
