#include "ash_eval.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

namespace ash::eval {

namespace {

std::string read_file(const std::filesystem::path& path) {
    std::ifstream input{path, std::ios::binary};
    if (!input) {
        throw std::runtime_error{"cannot open suite: " + path.string()};
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

// Every failure names the file it came from, because a suite is data and a
// typo in it should not read like a bug in the harness.
[[noreturn]] void fail(const std::filesystem::path& path, const std::string& message) {
    throw std::runtime_error{"suite " + path.string() + ": " + message};
}

std::vector<std::string> string_list(const nlohmann::json& json, const char* field, const std::filesystem::path& path) {
    if (!json.is_array()) {
        fail(path, std::string{field} + " must be an array of strings");
    }
    std::vector<std::string> values;
    values.reserve(json.size());
    for (const auto& item : json) {
        if (!item.is_string()) {
            fail(path, std::string{field} + " must contain only strings");
        }
        values.push_back(item.get<std::string>());
    }
    return values;
}

Checks parse_checks(const nlohmann::json& json, const std::filesystem::path& path) {
    if (!json.is_object()) {
        fail(path, "'check' must be an object");
    }

    Checks checks;
    for (const auto& [key, value] : json.items()) {
        if (key == "stop_reason") {
            checks.stop_reason = value.get<std::string>();
        } else if (key == "final_contains") {
            checks.final_contains = string_list(value, "final_contains", path);
        } else if (key == "tools_called") {
            checks.tools_called = string_list(value, "tools_called", path);
        } else if (key == "tools_forbidden") {
            checks.tools_forbidden = string_list(value, "tools_forbidden", path);
        } else if (key == "max_steps") {
            checks.max_steps = value.get<int>();
        } else if (key == "max_cost_usd") {
            checks.max_cost_usd = value.get<double>();
        } else if (key == "wrote_file") {
            FileWrite write;
            write.path = value.at("path").get<std::string>();
            write.content_contains = value.value("content_contains", "");
            checks.wrote_file = std::move(write);
        } else {
            // An unrecognised key is almost certainly a typo, and silently
            // ignoring it would make a suite that checks less than it reads.
            fail(path, "unknown check '" + key + "'");
        }
    }
    return checks;
}

Job parse_job(const nlohmann::json& json, const std::filesystem::path& suite_dir, const std::filesystem::path& path) {
    if (!json.is_object()) {
        fail(path, "each job must be an object");
    }

    Job job;
    job.id = json.at("id").get<std::string>();
    job.task = json.at("task").get<std::string>();
    job.actor = json.value("actor", "root");
    job.journal = suite_dir / json.at("journal").get<std::string>();
    if (json.contains("check")) {
        job.checks = parse_checks(json.at("check"), path);
    }
    if (job.id.empty()) {
        fail(path, "a job id cannot be empty");
    }
    return job;
}

}  // namespace

Suite load_suite(const std::filesystem::path& path) {
    const nlohmann::json parsed = nlohmann::json::parse(read_file(path), nullptr, false);
    if (parsed.is_discarded()) {
        fail(path, "not valid JSON");
    }
    if (!parsed.is_object()) {
        fail(path, "the document must be an object");
    }

    Suite suite;
    suite.name = parsed.value("name", path.stem().string());

    if (!parsed.contains("jobs") || !parsed.at("jobs").is_array()) {
        fail(path, "'jobs' must be an array");
    }

    const std::filesystem::path suite_dir = path.has_parent_path() ? path.parent_path() : std::filesystem::path{"."};
    for (const auto& entry : parsed.at("jobs")) {
        suite.jobs.push_back(parse_job(entry, suite_dir, path));
    }
    if (suite.jobs.empty()) {
        fail(path, "a suite with no jobs cannot fail, and so cannot tell you anything");
    }
    return suite;
}

}  // namespace ash::eval
