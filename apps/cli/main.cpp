#include <charconv>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "ash/model/providers.hpp"
#include "ash/record/decorators.hpp"
#include "ash/record/journal.hpp"
#include "ash/record/replay.hpp"
#include "ash/record/replay_run.hpp"
#include "ash/runtime.hpp"
#include "ash/tool/builtin.hpp"
#include "ash/tool/tool.hpp"
#include "ash/version.hpp"
#include "report.hpp"

namespace {

constexpr std::string_view kDefaultBaseUrl = "https://api.deepseek.com/v1";
constexpr std::string_view kDefaultModel = "deepseek-chat";

// M1 runs one logical task per process, so it has a single actor. Journals are
// keyed by actor already, so concurrent runs need no format change.
constexpr std::string_view kActor = "root";

constexpr int kDefaultMaxSteps = 16;

struct RunOptions {
    std::string task;
    std::string provider = "openai";
    std::string base_url;
    std::string api_key;
    std::string model;
    std::string system_prompt;
    std::string journal_path;
    int max_steps = kDefaultMaxSteps;
    int max_tokens = 0;  // 0 means "let the provider decide"
};

void print_usage() {
    std::cout << R"(ash -- a deterministic agent runtime

usage:
  ash run [options] <task>       run an agent, optionally recording a journal
  ash replay <journal>           re-run a recorded run offline, with no API key
  ash eval [options]             grade a suite of recorded runs, offline
  ash version                    print the version
  ash help                       print this message

options for `eval`:
  --suite <path>         the suite to run (required)
  --jobs <n>             replay n jobs at once; 0 means one per core (default 1)
  --baseline <path>      a report from an earlier run; regressions set exit 1
  --json <path>          write this run's report, for use as a later baseline

options for `run`:
  --task <text>          the task to run (may also be a positional argument)
  --journal <path>       record every model and tool call to this file
  --provider <name>      "openai" (default) or "anthropic"
  --base-url <url>       endpoint root, default $ASH_BASE_URL
  --api-key <key>        default $ASH_API_KEY (prefer the environment variable)
  --model <id>           model id, default $ASH_MODEL
  --system <prompt>      override the system prompt
  --max-steps <n>        model turns before giving up, default 16
  --max-tokens <n>       response token cap, default provider-chosen

examples:
  export ASH_API_KEY=sk-...
  ash run --journal demo.jsonl "list the files here, then summarize this project"
  ash replay demo.jsonl          # same output, no key, no network, no cost
  ash eval --suite examples/suites/core.json --json /tmp/report.json
)";
}

[[nodiscard]] std::string env_or(std::string_view name, std::string fallback) {
    const std::string key{name};
    const char* value = std::getenv(key.c_str());
    if (value == nullptr || *value == '\0') {
        return fallback;
    }
    return std::string{value};
}

[[nodiscard]] bool parse_int(const std::string& text, int& out) {
    const char* first = text.data();
    const char* last = text.data() + text.size();
    const auto result = std::from_chars(first, last, out);
    return result.ec == std::errc{} && result.ptr == last;
}

enum class ParseResult { kOk, kHelp, kError };

ParseResult parse_run_options(const std::vector<std::string>& args, RunOptions& options) {
    for (std::size_t i = 1; i < args.size(); ++i) {
        const std::string& arg = args[i];

        auto take_value = [&](std::string& slot) {
            if (i + 1 >= args.size()) {
                std::cerr << "ash: " << arg << " requires a value\n";
                return false;
            }
            slot = args[++i];
            return true;
        };

        auto take_int = [&](int& slot) {
            std::string text;
            if (!take_value(text)) {
                return false;
            }
            if (!parse_int(text, slot)) {
                std::cerr << "ash: " << arg << " expects an integer, got '" << text << "'\n";
                return false;
            }
            return true;
        };

        if (arg == "-h" || arg == "--help") {
            print_usage();
            return ParseResult::kHelp;
        }
        if (arg == "--task") {
            if (!take_value(options.task)) return ParseResult::kError;
        } else if (arg == "--journal") {
            if (!take_value(options.journal_path)) return ParseResult::kError;
        } else if (arg == "--provider") {
            if (!take_value(options.provider)) return ParseResult::kError;
        } else if (arg == "--base-url") {
            if (!take_value(options.base_url)) return ParseResult::kError;
        } else if (arg == "--api-key") {
            if (!take_value(options.api_key)) return ParseResult::kError;
        } else if (arg == "--model") {
            if (!take_value(options.model)) return ParseResult::kError;
        } else if (arg == "--system") {
            if (!take_value(options.system_prompt)) return ParseResult::kError;
        } else if (arg == "--max-steps") {
            if (!take_int(options.max_steps)) return ParseResult::kError;
        } else if (arg == "--max-tokens") {
            if (!take_int(options.max_tokens)) return ParseResult::kError;
        } else if (!arg.empty() && arg.front() == '-' && arg != "-") {
            std::cerr << "ash: unknown option '" << arg << "'\n";
            return ParseResult::kError;
        } else if (options.task.empty()) {
            options.task = arg;
        } else {
            options.task += ' ';
            options.task += arg;
        }
    }

    if (options.task.empty()) {
        std::cerr << "ash: a task is required\n\n";
        print_usage();
        return ParseResult::kError;
    }
    return ParseResult::kOk;
}

// One line of a tool result, so a long listing does not bury the transcript.
[[nodiscard]] std::string summarize(std::string_view text, std::size_t limit = 88) {
    const auto newline = text.find('\n');
    std::string_view line = newline == std::string_view::npos ? text : text.substr(0, newline);
    const bool clipped = line.size() > limit || newline != std::string_view::npos;
    if (line.size() > limit) {
        line = line.substr(0, limit);
    }
    std::string out{line};
    if (clipped) {
        out += " ...";
    }
    return out;
}

[[nodiscard]] std::string last_assistant_text(const ash::AgentResult& result) {
    for (auto it = result.transcript.rbegin(); it != result.transcript.rend(); ++it) {
        if (it->role == ash::Role::kAssistant && !it->content.empty()) {
            return it->content;
        }
    }
    return {};
}

// Shared by `run` and `replay`, which is the point: a replayed run produces the
// same output because it goes through the same printing.
void print_result(const ash::AgentResult& result) {
    for (const auto& step : result.steps) {
        for (const auto& call : step.assistant.tool_calls) {
            std::cout << "  -> " << call.name << " " << summarize(call.arguments.dump(), 64) << "\n";
        }
        for (const auto& tool_result : step.tool_results) {
            std::cout << "  <- " << summarize(tool_result.content) << "\n";
        }
    }

    const std::string answer = last_assistant_text(result);
    if (!answer.empty()) {
        std::cout << "\n" << answer << "\n";
    }

    std::cout << "\nash: stop=" << result.stop_reason << " steps=" << result.steps.size() << " tokens="
              << result.usage.total_tokens() << " (prompt " << result.usage.prompt_tokens << ", completion "
              << result.usage.completion_tokens << ")\n";
}

[[nodiscard]] ash::ToolRegistry make_builtin_tools() {
    ash::ToolRegistry tools;
    tools.add(ash::make_read_file_tool());
    tools.add(ash::make_write_file_tool());
    tools.add(ash::make_list_dir_tool());
    return tools;
}

int run_command(const std::vector<std::string>& args) {
    RunOptions options;
    options.base_url = env_or("ASH_BASE_URL", std::string{kDefaultBaseUrl});
    options.model = env_or("ASH_MODEL", std::string{kDefaultModel});
    options.api_key = env_or("ASH_API_KEY", "");

    switch (parse_run_options(args, options)) {
        case ParseResult::kHelp:
            return 0;
        case ParseResult::kError:
            return 2;
        case ParseResult::kOk:
            break;
    }

    if (options.api_key.empty()) {
        std::cerr << "ash: no API key. Export ASH_API_KEY or pass --api-key.\n";
        return 2;
    }
    if (options.provider != "anthropic" && options.provider != "openai") {
        std::cerr << "ash: unknown provider '" << options.provider << "'\n";
        return 2;
    }

    ash::ProviderConfig config;
    config.base_url = options.base_url;
    config.api_key = options.api_key;
    config.model = options.model;
    if (options.max_tokens > 0) {
        config.max_tokens = options.max_tokens;
    }

    std::unique_ptr<ash::ModelProvider> provider = options.provider == "anthropic"
                                                      ? ash::make_anthropic(config)
                                                      : ash::make_openai_compatible(config);

    ash::AgentOptions agent_options;
    agent_options.max_steps = options.max_steps;
    if (!options.system_prompt.empty()) {
        agent_options.system_prompt = options.system_prompt;
    }

    std::optional<ash::Journal> journal;
    if (!options.journal_path.empty()) {
        journal.emplace(ash::Journal::create(options.journal_path));

        ash::JournalHeader header;
        header.provider = std::string{provider->name()};
        header.model = provider->model();
        header.task = options.task;
        header.system_prompt = agent_options.system_prompt;
        header.max_steps = agent_options.max_steps;
        journal->set_header(std::move(header));

        provider = std::make_unique<ash::RecordingProvider>(std::move(provider), *journal, std::string{kActor});
    }

    ash::ToolRegistry tools = make_builtin_tools();
    if (journal.has_value()) {
        tools = ash::make_recording_registry(tools, *journal, std::string{kActor});
    }

    std::cout << "ash: " << provider->name() << " / " << provider->model() << " -> " << options.base_url
              << "\n";
    if (journal.has_value()) {
        std::cout << "ash: recording to " << options.journal_path << "\n";
    }
    std::cout << "ash: task: " << options.task << "\n\n";

    const ash::AgentResult result =
        ash::run_agent(*provider, tools, options.task, agent_options).sync_wait();
    print_result(result);

    return result.stop_reason == "completed" ? 0 : 1;
}

int replay_command(const std::vector<std::string>& args) {
    if (args.size() < 2 || args[1] == "-h" || args[1] == "--help") {
        std::cerr << "usage: ash replay <journal>\n";
        return args.size() < 2 ? 2 : 0;
    }
    if (args.size() > 2) {
        std::cerr << "ash: replay takes exactly one journal\n";
        return 2;
    }

    const std::filesystem::path path = args[1];
    ash::Journal journal = ash::Journal::load(path);
    const std::string actor{kActor};

    std::size_t available = 0;
    for (const ash::Event& event : journal.events()) {
        if (event.actor == actor) {
            ++available;
        }
    }

    std::cout << "ash: replaying " << path.string() << " (" << journal.header().provider << " / "
              << journal.header().model << ")\n";
    std::cout << "ash: recorded " << journal.header().created_at << ", " << available
              << " events for actor '" << actor << "'\n";
    std::cout << "ash: task: " << journal.header().task << "\n\n";

    // The same wiring the eval harness uses, so a replay means one thing.
    // A run that takes a different path than the recording throws here, which
    // is exactly the drift this tool exists to catch.
    const ash::ReplayOutcome outcome = ash::replay_run(journal, actor);
    print_result(outcome.result);
    std::cout << "ash: replay verified, " << outcome.events_consumed << " events consumed\n";

    return outcome.result.stop_reason == "completed" ? 0 : 1;
}

struct EvalOptions {
    std::string suite_path;
    std::string baseline_path;
    std::string json_path;
    std::size_t jobs = 1;  // one replays here and starts no pool at all
};

void print_eval_usage() {
    std::cout << R"(usage: ash eval --suite <path> [--jobs <n>] [--json <path>] [--baseline <path>]

  --suite <path>         the suite to run (required)
  --jobs <n>             replay n jobs at once; 0 means one per core (default 1)
  --baseline <path>      a report from an earlier run; regressions set exit 1
  --json <path>          write this run's report, for use as a later baseline

Every job replays a journal, so this needs no API key, no network, and no
budget. Exit status is 0 only when every job passed and nothing regressed.

--jobs changes how long the suite takes and nothing else: the report is
assembled in suite order and every number in it comes from a recording, so a
parallel run and a serial one produce the same file.
)";
}

void print_report_table(const ash::eval::SuiteReport& report) {
    std::cout << "  " << std::left << std::setw(30) << "job" << std::setw(7) << "result" << std::right
              << std::setw(7) << "steps" << std::setw(8) << "tokens" << std::setw(12) << "cost" << std::setw(10)
              << "latency"
              << "\n";

    for (const ash::eval::JobResult& job : report.jobs) {
        std::cout << "  " << std::left << std::setw(30) << job.id << std::setw(7)
                  << (job.passed ? "pass" : "FAIL") << std::right << std::setw(7) << job.steps << std::setw(8)
                  << (job.prompt_tokens + job.completion_tokens) << std::setw(12)
                  << ash::eval::format_cost(job.cost_usd) << std::setw(10)
                  << ash::eval::format_us(job.model_latency_us) << "\n";
        for (const std::string& failure : job.failures) {
            std::cout << "      " << failure << "\n";
        }
    }

    std::cout << "\n  suite " << report.suite << ": " << report.passed() << "/" << report.jobs.size()
              << " passed\n";
    std::cout << "  tokens  " << (report.total_prompt_tokens() + report.total_completion_tokens())
              << " (prompt " << report.total_prompt_tokens() << ", completion " << report.total_completion_tokens()
              << ")\n";

    std::cout << "  cost    " << ash::eval::format_cost(report.total_cost_usd());
    if (!report.every_cost_known()) {
        std::cout << " (some jobs use a model with no published price)";
    }
    std::cout << "\n";

    // The latencies come from the recordings, not from this replay, so the
    // number is the one the original run really saw.
    std::cout << "  latency p50 " << ash::eval::format_us(report.latency_percentile_us(50)) << "  p95 "
              << ash::eval::format_us(report.latency_percentile_us(95)) << "  ("
              << report.model_call_latencies_us.size() << " recorded model calls)\n";
}

void print_diff(const ash::eval::ReportDiff& diff) {
    int unchanged = 0;
    for (const ash::eval::JobDelta& delta : diff.jobs) {
        if (delta.kind == ash::eval::ChangeKind::kUnchanged) {
            ++unchanged;
            continue;
        }
        std::cout << "  " << std::left << std::setw(30) << delta.id << " " << ash::eval::to_string(delta.kind)
                  << "\n";
        for (const std::string& note : delta.notes) {
            std::cout << "      " << note << "\n";
        }
    }
    for (const std::string& note : diff.notes) {
        std::cout << "  " << note << "\n";
    }

    if (unchanged > 0) {
        std::cout << "  " << unchanged << " job(s) unchanged\n";
    }
    std::cout << (diff.regressed() ? "  regressions found\n" : "  no regressions\n");
}

int eval_command(const std::vector<std::string>& args) {
    EvalOptions options;
    for (std::size_t i = 1; i < args.size(); ++i) {
        const std::string& arg = args[i];

        auto take_value = [&](std::string& slot) {
            if (i + 1 >= args.size()) {
                std::cerr << "ash: " << arg << " requires a value\n";
                return false;
            }
            slot = args[++i];
            return true;
        };

        if (arg == "-h" || arg == "--help") {
            print_eval_usage();
            return 0;
        }
        if (arg == "--suite") {
            if (!take_value(options.suite_path)) return 2;
        } else if (arg == "--baseline") {
            if (!take_value(options.baseline_path)) return 2;
        } else if (arg == "--json") {
            if (!take_value(options.json_path)) return 2;
        } else if (arg == "--jobs") {
            std::string value;
            if (!take_value(value)) return 2;
            try {
                const long parsed = std::stol(value);
                if (parsed < 0) {
                    std::cerr << "ash: --jobs cannot be negative\n";
                    return 2;
                }
                options.jobs = static_cast<std::size_t>(parsed);
            } catch (const std::exception&) {
                std::cerr << "ash: --jobs expects a number, got '" << value << "'\n";
                return 2;
            }
        } else {
            std::cerr << "ash: unknown option '" << arg << "'\n";
            return 2;
        }
    }

    if (options.suite_path.empty()) {
        std::cerr << "ash: --suite is required\n\n";
        print_eval_usage();
        return 2;
    }

    const ash::eval::Suite suite = ash::eval::load_suite(options.suite_path);
    const ash::eval::SuiteReport report = ash::eval::run_suite(suite, options.jobs);
    print_report_table(report);

    if (!options.json_path.empty()) {
        ash::eval::write_report(report, options.json_path);
        std::cout << "\nash: wrote " << options.json_path << "\n";
    }

    int status = report.failed() == 0 ? 0 : 1;

    if (!options.baseline_path.empty()) {
        const ash::eval::SuiteReport baseline = ash::eval::read_report(options.baseline_path);
        std::cout << "\n  baseline " << options.baseline_path;
        if (baseline.suite != report.suite) {
            // Comparing two different suites would produce noise dressed up as
            // findings, so say so and let the numbers stand on their own.
            std::cout << " (suite '" << baseline.suite << "', not '" << report.suite << "')";
        }
        std::cout << "\n";

        const ash::eval::ReportDiff diff = ash::eval::diff_reports(baseline, report);
        print_diff(diff);
        if (diff.regressed()) {
            status = 1;
        }
    }

    return status;
}

int dispatch(const std::vector<std::string>& args) {
    const std::string& command = args.front();
    if (command == "run") {
        return run_command(args);
    }
    if (command == "replay") {
        return replay_command(args);
    }
    if (command == "eval") {
        return eval_command(args);
    }
    if (command == "version" || command == "--version") {
        std::cout << "ash " << ash::version() << "\n";
        return 0;
    }
    if (command == "help" || command == "-h" || command == "--help") {
        print_usage();
        return 0;
    }
    if (command == "trace") {
        std::cerr << "ash: '" << command << "' is not implemented yet\n";
        return 2;
    }

    std::cerr << "ash: unknown command '" << command << "'\n\n";
    print_usage();
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    const std::vector<std::string> args(argv + 1, argv + argc);
    if (args.empty()) {
        print_usage();
        return 2;
    }

    try {
        return dispatch(args);
    } catch (const std::exception& error) {
        std::cerr << "ash: " << error.what() << "\n";
        return 1;
    }
}
