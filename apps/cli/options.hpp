#pragma once

// The flags `ash run` takes, and the parsing of them.
//
// They live here rather than in main.cpp because the interactive session takes
// the same flags meaning the same things: the same ASH_BASE_URL fallback, the
// same provider check, the same "0 means the provider decides". Two parsers
// would drift on exactly the details nobody looks at until a variable someone
// exported stops having an effect. The usage text sits beside them because the
// options are most of what it describes.

#include <charconv>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace ash::cli {

constexpr std::string_view kDefaultBaseUrl = "https://api.deepseek.com/v1";
constexpr std::string_view kDefaultModel = "deepseek-chat";

constexpr int kDefaultMaxSteps = 16;

struct RunOptions {
    std::string task;
    std::string provider = "openai";
    std::string base_url;
    std::string api_key;
    std::string model;
    std::string system_prompt;
    std::string journal_path;
    std::string mode;  // the session's; empty means the default
    int max_steps = kDefaultMaxSteps;
    int max_tokens = 0;  // 0 means "let the provider decide"
    bool stream = false;
};

inline void print_usage() {
    std::cout << R"(ash -- a deterministic agent runtime

usage:
  ash                            start a session in the current directory
  ash run [options] <task>       run one task, optionally recording a journal
  ash replay <journal>           re-run a recorded run offline, with no API key
  ash eval [options]             grade a suite of recorded runs, offline
  ash version                    print the version
  ash help                       print this message

options for `chat` (and for a bare `ash`):
  the same as `run` below, minus --task and --journal, plus --mode. Type /help
  once you are in one.

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
  --stream               print the answer as the model writes it
  --no-stream            the opposite, for a wrapper that always passes --stream

examples:
  export ASH_API_KEY=sk-...
  ash run --journal demo.jsonl "list the files here, then summarize this project"
  ash replay demo.jsonl          # same output, no key, no network, no cost
  ash eval --suite examples/suites/core.json --json /tmp/report.json
)";
}

[[nodiscard]] inline std::string env_or(std::string_view name, std::string fallback) {
    const std::string key{name};
    const char* value = std::getenv(key.c_str());
    if (value == nullptr || *value == '\0') {
        return fallback;
    }
    return std::string{value};
}

[[nodiscard]] inline bool parse_int(const std::string& text, int& out) {
    const char* first = text.data();
    const char* last = text.data() + text.size();
    const auto result = std::from_chars(first, last, out);
    return result.ec == std::errc{} && result.ptr == last;
}

enum class ParseResult { kOk, kHelp, kError };

// The usage text belongs to the command, so the command passes it in. `run` and
// the session accept overlapping flags with different defaults and different
// requirements, and a parser that printed one command's help from inside the
// other would be a bug nobody finds until they type --help.
using UsageFn = void (*)();

// `require_task` is the one difference that is not a default: a command that
// takes a task has to be told it is missing one, and a session that takes none
// must not be.
inline ParseResult parse_run_options(const std::vector<std::string>& args,
                                    RunOptions& options,
                                    bool require_task,
                                    UsageFn print_usage) {
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
        } else if (arg == "--mode") {
            if (!take_value(options.mode)) return ParseResult::kError;
        } else if (arg == "--stream") {
            options.stream = true;
        } else if (arg == "--no-stream") {
            options.stream = false;
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

    if (require_task && options.task.empty()) {
        std::cerr << "ash: a task is required\n\n";
        print_usage();
        return ParseResult::kError;
    }
    return ParseResult::kOk;
}

}  // namespace ash::cli
