#include <charconv>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <filesystem>
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
#include "ash/runtime.hpp"
#include "ash/tool/builtin.hpp"
#include "ash/tool/tool.hpp"
#include "ash/version.hpp"

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
  ash version                    print the version
  ash help                       print this message

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

    ash::ReplayCursor cursor{journal, actor};
    ash::ReplayingProvider provider{cursor, journal.header().provider, journal.header().model};
    ash::ToolRegistry tools = ash::make_replaying_registry(journal, cursor, actor);

    // Rebuilt from the header so the requests match the recording exactly.
    ash::AgentOptions agent_options;
    if (!journal.header().system_prompt.empty()) {
        agent_options.system_prompt = journal.header().system_prompt;
    }
    agent_options.max_steps = journal.header().max_steps > 0 ? journal.header().max_steps : kDefaultMaxSteps;

    std::cout << "ash: replaying " << path.string() << " (" << provider.name() << " / " << provider.model()
              << ")\n";
    std::cout << "ash: recorded " << journal.header().created_at << ", " << cursor.available()
              << " events for actor '" << actor << "'\n";
    std::cout << "ash: task: " << journal.header().task << "\n\n";

    const ash::AgentResult result =
        ash::run_agent(provider, tools, journal.header().task, agent_options).sync_wait();
    print_result(result);

    // A replay that did not consume the whole journal took a different path than
    // the recording, which is exactly the drift this tool exists to catch.
    cursor.verify_consumed();
    std::cout << "ash: replay verified, " << cursor.consumed() << " events consumed\n";

    return result.stop_reason == "completed" ? 0 : 1;
}

int dispatch(const std::vector<std::string>& args) {
    const std::string& command = args.front();
    if (command == "run") {
        return run_command(args);
    }
    if (command == "replay") {
        return replay_command(args);
    }
    if (command == "version" || command == "--version") {
        std::cout << "ash " << ash::version() << "\n";
        return 0;
    }
    if (command == "help" || command == "-h" || command == "--help") {
        print_usage();
        return 0;
    }
    if (command == "eval" || command == "trace") {
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
