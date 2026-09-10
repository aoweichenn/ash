#include "chat.hpp"

#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "ash/model/providers.hpp"
#include "ash/model/stream.hpp"
#include "ash/runtime.hpp"
#include "ash/tool/builtin.hpp"
#include "ash/tool/permission.hpp"
#include "ash/tool/shell.hpp"
#include "ash/tool/tool.hpp"
#include "console.hpp"
#include "logic.hpp"
#include "options.hpp"
#include "terminal.hpp"

namespace ash::cli {
namespace {

// Long enough for a real command line to be read in full before it is
// approved, short enough that asking about a large file write does not fill the
// screen with the file.
constexpr std::size_t kApprovalPreview = 200;

void print_chat_usage() {
    std::cout << R"(ash -- the interactive session

usage:
  ash [options]                  start a session in the current directory
  ash chat [options]             the same thing, spelled out

options:
  --provider <name>      "openai" (default) or "anthropic"
  --base-url <url>       endpoint root, default $ASH_BASE_URL
  --api-key <key>        default $ASH_API_KEY (prefer the environment variable)
  --model <id>           model id, default $ASH_MODEL
  --system <prompt>      replace the built-in coding-agent prompt
  --max-steps <n>        model turns per message before giving up, default 16
  --max-tokens <n>       response token cap, default provider-chosen
  --no-stream            wait for the whole answer instead of showing it arrive
  --mode <name>          ask (default), edits, or yolo

in a session:
  /mode [ask|edits|yolo]   show or change what has to be confirmed
  /clear                   forget the conversation so far
  /help                    this list
  /exit                    leave (also /quit, or an end of input)

The whole conversation is kept in memory and sent with every message, and there
is no compaction: /clear is the only way to make it shorter.
)";
}

void print_session_help() {
    std::cout << R"(  /mode [ask|edits|yolo]   show or change what has to be confirmed
  /clear                   forget the conversation so far
  /help                    this list
  /exit                    leave (also /quit, or an end of input)

anything else you type goes to the model.
)";
}

// The session's answer to "may I?".
//
// The mode is the user's standing policy; `allowed_` is what they have said
// "always" about since the session started. Both belong to the session and not
// to the tools, which stay immutable -- that is the property that lets one
// registry be shared across threads, and it is not worth spending to save a
// capture.
//
// The reader is the session's, and this is called from inside the agent loop.
// `sync_wait` runs the loop on the thread that called it, which is the same
// thread that reads the prompt, so there is one reader and it is never used
// from two places at once.
class Gate {
public:
    Gate(PermissionMode mode, LineReader& reader, bool interactive)
        : mode_(mode), reader_(reader), interactive_(interactive) {}

    [[nodiscard]] PermissionMode mode() const noexcept { return mode_; }

    void set_mode(PermissionMode mode) noexcept { mode_ = mode; }

    bool approve(std::string_view tool, const nlohmann::json& arguments) {
        if (!should_require_approval(mode_, tool)) {
            return true;
        }
        if (allowed_.count(std::string{tool}) != 0) {
            return true;
        }

        std::cout << "\n  " << tool << " " << summarize(arguments.dump(), kApprovalPreview) << "\n";
        std::cout << "  allow? [y]es / [n]o / [a]lways for this tool ";
        // A prompt with no newline only makes sense where someone is watching
        // it; in a pipe it would run into whatever is printed next.
        std::cout << (interactive_ ? "> " : "\n") << std::flush;

        std::string answer;
        if (reader_.read_line(answer) != LineReader::Result::kLine) {
            // No answer is not consent. This is the end of the input, or an
            // interruption, and both of them mean "stop", not "go ahead".
            std::cout << "\n";
            return false;
        }
        if (answer == "a" || answer == "A") {
            allowed_.insert(std::string{tool});
            return true;
        }
        return answer == "y" || answer == "Y";
    }

private:
    PermissionMode mode_;
    LineReader& reader_;
    bool interactive_;
    std::set<std::string> allowed_;
};

}  // namespace

int chat_command(const std::vector<std::string>& args) {
    RunOptions options;
    options.base_url = env_or("ASH_BASE_URL", std::string{kDefaultBaseUrl});
    options.model = env_or("ASH_MODEL", std::string{kDefaultModel});
    options.api_key = env_or("ASH_API_KEY", "");
    // Unlike `run`, the session shows the answer arriving. Watching it work is
    // most of what an interactive agent is for.
    options.stream = true;

    switch (parse_run_options(args, options, /*require_task=*/false, print_chat_usage)) {
        case ParseResult::kHelp:
            return 0;
        case ParseResult::kError:
            return 2;
        case ParseResult::kOk:
            break;
    }

    if (!options.journal_path.empty()) {
        // Refused rather than ignored. A journal header holds one task and
        // `ash replay` expects to consume every event in it, so a multi-turn
        // recording would either fail to replay or -- much worse -- replay and
        // verify, which is the one thing the determinism story cannot afford.
        std::cerr << "ash: a session cannot record a journal; a journal holds one task.\n"
                     "ash: for a recording, use `ash run --journal <path> \"<task>\"`.\n";
        return 2;
    }

    PermissionMode mode = PermissionMode::kAsk;
    if (!options.mode.empty()) {
        const std::optional<PermissionMode> wanted = parse_mode(options.mode);
        if (!wanted.has_value()) {
            std::cerr << "ash: unknown mode '" << options.mode << "'; try ask, edits or yolo\n";
            return 2;
        }
        mode = *wanted;
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

    // The shell is registered here and nowhere else, because this is the only
    // command with someone at the keyboard to ask. `run` gets the read-only
    // three for exactly the same reason.
    ash::ToolRegistry tools = ash::make_builtin_tools();
    tools.add(ash::make_shell_tool());

    ConsoleSink sink;
    LineReader reader;
    const bool interactive = is_a_terminal(STDIN_FILENO);

    Gate gate{mode, reader, interactive};
    tools = ash::make_permission_registry(
        tools, [&gate](std::string_view tool, const nlohmann::json& arguments) {
            return gate.approve(tool, arguments);
        });

    // Printing goes on the outside of the gate, so what is about to be asked
    // about is on screen before the question is. Both are decorators over the
    // same registry and neither knows about the other.
    tools = make_printing_registry(tools, kResultSummaryLimit, kResultPreviewLines);

    ash::AgentOptions agent_options;
    agent_options.max_steps = options.max_steps;

    const std::string cwd = std::filesystem::current_path().string();
    agent_options.system_prompt =
        options.system_prompt.empty() ? build_system_prompt(cwd, tools) : options.system_prompt;

    std::cout << "ash: " << provider->name() << " / " << provider->model() << " -> " << options.base_url
              << "\n";
    std::cout << "ash: " << cwd << "\n";
    std::cout << "ash: " << mode_description(gate.mode()) << "\n";
    std::cout << "ash: /help for commands, /exit to leave\n\n";

    // The conversation so far, without the system message: the loop puts
    // options.system_prompt back at the front of every turn, so carrying it
    // here as well would send it twice.
    std::vector<ash::Message> body;

    for (;;) {
        if (interactive) {
            std::cout << "> " << std::flush;
        }

        std::string line;
        const LineReader::Result read = reader.read_line(line);

        if (read == LineReader::Result::kInterrupted) {
            // Ctrl-C on an empty prompt leaves. Interrupting a turn is a
            // different thing and is handled around the turn, not here.
            std::cout << "\n";
            return 0;
        }
        if (read == LineReader::Result::kEof) {
            if (interactive) {
                std::cout << "\n";
            }
            return 0;
        }
        if (!interactive) {
            // There is no prompt to echo, so say what was read: without it a
            // piped session's log is a wall of answers with no questions.
            std::cout << "ash> " << line << "\n";
        }

        if (line.empty()) {
            continue;
        }

        if (line.front() == '/') {
            const std::size_t space = line.find(' ');
            const std::string_view command{line.data(), space == std::string::npos ? line.size() : space};
            const std::string_view rest =
                space == std::string::npos ? std::string_view{} : std::string_view{line}.substr(space + 1);

            if (command == "/exit" || command == "/quit") {
                return 0;
            }
            if (command == "/clear") {
                body.clear();
                std::cout << "ash: forgotten\n";
                continue;
            }
            if (command == "/help") {
                print_session_help();
                continue;
            }
            if (command == "/mode") {
                if (rest.empty()) {
                    gate.set_mode(cycle_mode(gate.mode()));
                } else {
                    const std::optional<PermissionMode> wanted = parse_mode(rest);
                    if (!wanted.has_value()) {
                        std::cout << "ash: no mode '" << rest << "'; try ask, edits or yolo\n";
                        continue;
                    }
                    gate.set_mode(*wanted);
                }
                std::cout << "ash: " << mode_description(gate.mode()) << "\n";
                continue;
            }

            // Slash is a namespace this program owns. Nothing starting with one
            // is ever sent to the model, so a typo here is a typo and not a
            // question about slashes.
            std::cout << "ash: no command '" << command << "'; /help lists them\n";
            continue;
        }

        ash::AgentResult result;
        try {
            result = ash::run_agent(*provider, tools, body, line, agent_options, std::stop_token{},
                                    options.stream ? &sink : nullptr)
                         .sync_wait();
        } catch (const std::exception& error) {
            // One bad turn is not a reason to end the session: the transcript is
            // left as it was, so the next message starts from where this one
            // did rather than from a half-recorded failure.
            std::cerr << "ash: " << error.what() << "\n";
            continue;
        }

        if (!options.stream) {
            const std::string answer = last_assistant_text(result);
            if (!answer.empty()) {
                std::cout << "\n" << answer << "\n";
            }
        }
        // Tool activity was already printed as it happened, either way, so the
        // shared printer is asked only for the line it has not already said.
        print_result(result, /*shown_live=*/true);

        body = carry_forward_history(result);
    }
}

}  // namespace ash::cli
