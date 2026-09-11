#pragma once

// The session's decisions that are not I/O: which mode means what, when a tool
// call needs the user's consent, what one turn hands to the next, and what the
// agent is told it is.
//
// Pure on purpose. Each of these has an edge that is easy to get wrong and
// invisible once it is wrong -- a tool nobody has heard of that quietly runs
// itself, a history that carries an empty assistant turn into a request the
// real API rejects, a cwd that a run only appears to be using. A pure function
// is the only version of them that a test can pin down without a terminal and
// without a model.

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "ash/model/provider.hpp"
#include "ash/runtime.hpp"
#include "ash/tool/tool.hpp"

namespace ash::cli {

// How much the session asks before it acts.
//
// The middle one is `edits` and not `auto` because what it auto-approves is
// writing files; running a shell command still asks. A name that promised more
// than the mode does is a name that gets someone's project rewritten by a
// command they thought they had already consented to.
enum class PermissionMode { kAsk, kEdits, kYolo };

[[nodiscard]] inline std::string_view to_string(PermissionMode mode) noexcept {
    switch (mode) {
        case PermissionMode::kAsk:
            return "ask";
        case PermissionMode::kEdits:
            return "edits";
        case PermissionMode::kYolo:
            return "yolo";
    }
    return "ask";
}

// "auto" is taken as another spelling of yolo rather than rejected: it is the
// word people reach for, and a typo that silently leaves you in `ask` is a
// worse outcome than an alias that means what it sounds like.
[[nodiscard]] inline std::optional<PermissionMode> parse_mode(std::string_view name) noexcept {
    if (name == "ask") {
        return PermissionMode::kAsk;
    }
    if (name == "edits") {
        return PermissionMode::kEdits;
    }
    if (name == "yolo" || name == "auto") {
        return PermissionMode::kYolo;
    }
    return std::nullopt;
}

// What `/mode` with no argument does: cycle, so the whole setting is reachable
// from one command with no arguments to remember.
[[nodiscard]] inline PermissionMode cycle_mode(PermissionMode mode) noexcept {
    switch (mode) {
        case PermissionMode::kAsk:
            return PermissionMode::kEdits;
        case PermissionMode::kEdits:
            return PermissionMode::kYolo;
        case PermissionMode::kYolo:
            return PermissionMode::kAsk;
    }
    return PermissionMode::kAsk;
}

// One line, printed when the mode changes, saying what the mode now allows.
[[nodiscard]] inline std::string_view mode_description(PermissionMode mode) noexcept {
    switch (mode) {
        case PermissionMode::kAsk:
            return "ask -- every file write and every command is confirmed with you";
        case PermissionMode::kEdits:
            return "edits -- file writes go ahead; commands are still confirmed";
        case PermissionMode::kYolo:
            return "yolo -- nothing is confirmed; the agent acts on its own";
    }
    return "";
}

// Reads are never worth a prompt: they change nothing, and asking about them
// only teaches the user to hit `y` without reading.
[[nodiscard]] inline bool is_read_only_tool(std::string_view tool) noexcept {
    return tool == "read_file" || tool == "list_dir";
}

// Whether this call has to be put to the user first.
//
// Anything not recognised as read-only is asked about, which is the direction
// that fails safely: a tool added later is confirmed by default instead of
// running itself because nobody updated a list here.
[[nodiscard]] inline bool should_require_approval(PermissionMode mode, std::string_view tool) noexcept {
    if (is_read_only_tool(tool)) {
        return false;
    }
    if (mode == PermissionMode::kYolo) {
        return false;
    }
    if (mode == PermissionMode::kEdits && tool == "write_file") {
        return false;
    }
    return true;
}

// The body of the conversation so far, ready to open the next turn with.
//
// Three things happen here and all of them are load-bearing.
//
// The system message is dropped, because the loop puts `options.system_prompt`
// back at the front of every turn and carrying the old one too would send it
// twice. It is only dropped when it is actually a system message: an earlier
// turn's user text is not ours to discard, and a session that silently ate the
// first thing someone said would be a worse bug than a duplicate prompt.
//
// Then the unfinished tail goes: whatever is left at the end that the next
// request cannot open with. There are two shapes of it and they are one rule.
//
// An assistant message with no content and no tool calls is what the loop leaves
// behind when a step ends in nothing. The Anthropic encoder writes it as
// {"role":"assistant","content":[]}, which the real API refuses -- so left in, a
// single empty reply would break every later turn of the session and not just
// its own.
//
// A user message with no answer after it means the turn was interrupted before
// the model replied, and it goes for the same reason the reader drops a
// half-typed line on Ctrl-C: the interruption means "forget what I was saying".
// Keeping it would cost more than the words, because the next turn would then
// open with two user messages in a row -- which the API rejects, and it would
// reject every turn after that too, since the pair stays in the history. A model
// that answers with nothing at all leaves this same shape once its empty reply
// is dropped, which is why this is one rule and not two.
[[nodiscard]] inline std::vector<ash::Message> carry_forward_history(const ash::AgentResult& result) {
    std::size_t begin = 0;
    if (!result.transcript.empty() && result.transcript.front().role == ash::Role::kSystem) {
        begin = 1;
    }

    std::vector<ash::Message> body(result.transcript.begin() + static_cast<std::ptrdiff_t>(begin),
                                   result.transcript.end());

    while (!body.empty()) {
        const ash::Message& last = body.back();
        const bool empty_assistant =
            last.role == ash::Role::kAssistant && last.content.empty() && last.tool_calls.empty();
        const bool unanswered_user = last.role == ash::Role::kUser;
        if (!empty_assistant && !unanswered_user) {
            break;
        }
        body.pop_back();
    }

    return body;
}

// What the agent is told it is.
//
// The tool list is read off the registry rather than written out here, so the
// prompt cannot describe a tool that is not installed or miss one that is.
// `cwd` is in it because a tool's relative paths resolve against the process,
// and a model that has not been told where that is will guess.
[[nodiscard]] inline std::string build_system_prompt(std::string_view cwd, const ash::ToolRegistry& tools) {
    std::string names;
    for (const auto& tool : tools.tools()) {
        if (!names.empty()) {
            names += ", ";
        }
        names += tool->name();
    }
    if (names.empty()) {
        names = "nothing yet";
    }

    std::string prompt;
    prompt += "You are ash, a coding agent running on the user's own machine, in the directory ";
    prompt += cwd;
    prompt += ". Relative paths you pass to a tool resolve against that directory.\n\n";

    prompt += "You are in a conversation rather than running one task: the user will follow up, "
              "correct you, and ask for more, so each answer is one turn of a longer exchange and "
              "you keep the context of the ones before it.\n\n";

    prompt += "Your tools are: ";
    prompt += names;
    prompt += ".\n\n";

    prompt += "How to work:\n";
    prompt += "- Look before you change anything. Read the file you are about to edit and list the "
              "directory you are about to write into. A guess about what is on disk costs the user "
              "more than a tool call costs you.\n";
    prompt += "- Prefer running a command to predicting what it would print. If a build, a test, or "
              "a search can tell you the answer, find out rather than describing what you expect.\n";
    prompt += "- Leave the rest of the project alone. Do not reformat, reorganise, or tidy code you "
              "were not asked about, and do not claim something works that you have not run.\n";
    prompt += "- Say when you are done, and say what you changed. The user sees your tool calls but "
              "not your reasoning, so an answer that only describes intent leaves them to work out "
              "what actually happened.\n\n";

    prompt += "Some tool calls need the user's approval first and they may refuse. A refusal comes "
              "back as an ordinary tool result that says so, not as an error to retry: treat it as "
              "information about what the user wants and either find another way or ask them.";

    return prompt;
}

}  // namespace ash::cli
