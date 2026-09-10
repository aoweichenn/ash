#pragma once

// Everything the CLI draws on a terminal.
//
// It lives here rather than in the library because the library's headers are
// not allowed to reach for <iostream>: printing belongs to whoever is holding
// the terminal. Both `run` and the interactive session print through this, so
// they cannot drift apart on what a streamed answer looks like -- which is a
// thing the streaming check compares byte for byte.

#include <cstddef>
#include <iostream>
#include <memory>
#include <stop_token>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

#include <nlohmann/json.hpp>

#include "ash/model/stream.hpp"
#include "ash/runtime.hpp"
#include "ash/tool/tool.hpp"

namespace ash::cli {

// `run` prints one line per tool result: its transcript is a log to scan.
constexpr std::size_t kResultSummaryLimit = 88;
constexpr std::size_t kResultSummaryLines = 1;

// The session prints a short block instead. A command's output is the thing
// whoever typed the command is watching for, and a one-line summary of `ls`
// shows them the first file and nothing else.
constexpr std::size_t kResultPreviewLines = 6;

// The first `lines` lines of a tool result, each clipped to `limit` columns, so
// a long listing does not bury the transcript.
//
// The one-line form is spelled out rather than expressed as the general case,
// because it is what `run` has always printed and the recorded golden outputs
// are compared byte for byte -- including the detail that a trailing newline
// counts as a clip and a single short line with no newline at all does not.
[[nodiscard]] inline std::string summarize(std::string_view text, std::size_t limit = kResultSummaryLimit,
                                           std::size_t lines = kResultSummaryLines) {
    if (lines <= 1) {
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

    std::string out;
    std::size_t start = 0;
    for (std::size_t shown = 0; shown < lines; ++shown) {
        const std::size_t newline = text.find('\n', start);
        std::string_view line =
            newline == std::string_view::npos ? text.substr(start) : text.substr(start, newline - start);
        bool clipped = line.size() > limit;
        if (clipped) {
            line = line.substr(0, limit);
        }
        if (!out.empty()) {
            out += '\n';
        }
        out += line;

        if (newline == std::string_view::npos) {
            if (clipped) {
                out += " ...";
            }
            return out;
        }
        start = newline + 1;
    }

    // Whatever follows the last line shown is what the " ..." stands for. A
    // result that ended in a newline right here has nothing after it, and
    // saying it was clipped when nothing was dropped would be a small lie in
    // the one place the user is reading to find out what happened.
    if (start < text.size()) {
        out += " ...";
    }
    return out;
}

[[nodiscard]] inline std::string last_assistant_text(const ash::AgentResult& result) {
    for (auto it = result.transcript.rbegin(); it != result.transcript.rend(); ++it) {
        if (it->role == ash::Role::kAssistant && !it->content.empty()) {
            return it->content;
        }
    }
    return {};
}

// Writes the model's answer out as it is written.
//
// Unbuffered and without holding anything back to format later: a stream that
// waited for a whole line would look exactly like no stream at all for an
// answer that is one long line, which is most of them.
class ConsoleSink final : public ash::StreamSink {
public:
    void on_event(const ash::StreamEvent& event) override {
        if (const auto* delta = std::get_if<ash::TextDelta>(&event)) {
            std::cout << delta->text << std::flush;
            line_open_ = true;
            return;
        }
        // A step that said nothing -- a tool call -- must not leave a blank
        // line behind it, so the break is only written when there is a line to
        // break. Each streamed answer ends up on its own line.
        if (std::holds_alternative<ash::StreamDone>(event) && line_open_) {
            std::cout << "\n" << std::flush;
            line_open_ = false;
        }
    }

private:
    bool line_open_ = false;
};

// Prints tool activity the moment it happens.
//
// The loop reports its tool steps only once the run is over, so without this a
// streamed run would print the answer before the tool calls that produced it --
// the transcript would arrive after the thing it explains. This is the same
// seam the recorder uses, which is the point: presentation is a decorator here
// too, and the loop is not asked to know about it.
class PrintingTool final : public ash::Tool {
public:
    explicit PrintingTool(std::shared_ptr<ash::Tool> inner, std::size_t limit = kResultSummaryLimit,
                          std::size_t lines = kResultSummaryLines)
        : inner_(std::move(inner)), limit_(limit), lines_(lines) {}

    [[nodiscard]] std::string_view name() const noexcept override { return inner_->name(); }

    [[nodiscard]] std::string_view description() const noexcept override { return inner_->description(); }

    [[nodiscard]] const nlohmann::json& input_schema() const noexcept override {
        return inner_->input_schema();
    }

    ash::Task<ash::ToolResult> invoke(const nlohmann::json& arguments, std::stop_token stop) const override {
        std::cout << "  -> " << name() << " " << summarize(arguments.dump(), 64) << "\n" << std::flush;
        ash::ToolResult result = co_await inner_->invoke(arguments, stop);
        std::cout << "  <- " << summarize(result.content, limit_, lines_) << "\n" << std::flush;
        co_return result;
    }

private:
    std::shared_ptr<ash::Tool> inner_;
    std::size_t limit_;
    std::size_t lines_;
};

[[nodiscard]] inline ash::ToolRegistry make_printing_registry(
    const ash::ToolRegistry& tools, std::size_t limit = kResultSummaryLimit,
    std::size_t lines = kResultSummaryLines) {
    ash::ToolRegistry printing;
    for (const auto& tool : tools.tools()) {
        printing.add(std::make_shared<PrintingTool>(tool, limit, lines));
    }
    return printing;
}

// Shared by `run` and `replay`, which is the point: a replayed run produces the
// same output because it goes through the same printing.
//
// `shown_live` means the run already printed its transcript as it happened, so
// printing it again here would say everything twice. The summary line is always
// printed: it is only known once the run is over.
inline void print_result(const ash::AgentResult& result, bool shown_live = false) {
    if (!shown_live) {
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
    }

    std::cout << "\nash: stop=" << result.stop_reason << " steps=" << result.steps.size() << " tokens="
              << result.usage.total_tokens() << " (prompt " << result.usage.prompt_tokens << ", completion "
              << result.usage.completion_tokens << ")\n";
}

}  // namespace ash::cli
