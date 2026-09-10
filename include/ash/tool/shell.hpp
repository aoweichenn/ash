#pragma once

#include <chrono>
#include <cstddef>
#include <memory>
#include <string>

#include "ash/tool/tool.hpp"

namespace ash {

struct ShellOptions {
    // What to keep, not what to read. Output past this is counted and dropped
    // rather than left in the pipe, because a child blocked writing into a full
    // pipe never exits.
    std::size_t max_output_bytes = 262144;

    std::chrono::milliseconds timeout{120000};

    std::string shell = "/bin/sh";
};

// Runs a command line through a shell and returns its combined output.
//
// The child gets an empty stdin and its own process group: empty so a command
// that reads cannot take the caller's input, and its own group so a command
// that spawns children can be stopped as a tree rather than leaving them
// behind. Output past ShellOptions::max_output_bytes is reported as truncated,
// a command that outlives the timeout is killed, and every one of those endings
// comes back as an error result rather than an exception -- a tool that throws
// unwinds straight out of the agent loop.
[[nodiscard]] std::shared_ptr<Tool> make_shell_tool(ShellOptions options = {});

}  // namespace ash
