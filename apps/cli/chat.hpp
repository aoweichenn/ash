#pragma once

#include <string>
#include <vector>

namespace ash::cli {

// The interactive session: `ash` with no arguments, or `ash chat`.
//
// `args` is the whole argv as main.cpp received it, so args[0] is the command
// name in both spellings. A bare `ash` is turned into `ash chat` before it gets
// here rather than being a second entry point -- one command means one place
// where the session starts.
[[nodiscard]] int chat_command(const std::vector<std::string>& args);

}  // namespace ash::cli
