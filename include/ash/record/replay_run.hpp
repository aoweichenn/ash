#pragma once

#include <cstddef>
#include <string_view>

#include "ash/record/journal.hpp"
#include "ash/runtime.hpp"

namespace ash {

struct ReplayOutcome {
    AgentResult result;
    std::size_t events_consumed = 0;
};

// Rebuilds everything a replay needs from the journal alone -- provider, tool
// set, and run options -- and re-runs the agent loop against it. The tools
// answer from the journal rather than performing any work, so a run that wrote
// files can be replayed safely.
//
// Throws ReplayError when the run takes a different path than the recording, or
// finishes without consuming all of it.
[[nodiscard]] ReplayOutcome replay_run(const Journal& journal, std::string_view actor = "root");

}  // namespace ash
