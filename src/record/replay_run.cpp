#include "ash/record/replay_run.hpp"

#include <string>

#include "ash/record/decorators.hpp"
#include "ash/record/replay.hpp"

namespace ash {

ReplayOutcome replay_run(const Journal& journal, std::string_view actor) {
    const std::string name{actor};

    ReplayCursor cursor{journal, name};
    ReplayingProvider provider{cursor, journal.header().provider, journal.header().model};
    ToolRegistry tools = make_replaying_registry(journal, cursor, name);

    // Rebuilt from the header so the requests match the recording exactly.
    AgentOptions options;
    if (!journal.header().system_prompt.empty()) {
        options.system_prompt = journal.header().system_prompt;
    }
    if (journal.header().max_steps > 0) {
        options.max_steps = journal.header().max_steps;
    }

    ReplayOutcome outcome;
    outcome.result = run_agent(provider, tools, journal.header().task, options).sync_wait();

    // A run that did not consume the whole journal took a different path than
    // the recording, which is the drift this whole mechanism exists to catch.
    cursor.verify_consumed();
    outcome.events_consumed = cursor.consumed();
    return outcome;
}

}  // namespace ash
