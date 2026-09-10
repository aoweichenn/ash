#include "types.hpp"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "ash/record/decorators.hpp"
#include "ash/record/journal.hpp"
#include "ash/record/replay_run.hpp"
#include "provider.hpp"
#include "tools.hpp"

namespace ash::python {

namespace {

// The actor every binding-driven run records under. One run from Python is one
// logical task, and there is no way to start a second one that would need a
// different name -- see the note on Agent::run about state being per call.
constexpr std::string_view kActor = "root";

Result replay_into_result(const std::string& path, const std::string& actor) {
    // Reading the journal is a file read and stays on this side of the release;
    // the run is what can block, and it is the whole point of the exercise that
    // it does not hold the GIL while it does.
    Journal journal = Journal::load(path);

    ReplayOutcome outcome;
    {
        py::gil_scoped_release release;
        outcome = replay_run(journal, actor);
    }

    Result result;
    result.agent = std::move(outcome.result);
    result.events_consumed = outcome.events_consumed;
    return result;
}

// A configured agent: a provider to ask, a set of tools to offer.
//
// It holds no run state at all. Everything a run needs -- the stop source, the
// sink, the error a callback raised -- is a local of run() below, which is what
// makes two Python threads sharing one Agent safe rather than merely unlikely to
// break: the provider opens a fresh connection per request and the tools are
// immutable, so there is nothing for two runs to disagree about.
class Agent {
public:
    Agent(Provider provider, ToolSet tools)
        : provider_(std::move(provider)), tools_(std::move(tools)) {}

    [[nodiscard]] Result run(const std::string& task,
                             const std::optional<std::string>& journal_path,
                             const std::optional<std::string>& system_prompt,
                             const std::optional<int>& max_steps) const {
        AgentOptions options;
        if (system_prompt.has_value()) {
            options.system_prompt = *system_prompt;
        }
        if (max_steps.has_value()) {
            options.max_steps = *max_steps;
        }

        // Both of these are copies: the shared_ptr so the provider cannot be
        // collected from Python mid-run, and the registry so the tools cannot
        // be either. Everything below is destroyed in reverse order at the end
        // of this function, and the run is over before any of it goes.
        std::shared_ptr<ModelProvider> provider = provider_.shared();
        ToolRegistry tools = tools_.registry();

        std::optional<Journal> journal;
        if (journal_path.has_value()) {
            journal.emplace(Journal::create(*journal_path));

            JournalHeader header;
            header.provider = std::string{provider->name()};
            header.model = provider->model();
            header.task = task;
            header.system_prompt = options.system_prompt;
            header.max_steps = options.max_steps;
            journal->set_header(std::move(header));

            // The key is in the provider's headers and nowhere else, so it has
            // no way into the journal -- which is what makes a recording safe to
            // commit. The test suite asserts it rather than trusting it.
            provider = std::make_shared<RecordingProvider>(std::move(provider), *journal,
                                                           std::string{kActor});
            tools = make_recording_registry(tools, *journal, std::string{kActor});
        }

        AgentResult outcome;
        {
            // Released for the length of the run. The coroutine body runs on
            // this thread, so every callback that needs Python -- and there are
            // none yet -- would re-acquire the GIL itself.
            py::gil_scoped_release release;
            outcome = run_agent(*provider, tools, task, options).sync_wait();
        }

        Result result;
        result.agent = std::move(outcome);
        return result;
    }

    [[nodiscard]] std::string describe() const {
        return "Agent(" + provider_.name() + " / " + provider_.model() + ", " +
               std::to_string(tools_.registry().size()) + " tools)";
    }

private:
    Provider provider_;
    ToolSet tools_;
};

}  // namespace

void register_run(py::module_& m) {
    py::class_<Agent>(m, "Agent",
                      "A provider and a set of tools, ready to be given a task.\n\n"
                      "Holds no run state, so one Agent may be used from several threads at "
                      "once and may be run more than once.")
        .def(py::init<Provider, ToolSet>(), py::arg("provider"), py::arg("tools") = ToolSet{})
        .def("run", &Agent::run, py::arg("task"), py::kw_only(),
             py::arg("journal") = py::none(), py::arg("system_prompt") = py::none(),
             py::arg("max_steps") = py::none(),
             R"(Run the agent loop until the model answers without calling a tool.

With journal= a path, everything the run could not recompute on its own -- every
model call and every tool call -- is written there as it happens, and the file
can be replayed later with ash.replay() without a network or an API key. The
recording is taken at the provider and tool seams, so it does not change what
the run does; a journal written from Python replays exactly like one written by
the CLI.

Stops at max_steps, or when the model stops asking for tools. The result's
stop_reason says which happened.
)")
        .def("__repr__", &Agent::describe);

    m.def(
        "replay", &replay_into_result, py::arg("journal"), py::arg("actor") = "root",
        R"(Re-runs a recorded run against its own journal, offline.

Rebuilds the provider, the tools and the run options from the journal and drives
the agent loop again, answering every model call and every tool call from the
recording. No network, no filesystem, no API key, no tokens spent -- and because
nothing was left to the model, the same journal always produces the same result.

Nothing checks that the replay is honest except the run itself: if it takes a
different path than the recording, or stops before consuming all of it,
ReplayError is raised. That is what makes a journal a fixture rather than a
sample: a test that replays one is testing the runtime, and it fails when the
runtime changes.

    run = ash.replay("examples/journals/openai.jsonl")
    assert run.stop_reason == "completed"
    assert run.tools_called == ["list_dir"]
)");
}

}  // namespace ash::python
