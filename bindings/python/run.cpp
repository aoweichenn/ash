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
#include "stream.hpp"
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
                             const std::optional<int>& max_steps,
                             CancelToken* cancel,
                             py::object on_text,
                             py::object on_event) const {
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

        // Every run has a stop source, whether or not the caller brought one,
        // because a callback that raises has to have something to cancel. The
        // caller's token, when there is one, is that same object -- so a timer
        // thread calling cancel() and a callback raising reach the same source.
        CancelToken local_token;
        CancelToken& token = cancel != nullptr ? *cancel : local_token;

        // Ctrl-C, for the length of the run. Built here, with the GIL held and
        // before the run lets go of it, because taking the signal means asking
        // Python two questions -- which thread this is, and what handler is
        // installed -- and both are answered with the GIL in hand.
        //
        // It cancels the same token as everything else, so a Ctrl-C and a timer
        // and a failing callback all arrive at the run the same way.
        InterruptWatch interrupt{token};

        // Installed only when asked for. The sink takes the GIL once per event
        // from inside libcurl's write callback, so a caller that wants the
        // answer and not the play-by-play should not pay for it -- and, more to
        // the point, should not be slowed down by whichever Python thread
        // happens to be holding the GIL at that moment.
        std::optional<PythonSink> sink;
        if (!on_text.is_none() || !on_event.is_none()) {
            sink.emplace(std::move(on_text), std::move(on_event), token);
        }

        AgentResult outcome;
        {
            // Released for the length of the run. The coroutine body runs on
            // this thread, so the callbacks above re-acquire the GIL themselves
            // and none of them is ever called from a thread the interpreter does
            // not already know.
            py::gil_scoped_release release;
            outcome = run_agent(*provider, tools, task, options, token.token(),
                                sink.has_value() ? &*sink : nullptr)
                          .sync_wait();
        }
        if (sink.has_value()) {
            // Whatever a callback raised. Raised here rather than where it
            // happened because the exception holds Python references, and this
            // is the first moment since the run began that the GIL is safely
            // back in hand.
            sink->rethrow_if_failed();
        }
        if (interrupt.interrupted() && outcome.stop_reason == "cancelled") {
            // Ctrl-C arrived and is what ended the run, so it is reported the
            // way Python reports it -- from here rather than from the handler,
            // which could not raise, and not from inside the run, which never
            // reached a bytecode boundary to raise at.
            //
            // Only when the run ended cancelled. A signal that arrived once the
            // loop had already produced its answer would otherwise turn a
            // completed run into an exception and throw away work that was done
            // and paid for; a run that completed is one the model had already
            // finished answering, so the Ctrl-C was a moment too late to mean
            // anything. Asked of the result rather than assumed from the flag,
            // because the two are set by different threads and the run can win
            // that race -- which is the right way round for it to be lost.
            PyErr_SetNone(PyExc_KeyboardInterrupt);
            throw py::error_already_set();
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
             py::arg("max_steps") = py::none(), py::arg("cancel") = py::none(),
             py::arg("on_text") = py::none(), py::arg("on_event") = py::none(),
             R"(Run the agent loop until the model answers without calling a tool.

With journal= a path, everything the run could not recompute on its own -- every
model call and every tool call -- is written there as it happens, and the file
can be replayed later with ash.replay() without a network or an API key. The
recording is taken at the provider and tool seams, so it does not change what
the run does; a journal written from Python replays exactly like one written by
the CLI.

Stops at max_steps, or when the model stops asking for tools. The result's
stop_reason says which happened.

on_text= is called with each piece of the model's answer as it arrives, and
on_event= with every event, as a dict whose "type" is one of text, tool_call,
usage or done. Watching a run changes nothing about it: the journal written by a
run with callbacks is byte for byte the journal written by the same run without
them, because the recording is the assembled response and not the frames.

Both are called on the thread performing the transfer, inside libcurl's write
callback, with the GIL released and re-acquired around the call. So a callback
must not block for long -- while it runs, nothing is reading the socket -- and
must not assume it is on the main thread. An exception raised in a callback
stops the run and is raised again from here, except for ash.Cancelled, which
ends the run as cancelled instead of failing it.

cancel= takes an ash.CancelToken and is the way to end a run from outside, from
any thread; see CancelToken for why it is not a callback.

Ctrl-C stops a run too, and arrives as KeyboardInterrupt. That is only true of a
run made on the main thread of a program that has not installed its own SIGINT
handler: a program that has one is using the signal for something, and a run
made on a worker thread is not where the signal is delivered. In those two cases
the signal is left alone and a run is stopped by cancel= or not at all.
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
