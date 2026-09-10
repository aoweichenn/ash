#include "types.hpp"

#include <string>
#include <vector>

#include "ash/cancellation.hpp"
#include "ash/model/provider.hpp"
#include "ash/record/replay.hpp"

namespace ash::python {

namespace {

// Borrowed, not owned. The module dict holds the class for as long as the
// module lives, so a raw pointer is both sufficient and the only thing that is
// safe to keep: an owning py::object would need the GIL in its destructor,
// which can run while the interpreter is being torn down.
PyObject* g_cancelled = nullptr;

py::object answer_of(const Result& result) {
    // The loop ends "completed" on an assistant message with no tool calls, so
    // for the ordinary case this is that message. Taken as the last assistant
    // message rather than as the last message outright because a run stopped by
    // its step budget or by a cancellation ends on a tool result instead, and
    // the answer is still whatever the model last said.
    for (auto message = result.agent.transcript.rbegin();
         message != result.agent.transcript.rend();
         ++message) {
        if (message->role == Role::kAssistant) {
            return py::str(message->content);
        }
    }
    return py::str("");
}

py::list tools_called_of(const Result& result) {
    py::list called;
    for (const auto& message : result.agent.transcript) {
        if (message.role != Role::kAssistant) {
            continue;
        }
        for (const auto& call : message.tool_calls) {
            called.append(py::str(call.name));
        }
    }
    return called;
}

py::list steps_of(const Result& result) {
    // Built on access rather than up front. The loop already copies each step
    // out of the transcript, and a caller that only wants the answer -- which is
    // most of them, and every eval job -- should not pay a second copy.
    py::list steps;
    for (const auto& step : result.agent.steps) {
        py::dict entry;
        entry["assistant"] = to_python(step.assistant);
        entry["tool_results"] = to_python(step.tool_results);
        steps.append(std::move(entry));
    }
    return steps;
}

std::string describe(const Result& result) {
    std::string answer = answer_of(result).cast<std::string>();
    if (answer.size() > 40) {
        answer = answer.substr(0, 37) + "...";
    }
    std::string text = "Result(stop_reason=" + result.agent.stop_reason +
                       ", answer=" + py::repr(py::str(answer)).cast<std::string>();
    if (result.events_consumed.has_value()) {
        text += ", events_consumed=" + std::to_string(*result.events_consumed);
    }
    return text + ")";
}

}  // namespace

bool interpreter_running() {
    if (Py_IsInitialized() == 0) {
        return false;
    }
#if PY_VERSION_HEX >= 0x030D0000
    return Py_IsFinalizing() == 0;
#else
    // Py_IsFinalizing became public in 3.13; before that it is the same
    // function under its private name.
    return _Py_IsFinalizing() == 0;
#endif
}

void drop_reference(py::object& object) {
    if (!object) {
        return;
    }
    if (!interpreter_running()) {
        (void)object.release();  // deliberately not decremented
        return;
    }
    py::gil_scoped_acquire acquire;
    object = py::object();
}

py::handle cancelled_exception() { return py::handle(g_cancelled); }

bool is_cancelled(const py::error_already_set& error) {
    return g_cancelled != nullptr && error.matches(py::handle(g_cancelled));
}

void register_types(py::module_& m) {
    py::class_<Usage>(m, "Usage")
        .def(py::init<>())
        .def_readonly("prompt_tokens", &Usage::prompt_tokens)
        .def_readonly("completion_tokens", &Usage::completion_tokens)
        .def_property_readonly("total_tokens", &Usage::total_tokens)
        .def("__repr__", [](const Usage& usage) {
            return "Usage(prompt_tokens=" + std::to_string(usage.prompt_tokens) +
                   ", completion_tokens=" + std::to_string(usage.completion_tokens) + ")";
        });

    py::class_<Result>(m, "Result")
        .def_property_readonly(
            "stop_reason", [](const Result& result) { return result.agent.stop_reason; },
            "Why the run ended: completed, max_steps, or cancelled.")
        .def_property_readonly("answer", &answer_of, "The last thing the model said.")
        .def_property_readonly("transcript", [](const Result& result) {
            return to_python(result.agent.transcript);
        })
        .def_property_readonly("steps", &steps_of)
        .def_property_readonly("tools_called", &tools_called_of,
                               "The tools the model asked for, in the order it asked.")
        .def_property_readonly("usage", [](const Result& result) { return result.agent.usage; })
        .def_property_readonly("events_consumed", [](const Result& result) -> py::object {
            if (!result.events_consumed.has_value()) {
                return py::none();
            }
            return py::int_(*result.events_consumed);
        })
        .def("__repr__", &describe);

    // Registered so a replay that diverges from its recording can be caught by
    // itself. That is the point of the type: a caller that only had
    // RuntimeError could not tell drift from a crash.
    py::register_exception<ReplayError>(m, "ReplayError");

    // The runtime ends a cancelled run normally and reports it through
    // stop_reason, so this rarely escapes on its own. It is registered because a
    // callback raises it to stop a run from Python, and that needs a class with
    // a name.
    py::object cancelled = py::register_exception<Cancelled>(m, "Cancelled");
    g_cancelled = cancelled.ptr();
}

}  // namespace ash::python
