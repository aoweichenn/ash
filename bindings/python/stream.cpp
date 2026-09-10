#include "stream.hpp"

#include <exception>
#include <string>
#include <utility>
#include <variant>

#include "types.hpp"

namespace ash::python {

namespace {

// One event, as a dict. A dict rather than a class for the same reason the
// transcript is one: what a caller does with an event is look at two or three
// fields, and the keys here are the ones the runtime's own decoder names, so
// there is nothing to keep in step.
//
// Every event carries "type", including the ones whose name is already in the
// key set, because a handler written as a chain of isinstance checks over four
// classes is a worse thing to ask of Python than a dispatch on a string.
py::dict event_to_python(const StreamEvent& event) {
    py::dict rendered;
    std::visit(
        [&rendered](const auto& value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, TextDelta>) {
                rendered["type"] = "text";
                rendered["text"] = value.text;
            } else if constexpr (std::is_same_v<T, ToolCallDelta>) {
                rendered["type"] = "tool_call";
                rendered["index"] = value.index;
                rendered["id"] = value.id;
                rendered["name"] = value.name;
                // The arguments arrive in pieces the endpoint chose the cuts
                // for, so this field is a fragment and not JSON -- the same
                // reason the type it comes from is named for one.
                rendered["arguments"] = value.arguments_fragment;
            } else if constexpr (std::is_same_v<T, UsageDelta>) {
                rendered["type"] = "usage";
                rendered["prompt_tokens"] = value.usage.prompt_tokens;
                rendered["completion_tokens"] = value.usage.completion_tokens;
            } else {
                rendered["type"] = "done";
                rendered["finish_reason"] = value.finish_reason;
                rendered["model"] = value.model;
            }
        },
        event);
    return rendered;
}

}  // namespace

PythonSink::PythonSink(py::object on_text, py::object on_event, CancelToken& token)
    : on_text_(std::move(on_text)), on_event_(std::move(on_event)), token_(token) {
    // Asked once, here, rather than per event: this runs with the GIL held, on
    // the calling thread, while on_event below runs inside libcurl's callback
    // where every attribute lookup is time the socket is not being drained.
    if (on_text_.is_none()) {
        on_text_ = py::object();
    }
    if (on_event_.is_none()) {
        on_event_ = py::object();
    }
}

PythonSink::~PythonSink() {
    // Same guard as a tool's: the sink is a local of the run, so this normally
    // runs with the interpreter healthy and the GIL in hand. It is written the
    // long way because "normally" is not a guarantee -- a sink can be destroyed
    // while the interpreter is on its way down, and decrementing a refcount
    // then is worse than letting one reference go.
    drop_reference(on_event_);
    drop_reference(on_text_);
    drop_reference(error_value_);
    drop_reference(error_type_);
}

void PythonSink::on_event(const StreamEvent& event) {
    if (failed_) {
        // A callback has already raised. The transfer may still be delivering
        // the rest of a buffered body, and calling it again would raise a second
        // exception that overwrites the first -- the one worth reporting.
        return;
    }

    // The GIL is not held here: the run released it for the whole of its
    // duration, and this is being called from inside libcurl's write callback
    // on the thread that released it. Nothing below may throw past this point
    // except through a path that has already been caught.
    py::gil_scoped_acquire acquire;

    try {
        if (on_text_ && std::holds_alternative<TextDelta>(event)) {
            (void)on_text_(py::str(std::get<TextDelta>(event).text));
        }
        if (on_event_) {
            (void)on_event_(event_to_python(event));
        }
    } catch (py::error_already_set& error) {
        failed_ = true;
        // Raising ash.Cancelled is how a callback says "stop, this is not a
        // failure". Told apart from a real error by the class itself rather
        // than by the message, and no exception is kept: the run is about to
        // end as cancelled and there is nothing for Python to see.
        if (!is_cancelled(error)) {
            error_type_ = error.type();
            error_value_ = error.value();
        }
        token_.cancel();
    } catch (const std::exception& error) {
        // Not reachable from Python code, which raises error_already_set. It is
        // here because on_event is called from a C callback whose contract says
        // nothing may escape, and a std::exception that got out would cross a
        // C frame -- which is undefined behaviour rather than a traceback.
        failed_ = true;
        error_message_ = error.what();
        token_.cancel();
    } catch (...) {
        failed_ = true;
        error_message_ = "stream callback failed";
        token_.cancel();
    }
}

void PythonSink::rethrow_if_failed() const {
    if (error_type_) {
        // Rebuilt from the pieces rather than from a kept error_already_set:
        // that type's destructor needs the GIL, so holding one across the run
        // would put a refcount decrement on the wrong side of the boundary the
        // run deliberately crossed.
        PyErr_SetObject(error_type_.ptr(), error_value_ ? error_value_.ptr() : Py_None);
        throw py::error_already_set();
    }
    if (!error_message_.empty()) {
        throw std::runtime_error(error_message_);
    }
}

void register_stream(py::module_& m) {
    py::class_<CancelToken>(m, "CancelToken",
                            "Ends a run in progress. Can be cancelled from any thread.\n\n"
                            "Cancelling asks the run to stop rather than killing it: the run "
                            "returns normally with stop_reason == \"cancelled\" and keeps "
                            "whatever it had produced, including the journal it was writing. "
                            "The stop is seen at the next step boundary and by the HTTP "
                            "transfer within about a second, so it also works before the "
                            "model has produced anything -- which is when a run is most "
                            "likely to be stuck.\n\n"
                            "    token = ash.CancelToken()\n"
                            "    threading.Timer(30, token.cancel).start()   # a timeout\n"
                            "    run = agent.run(task, cancel=token)\n")
        .def(py::init<>())
        .def("cancel", &CancelToken::cancel,
             "Ask the run to stop. Safe to call from another thread, safe to call twice.")
        .def_property_readonly("cancelled", &CancelToken::cancelled,
                               "Whether a stop has been requested.")
        .def("__repr__", [](const CancelToken& token) {
            return std::string{"CancelToken(cancelled="} +
                   (token.cancelled() ? "True" : "False") + ")";
        });
}

}  // namespace ash::python
