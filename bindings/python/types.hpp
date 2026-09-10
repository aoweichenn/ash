#pragma once

// What the binding's translation units share: the result object they all hand
// back, and the registration entry points module.cpp calls.
//
// The Python-visible classes are deliberately few. Anything that already has a
// to_json -- messages, tool calls, specs, eval jobs -- crosses as a dict, so
// Python sees the journal's own rendering of it and there is no second set of
// field names to keep aligned. The classes here are the ones a caller does real
// work with: the outcome of a run, and the token counts it cost.

#include <cstddef>
#include <optional>

#include <pybind11/pybind11.h>
// For std::optional and std::vector in signatures. pybind11 keeps these out of
// the core header so that a module which does not need them does not pay for
// the type casters.
#include <pybind11/stl.h>

#include "ash/runtime.hpp"
#include "convert.hpp"

namespace ash::python {

// One run, live or replayed. `events_consumed` is present only on a replay,
// because only a replay has a journal to have consumed.
struct Result {
    ash::AgentResult agent;
    std::optional<std::size_t> events_consumed;
};

// Whether it is safe to touch a Python reference at all: the interpreter is up
// and not on its way down.
//
// Py_IsInitialized() alone is not enough. It stays true throughout finalization,
// and taking the GIL while the interpreter is tearing down is worse than not
// taking it -- so anything that holds a reference across an unknown lifetime
// asks this before trying, and lets the reference go instead.
[[nodiscard]] bool interpreter_running();

// The registered ash.Cancelled class, borrowed from the module's own dict so
// nothing here holds a reference that would have to be released at interpreter
// shutdown. Valid once register_types has run.
[[nodiscard]] py::handle cancelled_exception();

// ash.Cancelled, raised by a callback, ends the run rather than failing it.
[[nodiscard]] bool is_cancelled(const py::error_already_set& error);

// Drops a reference to a Python object at a moment nobody controls: under the
// GIL when the interpreter can still give it, and abandoned when it cannot.
//
// Every destructor that holds a Python reference past the point where a GIL is
// guaranteed calls this instead of letting the member go on its own. Taking the
// GIL while the interpreter is tearing down is worse than not taking it, and a
// refcount decremented without one is undefined behaviour -- so the reference is
// leaked instead. One object, once, at exit, is a bounded cost; the alternative
// is not.
void drop_reference(py::object& object);

void register_types(py::module_& m);
void register_provider(py::module_& m);
void register_tools(py::module_& m);
void register_stream(py::module_& m);
void register_run(py::module_& m);
void register_eval(py::module_& m);

}  // namespace ash::python
