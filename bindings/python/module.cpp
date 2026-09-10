// The ash runtime, reachable from Python.
//
// This extension is a consumer of the runtime in the same sense the eval harness
// is: it sits outside include/ash and links the libraries, and nothing in the
// core knows it exists. A test enforces that, so a binding that needed a hook
// added to the runtime would fail the build rather than quietly widen the seam.

#include <pybind11/pybind11.h>

#include "types.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "Python bindings for ash, a deterministic agent runtime.";
    m.attr("__version__") = ASH_VERSION_STRING;

    // Order matters only in that a name has to exist before something else
    // refers to it: the exceptions before the results that can raise them, and
    // both the value types before the Agent whose constructor takes them.
    ash::python::register_types(m);
    ash::python::register_provider(m);
    ash::python::register_tools(m);
    ash::python::register_run(m);
}
