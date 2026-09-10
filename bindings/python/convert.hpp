#pragma once

// The one place Python objects and the runtime's JSON meet.
//
// Every value that crosses the boundary goes through here, including the ones
// that have their own named structs. Those are rendered with the to_json the
// recorder and the replayer already share, so what Python sees is exactly what a
// journal would contain -- there is no second representation to keep in step,
// and a field that failed to serialize would be missing from both.
//
// All of these require the GIL.

#include <nlohmann/json.hpp>
#include <pybind11/pybind11.h>

namespace ash::python {

namespace py = pybind11;

// nlohmann -> Python. Objects become dicts, arrays become lists, numbers keep
// their width or exactness (integers stay integers, so nothing silently turns a
// token count into a float).
[[nodiscard]] py::object py_from_json(const nlohmann::json& value);

// Python -> nlohmann. Refuses rather than coerces: a bool is not an integer, an
// integer too large for the JSON number model is an error rather than a
// truncation, a dict key that is not a string is an error rather than a str(),
// and a float that is not finite is an error rather than a null. Passing
// something the runtime cannot represent is a bug in the caller, and it should
// be told so at the call rather than three layers down.
[[nodiscard]] nlohmann::json json_from_py(const py::handle& value);

// The runtime's own types, rendered through their own to_json and then through
// the same converter everything else uses -- so a message read out of a
// transcript and a message read out of a journal are the same dict.
//
// This is the nlohmann idiom rather than a hand-written overload per type: the
// constructor finds to_json by argument-dependent lookup, which is what keeps a
// type added to the runtime later from needing anything added here.
template <class T>
[[nodiscard]] py::object to_python(const T& value) {
    nlohmann::json json = value;
    return py_from_json(json);
}

}  // namespace ash::python
