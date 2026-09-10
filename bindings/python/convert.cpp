#include "convert.hpp"

#include <cmath>
#include <cstdint>
#include <string>

namespace ash::python {

py::object py_from_json(const nlohmann::json& value) {
    switch (value.type()) {
        case nlohmann::json::value_t::null:
            return py::none();
        case nlohmann::json::value_t::boolean:
            return py::bool_(value.get<bool>());
        case nlohmann::json::value_t::number_integer:
            return py::int_(value.get<std::int64_t>());
        case nlohmann::json::value_t::number_unsigned:
            return py::int_(value.get<std::uint64_t>());
        case nlohmann::json::value_t::number_float:
            return py::float_(value.get<double>());
        case nlohmann::json::value_t::string:
            return py::str(value.get_ref<const std::string&>());
        case nlohmann::json::value_t::array: {
            py::list out;
            for (const auto& item : value) {
                out.append(py_from_json(item));
            }
            return out;
        }
        case nlohmann::json::value_t::object: {
            // Key order is nlohmann's, which is sorted -- the same order the
            // journal writes and the same order a request hash is taken over, so
            // a dict built here prints the way the recording does.
            py::dict out;
            for (auto item = value.begin(); item != value.end(); ++item) {
                out[py::str(item.key())] = py_from_json(item.value());
            }
            return out;
        }
        case nlohmann::json::value_t::binary:
        case nlohmann::json::value_t::discarded:
            break;
    }
    throw py::type_error("cannot convert a " + std::string(value.type_name()) +
                         " json value to Python");
}

nlohmann::json json_from_py(const py::handle& value) {
    if (value.is_none()) {
        return nlohmann::json(nullptr);
    }

    // Bool first: Python's True is an int, so testing for an integer before
    // testing for a bool would quietly turn every flag into 1 or 0.
    if (PyBool_Check(value.ptr())) {
        return value.cast<bool>();
    }

    if (PyLong_Check(value.ptr())) {
        int overflow = 0;
        const long long signed_value = PyLong_AsLongLongAndOverflow(value.ptr(), &overflow);
        if (overflow == 0) {
            return static_cast<std::int64_t>(signed_value);
        }
        if (overflow > 0) {
            // Positive and past int64. JSON can hold it as an unsigned, so it is
            // kept rather than refused; it is still bounded at 64 bits, which is
            // where "the value survived the trip" stops being true.
            const unsigned long long unsigned_value = PyLong_AsUnsignedLongLong(value.ptr());
            if (unsigned_value == static_cast<unsigned long long>(-1) && PyErr_Occurred()) {
                throw py::error_already_set();
            }
            return static_cast<std::uint64_t>(unsigned_value);
        }
        throw py::value_error("integer does not fit in 64 bits");
    }

    if (PyFloat_Check(value.ptr())) {
        const double number = value.cast<double>();
        // nlohmann writes a non-finite double as null, which is a value the
        // caller never asked for and would not notice replacing theirs.
        if (!std::isfinite(number)) {
            throw py::value_error("cannot convert a non-finite float to json");
        }
        return number;
    }

    if (PyUnicode_Check(value.ptr())) {
        return value.cast<std::string>();
    }

    if (PyList_Check(value.ptr()) || PyTuple_Check(value.ptr())) {
        nlohmann::json out = nlohmann::json::array();
        for (const auto& item : py::reinterpret_borrow<py::sequence>(value)) {
            out.push_back(json_from_py(item));
        }
        return out;
    }

    if (PyDict_Check(value.ptr())) {
        nlohmann::json out = nlohmann::json::object();
        for (const auto& item : py::reinterpret_borrow<py::dict>(value)) {
            // A non-string key is a type error and not a str(): "the keys are
            // strings" is part of what a JSON object is, and silently stringifying
            // would turn {1: "a"} into {"1": "a"} in a journal that then replays
            // differently from how it recorded.
            if (!PyUnicode_Check(item.first.ptr())) {
                throw py::type_error("json object keys must be strings, got " +
                                     py::repr(item.first).cast<std::string>());
            }
            out[item.first.cast<std::string>()] = json_from_py(item.second);
        }
        return out;
    }

    throw py::type_error("cannot convert a " +
                         py::str(py::type::of(value)).cast<std::string>() + " to json");
}

}  // namespace ash::python
