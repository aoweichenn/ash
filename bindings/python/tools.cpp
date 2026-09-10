#include "tools.hpp"

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ash/tool/tool.hpp"
#include "types.hpp"

namespace ash::python {

namespace {

// A Python callable, kept alive for as long as the tool that wraps it.
//
// The destructor is the whole reason this is a class rather than a bare
// py::object. Releasing the last reference to a Python object means
// decrementing a refcount, which needs the GIL, and a tool can be destroyed at
// a moment nobody controls -- when the last Agent holding it is collected, or
// while the interpreter is shutting down. The first case is handled by taking
// the GIL; the second cannot be, so the reference is let go instead. Leaking one
// object once at exit is bounded; touching a refcount during finalization is
// not.
class PyCallable {
public:
    explicit PyCallable(py::object function) : function_(std::move(function)) {}

    PyCallable(const PyCallable&) = delete;
    PyCallable& operator=(const PyCallable&) = delete;

    ~PyCallable() {
        if (!function_) {
            return;
        }
        if (!interpreter_running()) {
            (void)function_.release();  // deliberately not decremented
            return;
        }
        py::gil_scoped_acquire acquire;
        function_ = py::object();
    }

    [[nodiscard]] py::handle get() const { return function_; }

private:
    py::object function_;
};

// ---------------------------------------------------------------------------
// Deriving a schema from the function's own signature.
//
// The model can only call a tool it has been told about, so the declaration and
// the function have to agree. Reading the declaration off the signature is what
// makes them agree by construction. An annotation this cannot describe is
// refused rather than guessed at: a schema that is wrong in a plausible way
// produces a model that calls the tool correctly and gets an argument error it
// cannot see.
//
// Nothing here iterates a set or a dict of annotations, and the parameter order
// comes from the signature, so two derivations of the same function produce the
// same bytes. That is the same bar the rest of the project holds prompt assembly
// to, and it is what lets a test compare a generated schema against a literal.
// ---------------------------------------------------------------------------

// True when the object *is* the given builtin type. Pointer comparison and not
// an isinstance check: `isinstance(True, int)` is true, and a subtype of list
// has no schema of its own.
[[nodiscard]] bool is_type(const py::handle& object, PyTypeObject* type) {
    return object.ptr() == reinterpret_cast<PyObject*>(type);
}

// NoneType, which the headers do not name: there is Py_None and there is
// whatever type it happens to have.
[[nodiscard]] bool is_none_type(const py::handle& object) {
    return object.ptr() == reinterpret_cast<PyObject*>(Py_TYPE(Py_None));
}

[[nodiscard]] nlohmann::json schema_for_annotation(const py::handle& annotation);

[[nodiscard]] nlohmann::json schema_for_plain(const py::handle& annotation) {
    if (is_type(annotation, &PyUnicode_Type)) {
        return {{"type", "string"}};
    }
    // Before int. bool is a subclass of int, so a check the other way round
    // would describe every flag as an integer.
    if (is_type(annotation, &PyBool_Type)) {
        return {{"type", "boolean"}};
    }
    if (is_type(annotation, &PyLong_Type)) {
        return {{"type", "integer"}};
    }
    if (is_type(annotation, &PyFloat_Type)) {
        return {{"type", "number"}};
    }
    if (is_type(annotation, &PyList_Type) || is_type(annotation, &PyTuple_Type)) {
        return {{"type", "array"}};
    }
    if (is_type(annotation, &PyDict_Type)) {
        return {{"type", "object"}};
    }
    if (is_none_type(annotation)) {
        // A parameter annotated None, with no default and nothing else said
        // about it. An empty schema is what "anything" looks like, and it is
        // what the model should be told.
        return nlohmann::json::object();
    }
    if (annotation.is(py::module_::import("typing").attr("Any"))) {
        return nlohmann::json::object();
    }
    throw py::type_error(
        "cannot describe the annotation " + py::repr(annotation).cast<std::string>() +
        "; use str, int, float, bool, list[...], dict[str, ...], Literal[...], or Optional[...]");
}

// Optional[X] and X | None are the only unions with a description: one member
// plus None, meaning "X, or leave it out". A wider union has no single schema,
// and saying so beats silently picking one arm.
[[nodiscard]] nlohmann::json schema_for_union(const py::object& members) {
    std::vector<py::object> present;
    for (py::handle member : members) {
        if (!is_none_type(member)) {
            present.push_back(py::reinterpret_borrow<py::object>(member));
        }
    }
    if (present.size() != 1) {
        throw py::type_error(
            "cannot describe a union of more than one type; use Optional[X] or a single type");
    }
    return schema_for_annotation(present.front());
}

[[nodiscard]] nlohmann::json schema_for_annotation(const py::handle& annotation) {
    if (!annotation || annotation.is_none()) {
        return nlohmann::json::object();
    }

    const py::object typing = py::module_::import("typing");
    const py::object origin = typing.attr("get_origin")(annotation);

    if (origin.is_none()) {
        return schema_for_plain(annotation);
    }

    const py::object members = typing.attr("get_args")(annotation);

    // types.UnionType is the `X | None` spelling and only exists from 3.10; on
    // 3.9 Optional[X] is the only way to write a union, and it arrives as
    // typing.Union.
    const py::object union_spelling =
        py::getattr(py::module_::import("types"), "UnionType", py::none());
    if (origin.is(typing.attr("Union")) ||
        (!union_spelling.is_none() && origin.is(union_spelling))) {
        return schema_for_union(members);
    }

    if (origin.is(typing.attr("Literal"))) {
        nlohmann::json values = nlohmann::json::array();
        std::string common_type;
        bool one_type = true;
        for (py::handle member : members) {
            nlohmann::json value = json_from_py(member);
            std::string kind;
            if (value.is_boolean()) {
                kind = "boolean";
            } else if (value.is_string()) {
                kind = "string";
            } else if (value.is_number_integer()) {
                kind = "integer";
            } else if (value.is_number_float()) {
                kind = "number";
            } else {
                throw py::type_error("Literal values must be strings, numbers, or booleans");
            }
            one_type = one_type && (common_type.empty() || common_type == kind);
            common_type = std::move(kind);
            values.push_back(std::move(value));
        }
        nlohmann::json schema;
        // A type alongside the enum only when every value agrees on one. Mixed
        // literals still get an enum, which is the part that constrains.
        if (one_type) {
            schema["type"] = common_type;
        }
        schema["enum"] = std::move(values);
        return schema;
    }

    if (is_type(origin, &PyList_Type) || is_type(origin, &PyTuple_Type)) {
        nlohmann::json schema = {{"type", "array"}};
        if (py::len(members) == 1) {
            const py::object element = members[py::int_(0)];
            schema["items"] = schema_for_annotation(element);
        }
        return schema;
    }

    if (is_type(origin, &PyDict_Type)) {
        nlohmann::json schema = {{"type", "object"}};
        if (py::len(members) == 2) {
            const py::object value_type = members[py::int_(1)];
            schema["additionalProperties"] = schema_for_annotation(value_type);
        }
        return schema;
    }

    throw py::type_error("cannot describe the annotation " +
                         py::repr(annotation).cast<std::string>());
}

// The first paragraph of the docstring, which is the part a model should read.
// The rest is usually usage notes for a human, and spending prompt tokens on it
// every step of every run adds up.
[[nodiscard]] std::string summary_of(const py::handle& function) {
    const py::object doc = py::module_::import("inspect").attr("getdoc")(function);
    if (doc.is_none()) {
        return {};
    }
    std::string text = doc.cast<std::string>();
    const std::size_t blank = text.find("\n\n");
    if (blank != std::string::npos) {
        text.resize(blank);
    }
    return text;
}

[[nodiscard]] bool is_describable_name(const std::string& name) {
    // The shape the hosted endpoints accept. Enforced here because the failure
    // otherwise arrives as a 400 from the endpoint, several layers away from the
    // function that caused it.
    if (name.empty() || name.size() > 64) {
        return false;
    }
    const auto is_letter = [](char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
    };
    const auto is_rest = [&is_letter](char c) {
        return is_letter(c) || (c >= '0' && c <= '9') || c == '-';
    };
    return is_letter(name.front()) && std::all_of(name.begin() + 1, name.end(), is_rest);
}

[[nodiscard]] ToolSpec derive_spec(const py::object& function,
                                   const std::optional<std::string>& name_override,
                                   const std::optional<std::string>& description_override) {
    if (!PyCallable_Check(function.ptr())) {
        throw py::type_error("a tool must be a callable");
    }

    ToolSpec spec;
    if (name_override.has_value()) {
        spec.name = *name_override;
    } else if (py::hasattr(function, "__name__")) {
        spec.name = py::cast<std::string>(function.attr("__name__"));
    }
    if (!is_describable_name(spec.name)) {
        throw py::type_error(
            "'" + spec.name +
            "' is not a usable tool name: the model-facing name must start with a letter and "
            "contain only letters, digits, underscores and dashes, up to 64 characters. Pass "
            "name= to choose one.");
    }

    spec.description =
        description_override.has_value() ? *description_override : summary_of(function);

    const py::object inspect = py::module_::import("inspect");
    py::object signature;
    py::dict hints;
    try {
        signature = inspect.attr("signature")(function);
        hints = py::module_::import("typing").attr("get_type_hints")(function).cast<py::dict>();
    } catch (py::error_already_set& error) {
        // An unresolvable annotation raises NameError here -- at import time
        // that almost always means the module the tool is defined in is missing
        // an import, and the message is worth more than the traceback.
        throw py::type_error("tool '" + spec.name +
                             "': cannot read its signature: " + std::string{error.what()});
    }

    const py::object parameters = signature.attr("parameters");
    const py::object parameter = inspect.attr("Parameter");
    const py::object empty = parameter.attr("empty");

    nlohmann::json properties = nlohmann::json::object();
    nlohmann::json required = nlohmann::json::array();

    for (py::handle entry : parameters.attr("values")()) {
        const std::string parameter_name = py::cast<std::string>(entry.attr("name"));
        const py::object kind = entry.attr("kind");

        // A model fills in named arguments; there is no way to describe the
        // rest, so a function that needs them cannot be a tool.
        if (kind.is(parameter.attr("VAR_POSITIONAL")) ||
            kind.is(parameter.attr("VAR_KEYWORD"))) {
            throw py::type_error("tool '" + spec.name + "': '" + parameter_name +
                                 "' has no name to describe it by, so a model cannot fill it in");
        }

        const py::object fallback = entry.attr("default");
        if (fallback.is(empty)) {
            required.push_back(parameter_name);
        }

        // The annotation if there is one, otherwise the type of the default, so
        // that `def read(path, count=3)` describes count as an integer instead
        // of accepting the Python default and handing the model a string.
        // Nothing at all means a string, which is the useful reading of a bare
        // parameter and the one a model will get right.
        nlohmann::json schema;
        if (hints.contains(py::str(parameter_name))) {
            schema = schema_for_annotation(py::object{hints[py::str(parameter_name)]});
        } else if (!fallback.is(empty) && !fallback.is_none()) {
            const py::handle type_of_default{reinterpret_cast<PyObject*>(Py_TYPE(fallback.ptr()))};
            schema = schema_for_annotation(type_of_default);
        } else {
            schema = {{"type", "string"}};
        }
        properties[parameter_name] = std::move(schema);
    }

    spec.input_schema = {
        {"type", "object"},
        {"properties", std::move(properties)},
        {"required", std::move(required)},
        {"additionalProperties", false},
    };
    return spec;
}

// ---------------------------------------------------------------------------
// Calling one.
// ---------------------------------------------------------------------------

[[nodiscard]] std::string describe_python_error(const py::error_already_set& error) {
    // The type name and the message, and deliberately not error.what(): that is
    // pybind11's rendering for a human, which is the type and message followed
    // by the entire traceback. The traceback names files the model cannot open
    // and costs prompt tokens on every step of every run, and the type name is
    // already in hand -- so it would also be said twice.
    const std::string name = py::cast<std::string>(error.type().attr("__name__"));
    try {
        return name + ": " + py::cast<std::string>(py::str(error.value()));
    } catch (const py::error_already_set&) {
        // str() on the exception raised in turn, which takes a __str__ that
        // throws. The name is the part worth keeping.
        return name;
    }
}

[[nodiscard]] ToolResult result_from(const py::object& returned) {
    ToolResult result;
    if (returned.is_none()) {
        return result;
    }
    if (PyUnicode_Check(returned.ptr())) {
        result.content = returned.cast<std::string>();
        return result;
    }
    if (PyDict_Check(returned.ptr())) {
        const py::dict mapping = py::reinterpret_borrow<py::dict>(returned);
        if (mapping.contains("content")) {
            result.content = py::str(mapping["content"]).cast<std::string>();
            if (mapping.contains("is_error")) {
                result.is_error = py::cast<bool>(mapping["is_error"]);
            }
            return result;
        }
    }
    // Anything else is rendered as JSON, which is what the rest of the runtime
    // speaks. A tool returning a plain list or dict of data therefore works
    // without having to wrap it.
    result.content = json_from_py(returned).dump();
    return result;
}

// A free function and not a coroutine lambda, so that the callable is copied
// into the coroutine frame with the other parameters. A lambda coroutine reaches
// its captures through the closure, and the closure belongs to the caller: since
// Task begins suspended, this body first runs when the Task is resumed, by which
// time a lambda's closure may be gone. A parameter is copied into the frame and
// needs nothing to outlive it.
Task<ToolResult> invoke_python_tool(std::shared_ptr<PyCallable> callable,
                                    nlohmann::json arguments) {
    ToolResult result;
    {
        // Scoped, so the GIL is given back at the end of this block rather than
        // when the coroutine frame is destroyed. A guard left to the end of the
        // body would stay held until the caller dropped the Task -- and the
        // caller dropped it precisely because it had released the GIL.
        py::gil_scoped_acquire acquire;
        try {
            py::dict keyword_arguments;
            for (auto argument = arguments.begin(); argument != arguments.end(); ++argument) {
                keyword_arguments[py::str(argument.key())] = py_from_json(argument.value());
            }
            result = result_from(callable->get()(**keyword_arguments));
        } catch (py::error_already_set& error) {
            // Turned into a result here, under the GIL, for the same reason: an
            // error_already_set holds Python objects and its destructor needs
            // the GIL, so it must never be carried past the release.
            result.is_error = true;
            result.content = describe_python_error(error);
        } catch (const py::builtin_exception& error) {
            // Thrown by this binding's own converters -- a dict key that is not
            // a string, an integer too wide for json. The messages are written
            // to be read by a model, so they pass through unchanged.
            result.is_error = true;
            result.content = error.what();
        } catch (const std::exception& error) {
            result.is_error = true;
            result.content = std::string{"tool failed: "} + error.what();
        }
    }
    co_return result;
}

// A failed tool call is a result and not an exception. The loop hands the text
// back to the model as the tool's output, which is what gives the model a chance
// to fix its own arguments -- the same treatment an unknown tool name gets.
[[nodiscard]] std::shared_ptr<Tool> make_python_tool(const ToolSpec& spec, py::object function) {
    auto callable = std::make_shared<PyCallable>(std::move(function));

    return make_tool(spec.name, spec.description, spec.input_schema,
                     [callable](const nlohmann::json& arguments, std::stop_token) {
                         return invoke_python_tool(callable, arguments);
                     });
}

// Where the decorator leaves what it worked out. A function is the one object
// the decorator and ToolSet.add() both hold a reference to, so it is the only
// place a name= given to the one can be found by the other. Visible to
// ToolSet::add, which is defined below but outside this anonymous namespace.
constexpr const char* kSpecAttribute = "__ash_tool_spec__";

void remember_spec(const py::object& function, const ToolSpec& spec, bool required) {
    try {
        function.attr(kSpecAttribute) = to_python(spec);
    } catch (py::error_already_set&) {
        // A builtin, or an instance of a class with __slots__, refuses new
        // attributes.
        if (required) {
            throw py::type_error("cannot attach a tool specification to " +
                                 py::repr(function).cast<std::string>() +
                                 "; pass name= to ToolSet.add() as well");
        }
        // With nothing overridden there is nothing to remember that add() could
        // not work out again for itself, so refusing is fine.
    }
}

[[nodiscard]] py::object decorate(const py::object& function,
                                  const std::optional<std::string>& name,
                                  const std::optional<std::string>& description) {
    // Derived now rather than at add() time so that a signature which cannot be
    // described is an error where the function is defined, and kept so that
    // @tool(name=...) means the same thing as ToolSet.add(name=...).
    const ToolSpec spec = derive_spec(function, name, description);
    remember_spec(function, spec, name.has_value() || description.has_value());
    return function;
}

[[nodiscard]] py::object register_tool(const py::object& function,
                                       const std::optional<std::string>& name,
                                       const std::optional<std::string>& description) {
    if (function.is_none()) {
        // Called with arguments, as @tool(name="x"), so what is wanted is the
        // decorator that will be applied to the function next.
        return py::cpp_function([name, description](const py::object& decorated) {
            return decorate(decorated, name, description);
        });
    }
    return decorate(function, name, description);
}

}  // namespace

void ToolSet::add(const py::object& function, const std::optional<std::string>& name,
                  const std::optional<std::string>& description) {
    ToolSpec spec;
    if (!name.has_value() && !description.has_value() && py::hasattr(function, kSpecAttribute)) {
        // @tool already worked this out, possibly with a name this call does
        // not know about. Reading it back rather than deriving a second time
        // keeps the check the decorator did and the schema the model is given
        // one and the same.
        spec = json_from_py(py::getattr(function, kSpecAttribute)).get<ToolSpec>();
    } else {
        // No decorator, or an override here that outranks it. Derived now, so
        // that a signature which cannot be described fails when the tool is
        // added rather than the first time a model calls it.
        spec = derive_spec(function, name, description);
    }
    registry_.add(make_python_tool(spec, function));
}

std::vector<std::string> ToolSet::names() const {
    std::vector<std::string> names;
    names.reserve(registry_.size());
    for (const auto& tool : registry_.tools()) {
        names.emplace_back(tool->name());
    }
    return names;
}

py::list ToolSet::specs() const {
    py::list specs;
    for (const ToolSpec& spec : registry_.specs()) {
        specs.append(to_python(spec));
    }
    return specs;
}

void register_tools(py::module_& m) {
    py::class_<ToolSet>(m, "ToolSet", "The tools an Agent may call.")
        .def(py::init<>())
        .def(
            "add",
            [](ToolSet& tools, const py::object& function, const std::optional<std::string>& name,
               const std::optional<std::string>& description) {
                tools.add(function, name, description);
            },
            py::arg("function"), py::kw_only(), py::arg("name") = py::none(),
            py::arg("description") = py::none(),
            "Add a function. Its signature becomes the schema the model is given, and a signature "
            "that cannot be described raises rather than being guessed at.")
        .def_property_readonly("names", &ToolSet::names, "The tools' names, in the order added.")
        .def_property_readonly("specs", &ToolSet::specs, "What the model is told about each tool.")
        .def("__len__", [](const ToolSet& tools) { return tools.registry().size(); })
        .def("__repr__", [](const ToolSet& tools) {
            py::list names;
            for (const std::string& name : tools.names()) {
                names.append(py::str(name));
            }
            return "ToolSet(" + py::repr(names).cast<std::string>() + ")";
        });

    m.def("tool", &register_tool, py::arg("function") = py::none(), py::kw_only(),
          py::arg("name") = py::none(), py::arg("description") = py::none(),
          R"(Check that a function can be described to a model, and return it unchanged.

Used as a decorator. It does not wrap the function: the function stays a
function, and the schema is derived from its annotations and docstring when it
is added to a ToolSet. What the decorator buys is that the derivation happens
where the function is defined, so a parameter that cannot be described is an
error at import time rather than the first time a model calls the tool.

    @ash.tool
    def read_file(path: str) -> str:
        """Read a file and return its contents."""

    tools = ash.ToolSet()
    tools.add(read_file)

    @ash.tool(name="list_dir", description="List a directory.")
    def ls(path: str = "."):
        ...
)");
}

}  // namespace ash::python
