#pragma once

// The tool collection, in its own header so that run.cpp can build an Agent on
// top of it. The schema derivation and the trampoline stay in tools.cpp.

#include <optional>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>

#include "ash/tool/tool.hpp"

namespace ash::python {

// The tools an Agent may call.
//
// It owns a ToolRegistry and nothing else. A tool is immutable and holds no
// invocation state, so copying the registry into an Agent -- which is what
// happens on construction -- shares the tools rather than duplicating them, and
// the same ToolSet can back several Agents at once.
class ToolSet {
public:
    ToolSet() = default;

    void add(const pybind11::object& function,
             const std::optional<std::string>& name,
             const std::optional<std::string>& description);

    [[nodiscard]] std::vector<std::string> names() const;
    [[nodiscard]] pybind11::list specs() const;

    [[nodiscard]] const ToolRegistry& registry() const noexcept { return registry_; }

private:
    ToolRegistry registry_;
};

}  // namespace ash::python
