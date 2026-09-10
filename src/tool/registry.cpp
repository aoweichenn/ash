#include "ash/tool/tool.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace ash {

namespace {

class FunctionTool final : public Tool {
public:
    FunctionTool(std::string name, std::string description, nlohmann::json input_schema, ToolFunction function)
        : name_(std::move(name)),
          description_(std::move(description)),
          input_schema_(std::move(input_schema)),
          function_(std::move(function)) {}

    [[nodiscard]] std::string_view name() const noexcept override { return name_; }

    [[nodiscard]] std::string_view description() const noexcept override { return description_; }

    [[nodiscard]] const nlohmann::json& input_schema() const noexcept override { return input_schema_; }

    Task<ToolResult> invoke(const nlohmann::json& arguments, std::stop_token stop) const override {
        return function_(arguments, std::move(stop));
    }

private:
    std::string name_;
    std::string description_;
    nlohmann::json input_schema_;
    ToolFunction function_;
};

}  // namespace

ToolSpec Tool::spec() const {
    ToolSpec spec;
    spec.name = std::string{name()};
    spec.description = std::string{description()};
    spec.input_schema = input_schema();
    return spec;
}

std::shared_ptr<Tool> make_tool(std::string name,
                                std::string description,
                                nlohmann::json input_schema,
                                ToolFunction function) {
    return std::make_shared<FunctionTool>(
        std::move(name), std::move(description), std::move(input_schema), std::move(function));
}

void ToolRegistry::add(std::shared_ptr<Tool> tool) {
    if (tool == nullptr) {
        throw std::invalid_argument{"cannot register a null tool"};
    }
    if (find(tool->name()) != nullptr) {
        throw std::invalid_argument{"duplicate tool name: " + std::string{tool->name()}};
    }
    tools_.push_back(std::move(tool));
}

const Tool* ToolRegistry::find(std::string_view name) const noexcept {
    const auto it = std::find_if(tools_.begin(), tools_.end(), [name](const std::shared_ptr<Tool>& tool) {
        return tool->name() == name;
    });
    return it == tools_.end() ? nullptr : it->get();
}

std::vector<ToolSpec> ToolRegistry::specs() const {
    std::vector<ToolSpec> specs;
    specs.reserve(tools_.size());
    for (const auto& tool : tools_) {
        specs.push_back(tool->spec());
    }
    return specs;
}

}  // namespace ash
