#pragma once

#include <functional>
#include <memory>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "ash/model/provider.hpp"
#include "ash/task.hpp"

namespace ash {

struct ToolResult {
    std::string content;
    bool is_error = false;

    friend bool operator==(const ToolResult&, const ToolResult&) = default;
};

class Tool {
public:
    virtual ~Tool() = default;

    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual std::string_view description() const noexcept = 0;
    [[nodiscard]] virtual const nlohmann::json& input_schema() const noexcept = 0;

    // A tool is immutable: all state belongs to the invocation, not the object.
    // That is what lets one registry be shared across threads without locking.
    virtual Task<ToolResult> invoke(const nlohmann::json& arguments, std::stop_token stop) const = 0;

    [[nodiscard]] ToolSpec spec() const;
};

using ToolFunction = std::function<Task<ToolResult>(const nlohmann::json&, std::stop_token)>;

[[nodiscard]] std::shared_ptr<Tool> make_tool(std::string name,
                                              std::string description,
                                              nlohmann::json input_schema,
                                              ToolFunction function);

// Owns the tools an agent is allowed to call.
class ToolRegistry {
public:
    void add(std::shared_ptr<Tool> tool);

    [[nodiscard]] const Tool* find(std::string_view name) const noexcept;

    [[nodiscard]] std::vector<ToolSpec> specs() const;

    [[nodiscard]] const std::vector<std::shared_ptr<Tool>>& tools() const noexcept { return tools_; }

    [[nodiscard]] std::size_t size() const noexcept { return tools_.size(); }

    [[nodiscard]] bool empty() const noexcept { return tools_.empty(); }

private:
    std::vector<std::shared_ptr<Tool>> tools_;
};

}  // namespace ash
