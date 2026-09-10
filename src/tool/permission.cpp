#include "ash/tool/permission.hpp"

#include <utility>

namespace ash {

PermissionTool::PermissionTool(std::shared_ptr<Tool> inner, ApprovalFn approve, std::string denial_message)
    : inner_(std::move(inner)), approve_(std::move(approve)), denial_message_(std::move(denial_message)) {}

std::string_view PermissionTool::name() const noexcept { return inner_->name(); }

std::string_view PermissionTool::description() const noexcept { return inner_->description(); }

const nlohmann::json& PermissionTool::input_schema() const noexcept { return inner_->input_schema(); }

Task<ToolResult> PermissionTool::invoke(const nlohmann::json& arguments, std::stop_token stop) const {
    if (!approve_(inner_->name(), arguments)) {
        co_return ToolResult{denial_message_, true};
    }

    ToolResult result = co_await inner_->invoke(arguments, stop);
    co_return result;
}

ToolRegistry make_permission_registry(const ToolRegistry& tools, ApprovalFn approve, std::string denial_message) {
    ToolRegistry gated;
    for (const auto& tool : tools.tools()) {
        gated.add(std::make_shared<PermissionTool>(tool, approve, denial_message));
    }
    return gated;
}

}  // namespace ash
