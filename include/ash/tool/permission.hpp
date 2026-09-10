#pragma once

#include <functional>
#include <memory>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

#include "ash/tool/tool.hpp"

namespace ash {

// Asks a question before a tool runs, and answers it elsewhere.
//
// The callback is the whole of the policy: this file has no idea what should be
// confirmed, only that someone has to be. That is deliberate. Which calls need
// consent is a decision about a person's project and belongs with whoever is
// holding the terminal, while the mechanism -- ask, and turn a "no" into a
// result the model can read -- is the same everywhere and is worth testing
// once.
//
// `approve` is called from inside the agent loop, so it must not throw: a throw
// from here unwinds past the loop and takes the run with it. It answers with a
// bool for the same reason a tool answers with an error result.
using ApprovalFn = std::function<bool(std::string_view tool, const nlohmann::json& arguments)>;

class PermissionTool final : public Tool {
public:
    PermissionTool(std::shared_ptr<Tool> inner, ApprovalFn approve, std::string denial_message);

    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] std::string_view description() const noexcept override;
    [[nodiscard]] const nlohmann::json& input_schema() const noexcept override;
    Task<ToolResult> invoke(const nlohmann::json& arguments, std::stop_token stop) const override;

private:
    std::shared_ptr<Tool> inner_;
    ApprovalFn approve_;
    std::string denial_message_;
};

// Wraps every tool in `tools` so each call is put to `approve` first.
//
// A refusal is returned as `is_error`, not raised: the model is told the call
// did not happen and can decide what to do about it, which is the same shape an
// unknown tool already produces. The alternative -- failing the run -- would
// make one "no" end a session the user was in the middle of.
//
// The tools themselves stay immutable and stateless, so whatever remembers "yes
// for this one, for the rest of the session" belongs to the caller and lives in
// the callback. That is the property that lets one registry be shared across
// threads without locking, and it is not worth spending to save a capture.
[[nodiscard]] ToolRegistry make_permission_registry(const ToolRegistry& tools,
                                                    ApprovalFn approve,
                                                    std::string denial_message = "the user denied this "
                                                                                 "tool call");

}  // namespace ash
