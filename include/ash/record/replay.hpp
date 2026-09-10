#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "ash/model/provider.hpp"
#include "ash/record/journal.hpp"
#include "ash/tool/tool.hpp"

namespace ash {

// Raised when a replayed run asks something the journal does not answer, which
// means the run has drifted from the recording.
class ReplayError : public std::runtime_error {
public:
    explicit ReplayError(const std::string& message) : std::runtime_error(message) {}
};

// Hands out one actor's recorded answers in order, checking that each question
// matches the one that was recorded. The cursor borrows the journal, which must
// outlive it.
class ReplayCursor {
public:
    ReplayCursor(const Journal& journal, std::string actor);

    [[nodiscard]] const ChatResponse& next_model_call(const ChatRequest& request);
    [[nodiscard]] const ToolResult& next_tool_call(std::string_view name, const nlohmann::json& arguments);

    // Fails if the run finished without consuming everything it recorded, which
    // means it took a different path than the original.
    void verify_consumed() const;

    [[nodiscard]] std::size_t consumed() const noexcept { return consumed_; }
    [[nodiscard]] std::size_t available() const noexcept { return events_.size(); }
    [[nodiscard]] const std::string& actor() const noexcept { return actor_; }

private:
    [[nodiscard]] const Event& next_event(std::string_view expected_kind) const;

    std::vector<const Event*> events_;
    std::string actor_;
    std::size_t consumed_ = 0;
};

// The tool declarations the model was shown, recovered from the recording. This
// is what lets a replay rebuild the tool set from the journal alone instead of
// asking the caller to register the same tools a second time.
[[nodiscard]] std::vector<ToolSpec> recorded_tool_specs(const Journal& journal, std::string_view actor);

}  // namespace ash
