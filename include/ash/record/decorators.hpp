#pragma once

#include <memory>
#include <stop_token>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

#include "ash/model/provider.hpp"
#include "ash/model/stream.hpp"
#include "ash/record/journal.hpp"
#include "ash/record/replay.hpp"
#include "ash/tool/tool.hpp"

namespace ash {

// Recording and replaying are decorators rather than branches inside the agent
// loop, so the loop itself has no idea which mode it is running in. That is what
// keeps the recorded run and the replayed run on the same code path.

class RecordingProvider final : public ModelProvider {
public:
    // Shared and not unique, matching RecordingTool below. A caller that has to
    // keep its own reference to the provider -- which is what a language binding
    // does, since it is the garbage collector that decides when a Python object
    // goes -- would otherwise have to give up the only owner it had.
    RecordingProvider(std::shared_ptr<ModelProvider> inner, Journal& journal, std::string actor);

    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] const std::string& model() const noexcept override;
    Task<ChatResponse> chat(ChatRequest request) override;

    // Overridden so that recording a run does not quietly cost the run its live
    // output: the events are forwarded as they arrive, and the assembled call is
    // written afterwards, exactly as chat() would have written it.
    Task<ChatResponse> chat_stream(ChatRequest request, StreamSink& sink, std::stop_token stop = {}) override;

private:
    std::shared_ptr<ModelProvider> inner_;
    Journal& journal_;
    std::string actor_;
};

// Serves answers from a journal and never touches the network.
class ReplayingProvider final : public ModelProvider {
public:
    ReplayingProvider(ReplayCursor& cursor, std::string name, std::string model);

    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] const std::string& model() const noexcept override;
    Task<ChatResponse> chat(ChatRequest request) override;

private:
    ReplayCursor& cursor_;
    std::string name_;
    std::string model_;
};

class RecordingTool final : public Tool {
public:
    RecordingTool(std::shared_ptr<Tool> inner, Journal& journal, std::string actor);

    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] std::string_view description() const noexcept override;
    [[nodiscard]] const nlohmann::json& input_schema() const noexcept override;
    Task<ToolResult> invoke(const nlohmann::json& arguments, std::stop_token stop) const override;

private:
    std::shared_ptr<Tool> inner_;
    Journal& journal_;
    std::string actor_;
};

// Answers from the journal instead of performing any side effect, which is why a
// replay can safely rerun a run that wrote files.
class ReplayingTool final : public Tool {
public:
    ReplayingTool(ReplayCursor& cursor, ToolSpec spec);

    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] std::string_view description() const noexcept override;
    [[nodiscard]] const nlohmann::json& input_schema() const noexcept override;
    Task<ToolResult> invoke(const nlohmann::json& arguments, std::stop_token stop) const override;

private:
    ReplayCursor& cursor_;
    ToolSpec spec_;
};

// Wraps every tool in `tools` so invocations are appended to the journal.
[[nodiscard]] ToolRegistry make_recording_registry(const ToolRegistry& tools,
                                                   Journal& journal,
                                                   std::string actor);

// Rebuilds the tool set from the recording, so a replay needs nothing but the
// journal file. The tools perform no work: they answer from the cursor.
[[nodiscard]] ToolRegistry make_replaying_registry(const Journal& journal,
                                                   ReplayCursor& cursor,
                                                   std::string_view actor);

}  // namespace ash
