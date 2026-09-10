#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "ash/model/provider.hpp"
#include "ash/tool/tool.hpp"

namespace ash {

// The journal is an append-only log of everything an agent run did that it
// could not recompute on its own: model calls and tool calls. Replaying it
// re-runs the agent loop itself but answers every side effect from the log, so
// a run is reproducible without a network, a filesystem, or an API key.
//
// Interception happens at the provider and tool seams rather than at the HTTP
// byte layer, which keeps the format readable and lets a journal survive a
// change of transport.

inline constexpr int kJournalVersion = 1;

struct JournalHeader {
    int version = kJournalVersion;
    std::string provider;
    std::string model;
    std::string task;
    // The run configuration is stored rather than assumed, because a replay
    // rebuilds the same requests and any difference would show up as a
    // divergence. Note that the API key is not here and cannot be: it never
    // enters a request.
    std::string system_prompt;
    int max_steps = 0;
    std::string created_at;  // ISO-8601 UTC, metadata only -- never replayed
};

struct ModelCallRecord {
    ChatRequest request;
    ChatResponse response;
};

struct ToolCallRecord {
    std::string name;
    nlohmann::json arguments;
    ToolResult result;
};

// The header is journal metadata, written once as its own line; every other
// line is one of these.
using EventPayload = std::variant<ModelCallRecord, ToolCallRecord>;

// One recorded event. `seq` is monotonic within an `actor`, which is the id of
// the logical task that produced it. Replay only ever orders events within an
// actor, so the order in which concurrent tasks interleave is irrelevant and
// does not have to be reproduced.
struct Event {
    std::string actor;
    std::uint64_t seq = 0;
    EventPayload payload;
};

// Reads or writes a journal. The same type serves both directions because a
// recorded run and a replayed run must agree on exactly one representation.
class Journal {
public:
    // Creates the file, truncating anything already there, and prepares to
    // append events. The header is written lazily by `set_header`.
    static Journal create(const std::filesystem::path& path);

    // Loads every event. Events come back ordered by (actor, seq).
    static Journal load(const std::filesystem::path& path);

    Journal(Journal&& other) noexcept;
    Journal& operator=(Journal&& other) noexcept;
    Journal(const Journal&) = delete;
    Journal& operator=(const Journal&) = delete;
    ~Journal();

    // Appends an event for `actor`, assigning the next sequence number for
    // that actor, and returns it.
    std::uint64_t append(std::string_view actor, EventPayload payload);

    void set_header(JournalHeader header);
    [[nodiscard]] const JournalHeader& header() const noexcept { return header_; }

    [[nodiscard]] bool is_writable() const noexcept { return output_ != nullptr; }

    // All events in (actor, seq) order. Empty for a writable journal.
    [[nodiscard]] const std::vector<Event>& events() const noexcept { return events_; }

private:
    Journal() = default;

    std::uint64_t next_seq_for(std::string_view actor);

    std::mutex mutex_;
    JournalHeader header_;
    std::vector<Event> events_;
    std::vector<std::pair<std::string, std::uint64_t>> next_seq_;  // per actor
    std::unique_ptr<std::ofstream> output_;
    std::filesystem::path path_;
};

// The canonical hashes a replay uses to prove it is answering the same
// question it recorded. Both hash the JSON rendering, which is stable because
// nlohmann objects are key-sorted.
[[nodiscard]] std::string request_hash(const ChatRequest& request);
[[nodiscard]] std::string tool_call_hash(std::string_view name, const nlohmann::json& arguments);

}  // namespace ash
