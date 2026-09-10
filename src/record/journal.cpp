#include "ash/record/journal.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>

#include <openssl/evp.h>

namespace ash {

namespace {

std::string iso8601_utc_now() {
    const std::time_t seconds = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm utc{};
    gmtime_r(&seconds, &utc);
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc);
    return std::string{buffer};
}

std::string sha256_hex(std::string_view data) {
    using Context = std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>;
    Context context{EVP_MD_CTX_new(), &EVP_MD_CTX_free};
    if (context == nullptr) {
        throw std::runtime_error{"failed to allocate a digest context"};
    }
    if (EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1 ||
        EVP_DigestUpdate(context.get(), data.data(), data.size()) != 1) {
        throw std::runtime_error{"failed to hash a journal record"};
    }

    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int length = 0;
    if (EVP_DigestFinal_ex(context.get(), digest, &length) != 1) {
        throw std::runtime_error{"failed to finalize a journal digest"};
    }

    static constexpr char kHexDigits[] = "0123456789abcdef";
    std::string hex;
    hex.reserve(length * 2);
    for (unsigned int i = 0; i < length; ++i) {
        hex.push_back(kHexDigits[digest[i] >> 4]);
        hex.push_back(kHexDigits[digest[i] & 0x0F]);
    }
    return hex;
}

nlohmann::json encode_header(const JournalHeader& header) {
    return nlohmann::json{{"kind", "header"},
                          {"version", header.version},
                          {"provider", header.provider},
                          {"model", header.model},
                          {"task", header.task},
                          {"system_prompt", header.system_prompt},
                          {"max_steps", header.max_steps},
                          {"created_at", header.created_at}};
}

JournalHeader decode_header(const nlohmann::json& line) {
    JournalHeader header;
    header.version = line.value("version", 0);
    if (header.version != kJournalVersion) {
        throw std::runtime_error{"unsupported journal version " + std::to_string(header.version) +
                                 ", expected " + std::to_string(kJournalVersion)};
    }
    header.provider = line.value("provider", "");
    header.model = line.value("model", "");
    header.task = line.value("task", "");
    header.system_prompt = line.value("system_prompt", "");
    header.max_steps = line.value("max_steps", 0);
    header.created_at = line.value("created_at", "");
    return header;
}

nlohmann::json encode_event(const Event& event) {
    nlohmann::json line;
    line["actor"] = event.actor;
    line["seq"] = event.seq;

    if (const auto* model_call = std::get_if<ModelCallRecord>(&event.payload)) {
        line["kind"] = "model_call";
        line["request"] = model_call->request;
        line["response"] = model_call->response;
    } else if (const auto* tool_call = std::get_if<ToolCallRecord>(&event.payload)) {
        line["kind"] = "tool_call";
        line["name"] = tool_call->name;
        line["arguments"] = tool_call->arguments;
        line["result"] = nlohmann::json{{"content", tool_call->result.content},
                                        {"is_error", tool_call->result.is_error}};
    }
    return line;
}

Event decode_event(const nlohmann::json& line) {
    Event event;
    event.actor = line.at("actor").get<std::string>();
    event.seq = line.at("seq").get<std::uint64_t>();

    const std::string kind = line.at("kind").get<std::string>();
    if (kind == "model_call") {
        ModelCallRecord call;
        call.request = line.at("request").get<ChatRequest>();
        call.response = line.at("response").get<ChatResponse>();
        event.payload = std::move(call);
    } else if (kind == "tool_call") {
        ToolCallRecord call;
        call.name = line.at("name").get<std::string>();
        call.arguments = line.at("arguments");
        call.result.content = line.at("result").value("content", "");
        call.result.is_error = line.at("result").value("is_error", false);
        event.payload = std::move(call);
    } else {
        throw std::runtime_error{"unknown journal event kind '" + kind + "'"};
    }
    return event;
}

}  // namespace

Journal Journal::create(const std::filesystem::path& path) {
    Journal journal;
    journal.path_ = path;

    if (path.has_parent_path() && !path.parent_path().empty()) {
        std::error_code error;
        std::filesystem::create_directories(path.parent_path(), error);
        if (error) {
            throw std::runtime_error{"cannot create " + path.parent_path().string() + ": " + error.message()};
        }
    }

    journal.output_ = std::make_unique<std::ofstream>(path, std::ios::binary | std::ios::trunc);
    if (!*journal.output_) {
        throw std::runtime_error{"cannot open journal for writing: " + path.string()};
    }
    return journal;
}

Journal Journal::load(const std::filesystem::path& path) {
    std::ifstream input{path, std::ios::binary};
    if (!input) {
        throw std::runtime_error{"cannot open journal: " + path.string()};
    }

    Journal journal;
    journal.path_ = path;

    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty()) {
            continue;
        }

        const nlohmann::json parsed = nlohmann::json::parse(line, nullptr, false);
        if (parsed.is_discarded()) {
            throw std::runtime_error{"journal " + path.string() + " line " + std::to_string(line_number) +
                                     " is not valid JSON"};
        }

        if (parsed.value("kind", "") == "header") {
            journal.header_ = decode_header(parsed);
            continue;
        }
        journal.events_.push_back(decode_event(parsed));
    }

    // Only the order within an actor carries meaning: the interleaving of
    // concurrent tasks is an artefact of scheduling, not of the run.
    std::stable_sort(journal.events_.begin(), journal.events_.end(), [](const Event& lhs, const Event& rhs) {
        return lhs.actor == rhs.actor ? lhs.seq < rhs.seq : lhs.actor < rhs.actor;
    });
    return journal;
}

Journal::Journal(Journal&& other) noexcept
    : header_(std::move(other.header_)),
      events_(std::move(other.events_)),
      next_seq_(std::move(other.next_seq_)),
      output_(std::move(other.output_)),
      path_(std::move(other.path_)) {}

Journal& Journal::operator=(Journal&& other) noexcept {
    if (this != &other) {
        const std::scoped_lock lock{mutex_, other.mutex_};
        header_ = std::move(other.header_);
        events_ = std::move(other.events_);
        next_seq_ = std::move(other.next_seq_);
        output_ = std::move(other.output_);
        path_ = std::move(other.path_);
    }
    return *this;
}

Journal::~Journal() = default;

std::uint64_t Journal::next_seq_for(std::string_view actor) {
    for (auto& [name, seq] : next_seq_) {
        if (name == actor) {
            return seq++;
        }
    }
    next_seq_.emplace_back(std::string{actor}, 1);
    return 0;
}

std::uint64_t Journal::append(std::string_view actor, EventPayload payload) {
    const std::scoped_lock lock{mutex_};
    if (output_ == nullptr) {
        throw std::logic_error{"journal is not writable"};
    }

    Event event;
    event.actor = std::string{actor};
    event.seq = next_seq_for(actor);
    event.payload = std::move(payload);

    *output_ << encode_event(event).dump() << '\n';
    output_->flush();
    return event.seq;
}

void Journal::set_header(JournalHeader header) {
    const std::scoped_lock lock{mutex_};
    if (output_ == nullptr) {
        throw std::logic_error{"journal is not writable"};
    }
    if (header.created_at.empty()) {
        header.created_at = iso8601_utc_now();
    }
    header_ = header;

    *output_ << encode_header(header).dump() << '\n';
    output_->flush();
}

std::string request_hash(const ChatRequest& request) {
    return sha256_hex(nlohmann::json{request}.dump());
}

std::string tool_call_hash(std::string_view name, const nlohmann::json& arguments) {
    const nlohmann::json payload{{"name", std::string{name}}, {"arguments", arguments}};
    return sha256_hex(payload.dump());
}

}  // namespace ash
