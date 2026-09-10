#include "ash/model/sse.hpp"

#include <cstddef>
#include <utility>

namespace ash {

namespace {

constexpr char kColon = ':';
constexpr char kSpace = ' ';
constexpr char kNewline = '\n';
constexpr char kCarriageReturn = '\r';

// The stream is text framed by lines, so the ending is part of the framing and
// not part of the value. Both endings appear in the wild: the specification
// allows either, and proxies rewrite one into the other.
[[nodiscard]] std::string_view strip_carriage_return(std::string_view line) noexcept {
    if (!line.empty() && line.back() == kCarriageReturn) {
        line.remove_suffix(1);
    }
    return line;
}

}  // namespace

void SseDecoder::consume_line(std::string_view line, std::vector<SseFrame>& out) {
    if (line.empty()) {
        // The blank line is the only thing that dispatches a frame. Without it
        // the fields so far are still accumulating.
        if (has_data_) {
            out.push_back(SseFrame{std::move(event_), std::move(data_)});
        }
        // The event name is reset even when nothing was dispatched. A frame
        // carrying only an `event:` line produces no frame at all, and if its
        // name survived it would be attached to whatever came next -- a decoder
        // reading `event: ping` followed by a data-only frame would file that
        // frame under the wrong type.
        event_.clear();
        data_.clear();
        has_data_ = false;
        return;
    }

    if (line.front() == kColon) {
        // A comment, which is what a proxy's keep-alive line is. It carries no
        // fields and must not disturb the frame being accumulated.
        return;
    }

    const std::size_t colon = line.find(kColon);
    const std::string_view field = colon == std::string_view::npos ? line : line.substr(0, colon);
    std::string_view value = colon == std::string_view::npos ? std::string_view{} : line.substr(colon + 1);

    // Exactly one leading space belongs to the framing. It is stripped here
    // rather than by a trimmed comparison, so that a data line whose value
    // really does start with a space keeps the second one.
    if (!value.empty() && value.front() == kSpace) {
        value.remove_prefix(1);
    }

    if (field == "event") {
        event_.assign(value);
    } else if (field == "data") {
        if (has_data_) {
            data_ += kNewline;
        }
        data_.append(value);
        has_data_ = true;
    }
    // `id` and `retry` are part of the protocol and mean nothing to a model
    // stream, so they are read and dropped rather than special-cased.
}

std::vector<SseFrame> SseDecoder::feed(std::string_view bytes) {
    std::vector<SseFrame> frames;
    buffer_.append(bytes);

    std::size_t start = 0;
    for (;;) {
        const std::size_t end = buffer_.find(kNewline, start);
        if (end == std::string::npos) {
            break;
        }
        consume_line(strip_carriage_return(std::string_view{buffer_}.substr(start, end - start)), frames);
        start = end + 1;
    }

    // Whatever is left is a line that has not ended. It stays until the next
    // call or until finish(), because it may be half a field name.
    buffer_.erase(0, start);
    return frames;
}

std::vector<SseFrame> SseDecoder::finish() {
    std::vector<SseFrame> frames;

    if (!buffer_.empty()) {
        const std::string_view line = strip_carriage_return(buffer_);
        consume_line(line, frames);
        buffer_.clear();
    }

    // A stream that stopped mid-frame still produced whatever fields it managed
    // to send, and a truncated answer is more useful than no answer.
    if (has_data_) {
        frames.push_back(SseFrame{std::move(event_), std::move(data_)});
        event_.clear();
        data_.clear();
        has_data_ = false;
    }
    return frames;
}

}  // namespace ash
