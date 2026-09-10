#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace ash {

// One Server-Sent Events frame: the name from an `event:` line, empty when the
// stream did not name one, and every `data:` line joined back together.
struct SseFrame {
    std::string event;
    std::string data;

    friend bool operator==(const SseFrame&, const SseFrame&) = default;
};

// Turns a byte stream into frames.
//
// The only interesting part is that the boundaries are arbitrary: a read can
// land inside a line, inside a field name, or between a CR and its LF, and none
// of those may be mistaken for a frame. So feed() keeps whatever it cannot yet
// dispatch and returns only the frames it has seen terminated. Nothing is
// assumed about how the caller's chunks line up with the frames, which is what
// makes this testable without a socket -- and the test that matters feeds one
// byte at a time, because that is the case a naive splitter gets wrong.
class SseDecoder {
public:
    // Feeds whatever arrived and returns every frame those bytes completed.
    [[nodiscard]] std::vector<SseFrame> feed(std::string_view bytes);

    // A stream that ends without a blank line still has a frame in hand, and the
    // last frame is usually the one that says the model is done.
    [[nodiscard]] std::vector<SseFrame> finish();

private:
    void consume_line(std::string_view line, std::vector<SseFrame>& out);

    std::string buffer_;  // the bytes of a line that has not ended yet
    std::string event_;
    std::string data_;
    bool has_data_ = false;
};

}  // namespace ash
