#include <catch2/catch_test_macros.hpp>

#include <string>
#include <string_view>
#include <vector>

#include "ash/model/sse.hpp"

namespace {

using ash::SseDecoder;
using ash::SseFrame;

// Feeds a whole stream one byte at a time and collects what came out. This is
// the shape the decoder actually faces: TCP hands up whatever arrived, and a
// boundary can land anywhere -- inside a field name, between the CR and the LF,
// or in the middle of the blank line that ends a frame. A splitter that works
// on whole lines looks correct until this test.
std::vector<SseFrame> feed_one_byte_at_a_time(std::string_view stream) {
    SseDecoder decoder;
    std::vector<SseFrame> frames;
    for (const char byte : stream) {
        for (SseFrame& frame : decoder.feed(std::string_view{&byte, 1})) {
            frames.push_back(std::move(frame));
        }
    }
    for (SseFrame& frame : decoder.finish()) {
        frames.push_back(std::move(frame));
    }
    return frames;
}

std::vector<SseFrame> feed_whole(std::string_view stream) {
    SseDecoder decoder;
    std::vector<SseFrame> frames = decoder.feed(stream);
    for (SseFrame& frame : decoder.finish()) {
        frames.push_back(std::move(frame));
    }
    return frames;
}

}  // namespace

TEST_CASE("a frame splits into its event name and data", "[sse]") {
    const auto frames = feed_whole("event: message\ndata: hello\n\n");

    REQUIRE(frames.size() == 1);
    CHECK(frames[0].event == "message");
    CHECK(frames[0].data == "hello");
}

TEST_CASE("a data-only frame has an empty event name", "[sse]") {
    const auto frames = feed_whole("data: hello\n\n");

    REQUIRE(frames.size() == 1);
    CHECK(frames[0].event.empty());
    CHECK(frames[0].data == "hello");
}

TEST_CASE("the same frames come out no matter where the reads split", "[sse]") {
    // The contract of the decoder stated as a test: the framing is a property
    // of the bytes, not of how they were delivered. If this passes and the
    // whole-buffer case passes, no chunking strategy can produce a wrong frame.
    const std::string stream =
        "event: message_start\n"
        "data: {\"type\":\"message_start\"}\n"
        "\n"
        ": keep-alive\n"
        "\n"
        "event: content_block_delta\n"
        "data: {\"delta\":\n"
        "data: \"hi\"}\n"
        "\n"
        "data: [DONE]\n"
        "\n";

    const auto whole = feed_whole(stream);
    const auto bytewise = feed_one_byte_at_a_time(stream);

    REQUIRE(whole.size() == 3);
    CHECK(whole == bytewise);
}

TEST_CASE("CRLF endings frame the same as LF", "[sse]") {
    const auto frames = feed_whole("event: message\r\ndata: hello\r\n\r\n");

    REQUIRE(frames.size() == 1);
    CHECK(frames[0].event == "message");
    CHECK(frames[0].data == "hello");
}

TEST_CASE("a CR and its LF split across reads is still one ending", "[sse]") {
    SseDecoder decoder;
    std::vector<SseFrame> frames = decoder.feed("data: hello\r");
    CHECK(frames.empty());

    // The CR was held as part of the line, and the LF that arrived next ends it.
    frames = decoder.feed("\n\r\n");
    REQUIRE(frames.size() == 1);
    CHECK(frames[0].data == "hello");
}

TEST_CASE("several data lines join with a newline", "[sse]") {
    const auto frames = feed_whole("data: one\ndata: two\ndata: three\n\n");

    REQUIRE(frames.size() == 1);
    CHECK(frames[0].data == "one\ntwo\nthree");
}

TEST_CASE("a colon inside the data is not a field separator", "[sse]") {
    const auto frames = feed_whole("data: {\"url\":\"http://example.com\"}\n\n");

    REQUIRE(frames.size() == 1);
    CHECK(frames[0].data == "{\"url\":\"http://example.com\"}");
}

TEST_CASE("only the first space after the colon is framing", "[sse]") {
    // `data:  x` is a value of " x". Trimming whitespace instead of stripping
    // exactly one space would silently eat a leading space out of model output,
    // which is a content bug that looks like a rendering bug.
    const auto frames = feed_whole("data:  x\n\n");

    REQUIRE(frames.size() == 1);
    CHECK(frames[0].data == " x");
}

TEST_CASE("a data line with no space after the colon is accepted", "[sse]") {
    const auto frames = feed_whole("data:tight\n\n");

    REQUIRE(frames.size() == 1);
    CHECK(frames[0].data == "tight");
}

TEST_CASE("comment lines are ignored", "[sse]") {
    const auto frames = feed_whole(": keep-alive\ndata: hello\n: another\n\n");

    REQUIRE(frames.size() == 1);
    CHECK(frames[0].data == "hello");
}

TEST_CASE("id and retry are read and dropped", "[sse]") {
    const auto frames = feed_whole("id: 42\nretry: 3000\ndata: hello\n\n");

    REQUIRE(frames.size() == 1);
    CHECK(frames[0].event.empty());
    CHECK(frames[0].data == "hello");
}

TEST_CASE("an event name does not survive into the next frame", "[sse]") {
    // A frame with an event name and no data dispatches nothing, and its name
    // must not be attached to the data-only frame that follows it.
    const auto frames = feed_whole("event: ping\n\ndata: hello\n\n");

    REQUIRE(frames.size() == 1);
    CHECK(frames[0].event.empty());
    CHECK(frames[0].data == "hello");
}

TEST_CASE("a blank line with no fields dispatches nothing", "[sse]") {
    const auto frames = feed_whole("\n\n\ndata: hello\n\n\n");

    REQUIRE(frames.size() == 1);
    CHECK(frames[0].data == "hello");
}

TEST_CASE("back-to-back frames in one read both come out", "[sse]") {
    const auto frames = feed_whole("data: one\n\ndata: two\n\n");

    REQUIRE(frames.size() == 2);
    CHECK(frames[0].data == "one");
    CHECK(frames[1].data == "two");
}

TEST_CASE("a frame that never got its blank line is delivered by finish", "[sse]") {
    // The last frame of a stream is usually the one that says the model is
    // done, and a connection that closes right after it would otherwise lose
    // exactly the frame that matters most.
    SseDecoder decoder;
    CHECK(decoder.feed("data: [DONE]\n").empty());

    const auto frames = decoder.finish();
    REQUIRE(frames.size() == 1);
    CHECK(frames[0].data == "[DONE]");
}

TEST_CASE("finish delivers a trailing line with no newline at all", "[sse]") {
    SseDecoder decoder;
    CHECK(decoder.feed("data: tail").empty());

    const auto frames = decoder.finish();
    REQUIRE(frames.size() == 1);
    CHECK(frames[0].data == "tail");
}

TEST_CASE("finish on a cleanly ended stream returns nothing", "[sse]") {
    SseDecoder decoder;
    REQUIRE(decoder.feed("data: hello\n\n").size() == 1);

    CHECK(decoder.finish().empty());
}

TEST_CASE("a half-written line that never completes is discarded", "[sse]") {
    // Nothing can be made of a lone field name, and a truncated stream is not
    // a reason to invent a frame.
    SseDecoder decoder;
    // The complete frame comes out, and the half-line that followed it stays
    // in the buffer, where it can still be completed by the next read.
    REQUIRE(decoder.feed("data: ok\n\ndat").size() == 1);

    // But a stream that ended there has nothing to make a frame out of, and
    // inventing one would put content in the journal that the model never sent.
    CHECK(decoder.finish().empty());
}

TEST_CASE("the decoder can be reused after finish", "[sse]") {
    SseDecoder decoder;
    REQUIRE(decoder.feed("data: one\n\n").size() == 1);
    CHECK(decoder.finish().empty());

    const auto frames = decoder.feed("data: two\n\n");
    REQUIRE(frames.size() == 1);
    CHECK(frames[0].data == "two");
}
