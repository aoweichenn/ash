#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <string>
#include <string_view>
#include <thread>

#include <pthread.h>
#include <unistd.h>

#include "terminal.hpp"

namespace {

// A pipe standing in for stdin, so the reader is tested against the thing it
// actually does instead of against a terminal this suite does not have.
class Pipe {
public:
    Pipe() { REQUIRE(::pipe(fds_) == 0); }

    ~Pipe() {
        for (const int fd : fds_) {
            if (fd >= 0) {
                ::close(fd);
            }
        }
    }

    Pipe(const Pipe&) = delete;
    Pipe& operator=(const Pipe&) = delete;

    // No assertion in here: this is called from a writer thread too, and a
    // Catch2 assertion off the main thread is not something to reach for. The
    // flag is checked afterwards instead.
    void write(std::string_view text) {
        const ssize_t written = ::write(fds_[1], text.data(), text.size());
        wrote_all_ = wrote_all_ && written == static_cast<ssize_t>(text.size());
    }

    void close_write_end() {
        ::close(fds_[1]);
        fds_[1] = -1;
    }

    [[nodiscard]] bool wrote_all() const { return wrote_all_; }
    [[nodiscard]] int read_fd() const { return fds_[0]; }

private:
    int fds_[2];
    bool wrote_all_ = true;
};

using Result = ash::cli::LineReader::Result;

}  // namespace

TEST_CASE("a line comes back without its newline") {
    Pipe pipe;
    pipe.write("hello\n");

    ash::cli::LineReader reader{pipe.read_fd()};
    std::string line;

    CHECK(reader.read_line(line) == Result::kLine);
    CHECK(line == "hello");
}

TEST_CASE("lines that arrived in one read are handed over one at a time") {
    // A pipe can deliver several lines in a single read, a terminal never does,
    // and the caller must not have to know which one it is talking to.
    Pipe pipe;
    pipe.write("one\ntwo\nthree\n");

    ash::cli::LineReader reader{pipe.read_fd()};
    std::string line;

    for (const char* expected : {"one", "two", "three"}) {
        CHECK(reader.read_line(line) == Result::kLine);
        CHECK(line == expected);
    }
}

TEST_CASE("a line split across two reads is put back together") {
    Pipe pipe;
    ash::cli::LineReader reader{pipe.read_fd()};
    std::string line;

    // The sleep is what makes this a test of the buffering: the first read can
    // only see "he" because "llo\n" has not been written yet, so the reader has
    // to hold onto it and carry on.
    std::jthread writer{[&pipe] {
        pipe.write("he");
        std::this_thread::sleep_for(std::chrono::milliseconds{50});
        pipe.write("llo\n");
        pipe.close_write_end();
    }};

    CHECK(reader.read_line(line) == Result::kLine);
    CHECK(line == "hello");
    CHECK(reader.read_line(line) == Result::kEof);
    CHECK(pipe.wrote_all());
}

TEST_CASE("the last line of the input still counts when it has no newline") {
    Pipe pipe;
    pipe.write("no newline here");
    pipe.close_write_end();

    ash::cli::LineReader reader{pipe.read_fd()};
    std::string line;

    CHECK(reader.read_line(line) == Result::kLine);
    CHECK(line == "no newline here");
    // And the end is reported after it, not instead of it.
    CHECK(reader.read_line(line) == Result::kEof);
}

TEST_CASE("the end of the input keeps being the end") {
    Pipe pipe;
    pipe.close_write_end();

    ash::cli::LineReader reader{pipe.read_fd()};
    std::string line;

    CHECK(reader.read_line(line) == Result::kEof);
    CHECK(reader.read_line(line) == Result::kEof);
}

TEST_CASE("empty input is the end, not an empty line") {
    // A blank line and no line at all are different things, and a session that
    // confused them would loop forever on a closed stdin.
    Pipe pipe;
    pipe.close_write_end();

    ash::cli::LineReader reader{pipe.read_fd()};
    std::string line = "not touched by anything";

    CHECK(reader.read_line(line) == Result::kEof);
}

TEST_CASE("a blank line is a line") {
    Pipe pipe;
    pipe.write("\n\n");
    pipe.close_write_end();

    ash::cli::LineReader reader{pipe.read_fd()};
    std::string line;

    CHECK(reader.read_line(line) == Result::kLine);
    CHECK(line.empty());
    CHECK(reader.read_line(line) == Result::kLine);
    CHECK(line.empty());
    CHECK(reader.read_line(line) == Result::kEof);
}

TEST_CASE("a trailing carriage return is dropped") {
    Pipe pipe;
    pipe.write("windows\r\nunix\n");
    pipe.close_write_end();

    ash::cli::LineReader reader{pipe.read_fd()};
    std::string line;

    CHECK(reader.read_line(line) == Result::kLine);
    CHECK(line == "windows");
    CHECK(reader.read_line(line) == Result::kLine);
    CHECK(line == "unix");
}

TEST_CASE("the line a previous call returned is overwritten, not appended to") {
    Pipe pipe;
    pipe.write("first\nsecond\n");
    pipe.close_write_end();

    ash::cli::LineReader reader{pipe.read_fd()};
    std::string line = "left over from somewhere";

    CHECK(reader.read_line(line) == Result::kLine);
    CHECK(line == "first");
    CHECK(reader.read_line(line) == Result::kLine);
    CHECK(line == "second");
}

TEST_CASE("a terminal is told apart from a pipe") {
    Pipe pipe;
    CHECK_FALSE(ash::cli::is_a_terminal(pipe.read_fd()));
    CHECK_FALSE(ash::cli::is_a_terminal(-1));
}

namespace {

extern "C" void ignore_interrupt(int) {}

// A handler that does nothing and does not restart the read, which is what the
// session installs: without it there is no EINTR to see, and with SA_RESTART
// there is no interruption either.
class InterruptHandler {
public:
    InterruptHandler() {
        struct sigaction action {};
        action.sa_handler = ignore_interrupt;
        sigemptyset(&action.sa_mask);
        action.sa_flags = 0;
        REQUIRE(::sigaction(SIGINT, &action, &previous_) == 0);
    }

    ~InterruptHandler() { ::sigaction(SIGINT, &previous_, nullptr); }

    InterruptHandler(const InterruptHandler&) = delete;
    InterruptHandler& operator=(const InterruptHandler&) = delete;

private:
    struct sigaction previous_ {};
};

}  // namespace

TEST_CASE("a read that is interrupted says so instead of reporting a line") {
    // This is the whole reason the class exists, so it is tested with a real
    // signal rather than left to the one end-to-end check that also needs a
    // terminal, a model, and a user.
    InterruptHandler handler;
    Pipe pipe;

    ash::cli::LineReader reader{pipe.read_fd()};
    std::string line = "left as it was";

    const pthread_t reading_thread = ::pthread_self();
    std::atomic<bool> finished{false};
    std::jthread interrupter{[reading_thread, &finished] {
        // Repeated until the read comes back, so this cannot hang waiting for a
        // signal that arrived a moment too early to interrupt anything.
        for (int attempt = 0; attempt < 400 && !finished.load(); ++attempt) {
            ::pthread_kill(reading_thread, SIGINT);
            std::this_thread::sleep_for(std::chrono::milliseconds{5});
        }
    }};

    CHECK(reader.read_line(line) == Result::kInterrupted);
    finished.store(true);

    // Nothing was typed, so there is no line, and the caller's string is not
    // quietly emptied on the way past.
    CHECK(line == "left as it was");
}

TEST_CASE("an interruption drops the half-typed line rather than carrying it forward") {
    InterruptHandler handler;
    Pipe pipe;

    ash::cli::LineReader reader{pipe.read_fd()};
    std::string line;

    // No newline, so the reader is left holding half a line when the signal
    // arrives. Ctrl-C means "forget what I was saying"; letting it survive
    // would put it in front of whatever the user typed next.
    pipe.write("half typed");

    const pthread_t reading_thread = ::pthread_self();
    std::atomic<bool> finished{false};
    std::jthread interrupter{[reading_thread, &finished] {
        for (int attempt = 0; attempt < 400 && !finished.load(); ++attempt) {
            ::pthread_kill(reading_thread, SIGINT);
            std::this_thread::sleep_for(std::chrono::milliseconds{5});
        }
    }};

    // The first call has to get far enough to buffer the text before the signal
    // lands, and the interrupter keeps signalling until it does.
    CHECK(reader.read_line(line) == Result::kInterrupted);
    finished.store(true);

    pipe.write("next line\n");
    pipe.close_write_end();

    CHECK(reader.read_line(line) == Result::kLine);
    CHECK(line == "next line");
    CHECK(pipe.wrote_all());
}
