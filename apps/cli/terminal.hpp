#pragma once

// Reading a line, without stdio.
//
// std::getline is not used and std::cin is not touched, for a reason that only
// shows up when someone presses Ctrl-C. Ctrl-C interrupts the read with EINTR,
// and stdio does one of two things with that, both wrong here: it sets failbit
// on the stream, after which every later getline fails and the session is
// silently dead; or it restarts the read internally, in which case the
// interruption is never seen at all. The session needs neither -- Ctrl-C at an
// empty prompt should end it, and the only way to notice is to see the EINTR.
//
// So this reads the descriptor itself. It buffers, because a pipe hands over
// several lines in one read where a terminal hands over one, and a caller that
// had to know which it was talking to would be a caller that gets it wrong.

#include <cerrno>
#include <cstddef>
#include <string>
#include <sys/types.h>
#include <unistd.h>

namespace ash::cli {

[[nodiscard]] inline bool is_a_terminal(int fd) { return ::isatty(fd) == 1; }

class LineReader {
public:
    enum class Result { kLine, kEof, kInterrupted };

    explicit LineReader(int fd = STDIN_FILENO) : fd_(fd) {}

    // Reads one line into `line`, which is overwritten when there is a line to
    // report and left alone when there is not.
    //
    // A line is whatever precedes a newline, with a trailing carriage return
    // dropped so a CRLF terminal does not put a stray \r in every command. The
    // last line of a file that does not end in a newline is still a line: EOF
    // with something buffered hands that over first and reports the end on the
    // call after.
    Result read_line(std::string& line) {
        for (;;) {
            const std::size_t newline = buffer_.find('\n');
            if (newline != std::string::npos) {
                line.assign(buffer_, 0, newline);
                buffer_.erase(0, newline + 1);
                strip_carriage_return(line);
                return Result::kLine;
            }

            if (eof_) {
                if (buffer_.empty()) {
                    return Result::kEof;
                }
                line = std::move(buffer_);
                buffer_.clear();
                strip_carriage_return(line);
                return Result::kLine;
            }

            char chunk[256];
            const ssize_t got = ::read(fd_, chunk, sizeof chunk);
            if (got > 0) {
                buffer_.append(chunk, static_cast<std::size_t>(got));
                continue;
            }
            if (got == 0) {
                eof_ = true;
                continue;
            }
            if (errno == EINTR) {
                // Whatever was half-typed is dropped: Ctrl-C means "forget what
                // I was saying", and carrying it into the next line would put
                // it in front of something the user wrote afterwards.
                buffer_.clear();
                return Result::kInterrupted;
            }
            // A descriptor that cannot be read has nothing more to give, and
            // there is no line to report it on. The session treats it as the
            // end, which is what it is.
            eof_ = true;
        }
    }

private:
    static void strip_carriage_return(std::string& line) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
    }

    int fd_;
    std::string buffer_;
    bool eof_ = false;
};

}  // namespace ash::cli
