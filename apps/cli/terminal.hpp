#pragma once

// Reading a line, and reading Ctrl-C.
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
//
// Ctrl-C means two different things depending on where the session is, and the
// split is the whole design. At the prompt it means "stop reading", which the
// read above reports on its own. During a turn there is no read in progress, so
// the signal has to be caught and turned into a stop request for the run -- see
// InterruptHandler and TurnWatchdog below.

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <stop_token>
#include <string>
#include <sys/types.h>
#include <thread>
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

// Whether a Ctrl-C has been seen and not yet dealt with.
//
// The only object a signal handler may write. Everything the session would
// rather do in response -- stopping a run, waking a thread -- takes a lock or
// runs a callback, and none of that is async-signal-safe; a handler that tried
// it could deadlock against the very thread it interrupted.
//
// `volatile sig_atomic_t` and not `std::atomic`: the standard says a handler may
// store to the former and says nothing about the latter. There is exactly one
// consumer at a time, so no locking is needed beyond what the stop source does
// internally -- during a turn it is the watchdog, and at the prompt it is the
// read, which the kernel unblocks with EINTR.
inline volatile std::sig_atomic_t g_interrupted = 0;

inline void handle_interrupt(int) { g_interrupted = 1; }

inline void clear_interrupt() noexcept { g_interrupted = 0; }

// Installs the session's SIGINT handler, and puts the previous one back when it
// goes out of scope.
//
// One handler for the whole session rather than one per turn, because Ctrl-C
// should mean the same thing at the prompt as it does during a turn, and two
// handlers taking turns being installed is two chances to get the handover
// wrong.
//
// `sa_flags = 0`, so deliberately no SA_RESTART. The kernel restarts an
// interrupted read for you when SA_RESTART is set, and the session needs the
// opposite: "Ctrl-C at an empty prompt leaves" is a fact about the read that was
// in progress, and a read that was restarted never comes back to say so. A
// handler and a blocking read that must see EINTR cannot both have what they
// want, and here the read wins.
class InterruptHandler {
public:
    InterruptHandler() {
        struct sigaction action {};
        action.sa_handler = handle_interrupt;
        action.sa_flags = 0;
        ::sigemptyset(&action.sa_mask);
        ::sigaction(SIGINT, &action, &previous_);
    }

    ~InterruptHandler() { ::sigaction(SIGINT, &previous_, nullptr); }

    InterruptHandler(const InterruptHandler&) = delete;
    InterruptHandler& operator=(const InterruptHandler&) = delete;

private:
    struct sigaction previous_ {};
};

// Watches the interrupt flag for as long as a turn is running.
//
// The flag is read here rather than acted on in the handler because stopping a
// run means calling into a stop source. This polls instead, which costs one
// wake-up every few milliseconds against a turn that is waiting on a socket --
// an amount of delay nobody can perceive, bought in exchange for a handler that
// cannot possibly deadlock.
//
// It lives for exactly one turn. That is what keeps a Ctrl-C from ending the
// turn after the one it was aimed at: the signal is only turned into a stop
// request while something is watching, and between turns nothing is.
class TurnWatchdog {
public:
    // The delay a Ctrl-C can sit unnoticed before the run is told to stop. It
    // bounds how long the session feels unresponsive, and it is short enough
    // that the shell tool's own poll, at a hundred milliseconds, is the coarser
    // of the two.
    static constexpr std::chrono::milliseconds kPollInterval{5};

    explicit TurnWatchdog(std::stop_source& stop) : stop_(stop) {
        thread_ = std::jthread([this](std::stop_token self) {
            while (!self.stop_requested()) {
                if (g_interrupted != 0) {
                    stop_.request_stop();
                    return;
                }
                std::this_thread::sleep_for(kPollInterval);
            }
        });
    }

    // The jthread joins in its own destructor, one member after this body, and
    // by then it can only be in the sleep. Returning from here therefore means
    // nothing is watching the flag any more, which is what makes it safe for the
    // caller to clear it next.
    ~TurnWatchdog() { thread_.request_stop(); }

    TurnWatchdog(const TurnWatchdog&) = delete;
    TurnWatchdog& operator=(const TurnWatchdog&) = delete;

private:
    std::stop_source& stop_;
    std::jthread thread_;
};

}  // namespace ash::cli
