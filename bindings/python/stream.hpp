#pragma once

// Streaming callbacks and cancellation, which are one subject: a caller that
// wants to watch work in progress is a caller that wants to stop it, and the
// reason the two are not implemented through the same mechanism is worth
// reading the notes on CancelToken.

#include <optional>
#include <stop_token>
#include <string>
#include <thread>

#include <pybind11/pybind11.h>

#include "ash/model/stream.hpp"

namespace ash::python {

namespace py = pybind11;  // the same alias convert.hpp declares

// Ends a run from outside it.
//
// It owns the stop source rather than polling something, so cancel() is a plain
// request_stop(): thread-safe, no GIL, callable from any thread. That is what
// makes threading.Timer(30, token.cancel) a timeout in one line.
//
// Deliberately not routed through the stream sink. A sink only exists when the
// caller asked for text as it arrives, so cancelling through one would leave
// the timeout unable to fire before the first token -- which is exactly when a
// stuck run needs stopping, since a connection that never answers holds for the
// whole request timeout. The stop request instead reaches the agent loop at its
// next step boundary and libcurl through its progress callback within about a
// second, so it works from the moment the run starts.
class CancelToken {
public:
    void cancel() noexcept { source_.request_stop(); }
    [[nodiscard]] bool cancelled() const noexcept { return source_.stop_requested(); }
    [[nodiscard]] std::stop_token token() const noexcept { return source_.get_token(); }

private:
    std::stop_source source_;
};

// Takes SIGINT for the length of a run, so that Ctrl-C stops the run.
//
// CPython raises KeyboardInterrupt from its own C signal handler by tripping a
// flag that the main thread checks at a bytecode boundary. A run never reaches
// one: from the first byte of the request to the last it is inside libcurl with
// the GIL released, so Ctrl-C during a run does nothing at all until the run is
// over -- which, for an endpoint that accepted the connection and then went
// quiet, is the full two-minute request timeout.
//
// So a run takes the signal and gives it back. The handler does the only thing
// a signal handler may do -- set a flag -- and a watchdog thread turns that flag
// into request_stop(), the same call a timer or a failing callback would have
// made. Nothing here needs the main thread to do anything, which is the point:
// the main thread is the one that is blocked. The watchdog is a plain
// std::thread that never touches Python, so it does not become a second place
// from which a callback could be made.
//
// Installed only where it is safe to install. The run has to be on the main
// thread, because that is the only thread a signal is delivered to, and taking
// the signal on a worker would take it away from the thread that owns it. And
// the program's own handler for SIGINT has to be the default one: a program
// that installed a handler is using the signal for something, and a library
// that replaced it would be making that decision on the program's behalf.
//
// Requires the GIL, which is why it is built before the run releases it.
class InterruptWatch {
public:
    explicit InterruptWatch(CancelToken& token);
    ~InterruptWatch();

    InterruptWatch(const InterruptWatch&) = delete;
    InterruptWatch& operator=(const InterruptWatch&) = delete;

    // Whether SIGINT arrived while the watch was up, whether or not the run got
    // as far as noticing it. False when no watch was installed: a signal nobody
    // was holding is not this object's to report.
    [[nodiscard]] bool interrupted() const noexcept;

private:
    CancelToken& token_;
    PyOS_sighandler_t previous_ = nullptr;
    bool installed_ = false;
    std::jthread watchdog_;
};

// Delivers a call while it is still arriving.
//
// on_event runs inside libcurl's write callback, on the transfer thread, with
// the GIL released by the run that is in flight. It therefore has to acquire
// the GIL -- there is no way to call Python without it -- and that is the whole
// argument for installing a sink only when the caller asked for one: a run
// without callbacks takes the GIL not once.
//
// The contract on StreamSink still holds: nothing escapes, and nothing blocks
// indefinitely. An exception a callback raises is caught here, remembered, and
// turned into a stop request; it is raised again on the way out, once the run
// has ended and the GIL is back in hand.
class PythonSink final : public StreamSink {
public:
    PythonSink(py::object on_text, py::object on_event, CancelToken& token);

    // The references are dropped under the same guard the tools use, in case a
    // sink somehow outlives the GIL.
    ~PythonSink() override;

    PythonSink(const PythonSink&) = delete;
    PythonSink& operator=(const PythonSink&) = delete;

    void on_event(const StreamEvent& event) override;

    // Raises whatever a callback raised, if anything. Called after the run with
    // the GIL held: an exception cannot be carried across the boundary where the
    // GIL went away, so it waits here instead.
    void rethrow_if_failed() const;

private:
    py::object on_text_;
    py::object on_event_;
    CancelToken& token_;

    // Set once the callbacks have failed, so that the events still sitting in
    // libcurl's buffer do not call a callback that has already raised and do not
    // overwrite the error worth reporting.
    bool failed_ = false;
    py::object error_type_;
    py::object error_value_;
    std::string error_message_;
};

void register_stream(py::module_& m);

}  // namespace ash::python
