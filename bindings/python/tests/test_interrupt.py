"""Ctrl-C, and what it does to a run in progress.

CPython raises KeyboardInterrupt from a C handler that trips a flag the main
thread looks at when it next reaches a bytecode boundary. A run never reaches
one: from the first byte of the request to the last it is inside libcurl with
the GIL released. Left alone, Ctrl-C during a run would therefore do nothing at
all -- and for an endpoint that accepted the connection and then went quiet, it
would do nothing for the full two-minute request timeout, which is exactly when
someone is most likely to be pressing it.

So a run takes the signal for its own length and hands it back afterwards. These
tests are about the three things that follow. A Ctrl-C stops a run that nothing
else would have stopped. A Ctrl-C that arrives after the run had already decided
how it ends does not turn that ending into an exception. And the signal is only
taken where taking it cannot mean taking it from someone else.
"""

import os
import signal
import socket
import threading
import time

import pytest

import ash

TASK = "list the files here and tell me what this project is"


class SilentServer:
    """A socket that accepts and then says nothing at all.

    The case Ctrl-C exists for: a connection that was established and never
    answered. No event has happened, so nothing a stream callback could see
    would let it stop this, and no answer is coming before the request timeout.
    """

    def __init__(self):
        self.listener = socket.socket()
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(8)
        self.port = self.listener.getsockname()[1]
        self.held = []
        threading.Thread(target=self._accept_forever, daemon=True).start()

    def _accept_forever(self):
        while True:
            try:
                self.held.append(self.listener.accept()[0])
            except OSError:
                return

    def close(self):
        for connection in self.held:
            connection.close()
        self.listener.close()


def interrupt_after(seconds):
    """Send this process a SIGINT once the run is under way.

    From a thread, because the main thread is the one that is blocked, and that
    is the situation being tested. Which thread the kernel hands the signal to
    does not matter: the handler is process-wide and the flag it sets is the
    same one either way.
    """
    timer = threading.Timer(seconds, os.kill, [os.getpid(), signal.SIGINT])
    timer.start()
    return timer


def test_ctrl_c_stops_a_run_that_nothing_else_would(provider, file_tools):
    server = SilentServer()
    timer = interrupt_after(0.5)
    started = time.time()
    try:
        with pytest.raises(KeyboardInterrupt):
            # No cancel token: the signal is the only thing that can end this.
            ash.Agent(provider(server.port), file_tools).run("say hi")
        elapsed = time.time() - started
    finally:
        timer.cancel()
        server.close()

    # The default request timeout is two minutes. A Ctrl-C that waited for it
    # would not be a Ctrl-C.
    assert elapsed < 30


def test_a_ctrl_c_that_arrives_too_late_is_not_an_exception(
        stub, provider, scratch_project, file_tools):
    # One step, and the model's first call asks for a tool -- so the tool is the
    # last thing the run does, and the loop then leaves on max_steps without
    # looking at the stop token again. The signal is delivered from inside that
    # tool, so by the time anything could act on it the run had already arrived
    # at its ending; the ending is what says whether the signal meant anything.
    def list_dir(path: str = ".") -> str:
        """List the entries of a directory."""
        os.kill(os.getpid(), signal.SIGINT)
        return "\n".join(sorted(os.listdir(path)))

    tools = ash.ToolSet()
    tools.add(list_dir)
    token = ash.CancelToken()
    server = stub()

    run = ash.Agent(provider(server.port), tools).run(TASK, max_steps=1, cancel=token)

    assert run.stop_reason == "max_steps"
    assert run.tools_called == ["list_dir"]
    # The signal did reach the run and did stop it -- the run simply did not need
    # stopping, so what came back is the ending it had already decided on rather
    # than an exception about a Ctrl-C that was a moment too late.
    assert token.cancelled


def test_a_program_with_its_own_handler_keeps_it(provider, file_tools):
    # A program that installed a handler for SIGINT is using the signal for
    # something of its own, and a library that took it would be making that
    # decision on the program's behalf. So the run leaves it alone -- and a
    # Ctrl-C during such a run stops nothing, which is why the timer below is
    # what ends this one.
    seen = []

    def handler(signum, frame):
        seen.append(signum)

    server = SilentServer()
    token = ash.CancelToken()
    timer = threading.Timer(1.5, token.cancel)
    timer.start()
    previous = signal.signal(signal.SIGINT, handler)
    started = time.time()
    try:
        signal_timer = interrupt_after(0.3)
        run = ash.Agent(provider(server.port), file_tools).run("say hi", cancel=token)
        elapsed = time.time() - started
    finally:
        signal_timer.cancel()
        signal.signal(signal.SIGINT, previous)
        timer.cancel()
        server.close()

    assert run.stop_reason == "cancelled"
    assert elapsed >= 1.4
    # And the program's own handler is still its own, and ran as soon as the run
    # let the main thread back into Python: the signal was left alone rather than
    # swallowed. Handlers are run at a bytecode boundary in the main thread, and
    # the run is the reason there was not one.
    assert seen == [signal.SIGINT]


def test_a_run_on_a_worker_thread_leaves_the_signal_alone(provider, file_tools):
    # A signal is delivered to the process, but Python only raises it in the main
    # thread -- that is where the handler runs and where the interrupt belongs. A
    # run made on a worker therefore does not take the signal, because it would
    # then be stopping itself with one that was meant for whoever called join().
    server = SilentServer()
    token = ash.CancelToken()
    timer = threading.Timer(1.5, token.cancel)
    timer.start()
    outcome = {}

    def work():
        try:
            outcome["run"] = ash.Agent(provider(server.port), file_tools).run(
                "say hi", cancel=token)
        except BaseException as error:  # whatever came out of the run is the point
            outcome["error"] = error

    worker = threading.Thread(target=work)
    started = time.time()
    worker.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            time.sleep(0.2)
            os.kill(os.getpid(), signal.SIGINT)
            time.sleep(0.5)
        worker.join(timeout=30)
        elapsed = time.time() - started
    finally:
        timer.cancel()
        server.close()

    # The main thread got the interrupt it was owed.
    assert "error" not in outcome
    # The worker's run was stopped by its timer, not by the signal: had the
    # worker taken it, this would have come back in milliseconds.
    assert outcome["run"].stop_reason == "cancelled"
    assert elapsed >= 1.4
