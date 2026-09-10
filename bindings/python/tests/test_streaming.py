"""Watching a run, and stopping one.

Two claims are checked here rather than asserted in prose. Watching a run does
not change it: a run with callbacks writes the journal a run without them wrote.
And stopping a run is not the same thing as failing one: the run ends, says why,
and keeps what it had.
"""

import socket
import threading
import time

import pytest

import ash
from conftest import ANSWER_PHRASE, normalize

TASK = "list the files here and tell me what this project is"


def test_events_arrive_in_order_and_end_with_done(stub, provider, scratch_project, file_tools):
    events = []
    server = stub()

    ash.Agent(provider(server.port), file_tools).run(TASK, on_event=events.append)

    kinds = [event["type"] for event in events]
    assert kinds[-1] == "done"
    assert "text" in kinds
    assert "usage" in kinds
    # The tool is called between the model calls, so the tool-call fragment the
    # endpoint streamed is in there too.
    assert "tool_call" in kinds
    assert kinds.count("done") == 2  # one per model call


def test_every_event_carries_its_own_fields(stub, provider, scratch_project, file_tools):
    events = []
    server = stub()

    ash.Agent(provider(server.port), file_tools).run(TASK, on_event=events.append)

    for event in events:
        assert isinstance(event, dict)
        assert isinstance(event["type"], str)
    for event in events:
        if event["type"] == "text":
            assert isinstance(event["text"], str)
        elif event["type"] == "usage":
            assert isinstance(event["prompt_tokens"], int)
            assert isinstance(event["completion_tokens"], int)
        elif event["type"] == "done":
            assert event["finish_reason"] in ("stop", "tool_calls")
            assert event["model"]
        elif event["type"] == "tool_call":
            assert isinstance(event["index"], int)
            assert isinstance(event["arguments"], str)  # a fragment, not JSON


def test_on_text_is_the_answer_arriving_in_pieces(stub, provider, scratch_project, file_tools):
    pieces = []
    server = stub()

    run = ash.Agent(provider(server.port), file_tools).run(TASK, on_text=pieces.append)

    assert len(pieces) > 1
    assert "".join(pieces) == run.answer
    assert ANSWER_PHRASE in "".join(pieces)


def test_watching_a_run_does_not_change_it(stub, provider, scratch_project, file_tools, tmp_path):
    plain = str(tmp_path / "plain.jsonl")
    watched = str(tmp_path / "watched.jsonl")

    # Two stubs, because one script entry is consumed per request and the second
    # run has to start where the first one did.
    server = stub()
    ash.Agent(provider(server.port), file_tools).run(TASK, journal=plain)
    server.stop()

    server = stub()
    ash.Agent(provider(server.port), file_tools).run(
        TASK, journal=watched, on_text=lambda text: None, on_event=lambda event: None)
    server.stop()

    # Everything but the two fields that are facts about the machine.
    assert normalize(plain) == normalize(watched)


def test_a_callback_that_raises_stops_the_run_and_the_error_comes_back(
        stub, provider, scratch_project, file_tools):
    def explode(text):
        raise ValueError("callback said no")

    server = stub()

    with pytest.raises(ValueError, match="callback said no"):
        ash.Agent(provider(server.port), file_tools).run(TASK, on_text=explode)


def test_raising_cancelled_from_a_callback_ends_the_run_instead(
        stub, provider, scratch_project, file_tools):
    def stop_it(event):
        raise ash.Cancelled()

    server = stub()

    run = ash.Agent(provider(server.port), file_tools).run(TASK, on_event=stop_it)

    assert run.stop_reason == "cancelled"


def test_a_token_cancelled_before_the_run_stops_it_before_it_asks(
        stub, provider, scratch_project, file_tools):
    server = stub()
    token = ash.CancelToken()
    token.cancel()

    run = ash.Agent(provider(server.port), file_tools).run("say hi", cancel=token)

    assert run.stop_reason == "cancelled"
    assert run.tools_called == []
    assert token.cancelled


def test_cancelling_mid_run_keeps_what_the_run_had(stub, provider, scratch_project, file_tools):
    server = stub()
    token = ash.CancelToken()

    def cancel_when_the_first_call_ends(event):
        if event["type"] == "done":
            token.cancel()

    run = ash.Agent(provider(server.port), file_tools).run(
        TASK, cancel=token, on_event=cancel_when_the_first_call_ends)

    assert run.stop_reason == "cancelled"
    # The model call that finished before the stop, and the tool it asked for.
    assert len(run.transcript) >= 4
    assert run.tools_called == ["list_dir"]


def test_cancelling_a_run_that_never_answered(stub, provider, file_tools):
    # A socket that accepts and then says nothing. No event has happened, so
    # nothing a sink could see would have stopped this -- which is exactly why
    # cancellation does not go through one.
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    port = listener.getsockname()[1]
    held = []

    def accept_forever():
        while True:
            try:
                held.append(listener.accept()[0])
            except OSError:
                return

    threading.Thread(target=accept_forever, daemon=True).start()

    token = ash.CancelToken()
    timer = threading.Timer(1.0, token.cancel)
    timer.start()
    started = time.time()
    try:
        run = ash.Agent(provider(port), file_tools).run("say hi", cancel=token)
        elapsed = time.time() - started
    finally:
        timer.cancel()
        for connection in held:
            connection.close()
        listener.close()

    assert run.stop_reason == "cancelled"
    # The default request timeout is two minutes; a stop that waited for it
    # would not be a stop.
    assert elapsed < 30


def test_a_token_prints_itself_readably():
    token = ash.CancelToken()

    assert repr(token) == "CancelToken(cancelled=False)"
    token.cancel()
    assert repr(token) == "CancelToken(cancelled=True)"
    token.cancel()  # cancelling twice is not an error
