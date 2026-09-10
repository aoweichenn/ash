"""Recording a run from Python, and replaying it once the model is gone.

The recipe this file exists to show is three lines long: run against an
endpoint with a journal, stop the endpoint, replay the journal forever. What
makes it work is that the recording is taken at the provider and the tool seams,
so what is written down is what the run did rather than what the wire carried.
"""

import json
import os

import pytest

import ash
from conftest import ANSWER_PHRASE, API_KEY, MODEL


TASK = "list the files here and tell me what this project is"


def test_a_run_records_a_journal_that_then_replays_offline(
        stub, provider, scratch_project, file_tools, tmp_path):
    journal = str(tmp_path / "run.jsonl")
    server = stub()

    run = ash.Agent(provider(server.port), file_tools).run(TASK, journal=journal)

    assert run.stop_reason == "completed"
    assert run.tools_called == ["list_dir"]
    assert ANSWER_PHRASE in run.answer

    # The key lives in the provider's own request headers and nowhere else, so
    # there is no path by which it could reach the journal. The suite checks
    # rather than trusts it, because a journal is meant to be committed.
    text = open(journal, encoding="utf-8").read()
    assert API_KEY not in text
    assert "authorization" not in text.lower()

    # With the model server shut down, and no key in the environment.
    server.stop()
    replayed = ash.replay(journal)

    assert replayed.stop_reason == run.stop_reason
    assert replayed.tools_called == run.tools_called
    assert replayed.answer == run.answer
    assert replayed.usage.total_tokens == run.usage.total_tokens


def test_the_header_says_what_the_run_was(stub, provider, scratch_project, file_tools, tmp_path):
    journal = str(tmp_path / "run.jsonl")
    server = stub()

    ash.Agent(provider(server.port), file_tools).run(
        TASK, journal=journal, system_prompt="You are terse.", max_steps=5)

    header = json.loads(open(journal, encoding="utf-8").readline())
    assert header["kind"] == "header"
    assert header["task"] == TASK
    assert header["system_prompt"] == "You are terse."
    assert header["model"] == MODEL
    assert header["max_steps"] == 5


def test_the_system_prompt_is_a_message_in_the_transcript(stub, provider, scratch_project, file_tools):
    server = stub()

    run = ash.Agent(provider(server.port), file_tools).run(TASK, system_prompt="You are terse.")

    assert run.transcript[0]["role"] == "system"
    assert run.transcript[0]["content"] == "You are terse."


def test_a_run_without_a_journal_writes_nothing(stub, provider, scratch_project, file_tools, tmp_path):
    server = stub()
    before = set(os.listdir(tmp_path))

    ash.Agent(provider(server.port), file_tools).run(TASK)

    assert set(os.listdir(tmp_path)) == before


def test_max_steps_ends_a_run_that_would_keep_going(stub, provider, scratch_project, file_tools):
    server = stub()

    run = ash.Agent(provider(server.port), file_tools).run(TASK, max_steps=1)

    # One model call and the tool it asked for, and then the budget ran out --
    # which the run says rather than treating as an answer.
    assert run.stop_reason == "max_steps"
    assert run.tools_called == ["list_dir"]


def test_both_dialects_can_be_recorded_and_replayed(
        stub, provider, scratch_project, file_tools, tmp_path):
    for dialect in ("openai", "anthropic"):
        journal = str(tmp_path / ("%s.jsonl" % dialect))
        server = stub(dialect=dialect)

        run = ash.Agent(provider(server.port, dialect), file_tools).run(TASK, journal=journal)
        server.stop()

        assert run.stop_reason == "completed"
        assert ash.replay(journal).answer == run.answer


def test_a_python_tool_really_runs(stub, provider, scratch_project):
    calls = []

    def list_dir(path="."):
        calls.append(path)
        return "\n".join(sorted(os.listdir(path)))

    spy = ash.ToolSet()
    spy.add(list_dir, name="list_dir", description="List the entries of a directory.")
    server = stub()

    run = ash.Agent(provider(server.port), spy).run(TASK)

    assert calls == ["."]
    assert run.tools_called == ["list_dir"]


def test_a_tool_that_raises_becomes_a_tool_result_not_a_crash(
        stub, provider, scratch_project):
    def explode(path: str) -> str:
        """A tool that cannot do its job."""
        raise ValueError("no such file: " + path)

    tools = ash.ToolSet()
    tools.add(explode, name="list_dir", description="A tool that cannot do its job.")
    server = stub()

    run = ash.Agent(provider(server.port), tools).run(TASK)

    # The run finishes, and the model is handed the failure as the tool's
    # output -- which is what gives it a chance to correct its own arguments.
    assert run.stop_reason == "completed"
    assert run.tools_called == ["list_dir"]
    outputs = [message["content"] for message in run.transcript if message["role"] == "tool"]
    assert outputs == ["ValueError: no such file: ."]


def test_an_unknown_tool_is_reported_to_the_model(stub, provider, scratch_project):
    server = stub()

    # No tools at all: the script still asks for list_dir.
    run = ash.Agent(provider(server.port), ash.ToolSet()).run(TASK)

    outputs = [message["content"] for message in run.transcript if message["role"] == "tool"]
    assert outputs == ["unknown tool: list_dir"]
    assert run.stop_reason == "completed"


def test_the_agent_prints_itself_readably(stub, provider, file_tools):
    server = stub()
    agent = ash.Agent(provider(server.port), file_tools)

    assert "3 tools)" in repr(agent)
    assert MODEL in repr(agent)
