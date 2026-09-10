"""The headline claim, from Python: a recorded run replays offline, forever.

This is the file the bindings exist for. Everything in it answers a question
about the runtime by replaying a journal that is committed to the repository --
no key, no network, no budget -- which is what makes a journal a fixture rather
than a sample.
"""

import json
import os

import pytest

import ash

JOURNALS = ["examples/journals/openai.jsonl", "examples/journals/anthropic.jsonl"]
TOOLS = ["list_dir", "read_file", "write_file"]


@pytest.mark.parametrize("journal", JOURNALS)
def test_a_committed_journal_replays(repo_root, journal):
    run = ash.replay(os.path.join(repo_root, journal))

    assert run.stop_reason == "completed"
    assert run.tools_called == TOOLS
    assert "calc.py" in run.answer


def test_replaying_twice_gives_the_same_result(repo_root):
    path = os.path.join(repo_root, JOURNALS[0])

    first, second = ash.replay(path), ash.replay(path)

    assert first.stop_reason == second.stop_reason
    assert first.answer == second.answer
    assert first.tools_called == second.tools_called
    assert first.usage.total_tokens == second.usage.total_tokens


def test_a_replay_says_how_much_of_the_journal_it_used(repo_root):
    path = os.path.join(repo_root, JOURNALS[0])
    records = sum(1 for line in open(path, encoding="utf-8") if line.strip())

    run = ash.replay(path)

    # The header is a line but not an event, so the two counts differ by one.
    assert run.events_consumed == records - 1


def test_a_live_run_has_no_journal_to_have_consumed(stub, provider, scratch_project, file_tools):
    server = stub()
    run = ash.Agent(provider(server.port), file_tools).run("list the files here")

    assert run.events_consumed is None


def test_the_transcript_is_the_journals_own_rendering(repo_root):
    run = ash.replay(os.path.join(repo_root, JOURNALS[0]))

    # Dicts, not classes: these are the field names a journal uses, so there is
    # no second set of them to keep in step.
    assert [message["role"] for message in run.transcript] == [
        "system", "user", "assistant", "tool", "tool", "assistant", "tool", "assistant"]
    assert isinstance(run.transcript[1]["content"], str)


def test_usage_is_the_sum_over_every_model_call(repo_root):
    path = os.path.join(repo_root, JOURNALS[0])
    recorded = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    calls = [record for record in recorded if record.get("kind") == "model_call"]

    run = ash.replay(path)

    assert run.usage.prompt_tokens == sum(call["response"]["usage"]["prompt_tokens"] for call in calls)
    assert run.usage.completion_tokens == sum(
        call["response"]["usage"]["completion_tokens"] for call in calls)
    assert run.usage.total_tokens == run.usage.prompt_tokens + run.usage.completion_tokens


def test_steps_pair_each_answer_with_what_it_produced(repo_root):
    run = ash.replay(os.path.join(repo_root, JOURNALS[0]))

    # Two steps, not three: the recording's first answer asks for two tools at
    # once, and a step is a model call and everything it asked for. The last
    # model call asked for nothing and ended the run, so it is the answer rather
    # than a step.
    assert len(run.steps) == 2
    assert [call["name"] for call in run.steps[0]["assistant"]["tool_calls"]] == \
        ["list_dir", "read_file"]
    assert [message["role"] for message in run.steps[0]["tool_results"]] == ["tool", "tool"]
    assert run.steps[-1]["assistant"]["tool_calls"][0]["name"] == "write_file"


def test_a_journal_is_not_a_place_a_key_could_hide(repo_root):
    for journal in JOURNALS:
        text = open(os.path.join(repo_root, journal), encoding="utf-8").read()
        assert "sk-" not in text
        assert "api_key" not in text
        assert "authorization" not in text.lower()


def test_a_truncated_journal_is_a_replay_error(repo_root, tmp_path):
    # The run takes a different path than the recording did, which is the thing
    # a replay exists to notice. Catching it as its own type is the difference
    # between detecting drift and guessing at a crash.
    source = os.path.join(repo_root, JOURNALS[0])
    lines = [line for line in open(source, encoding="utf-8") if line.strip()]

    truncated = tmp_path / "truncated.jsonl"
    truncated.write_text("".join(lines[:3]), encoding="utf-8")

    with pytest.raises(ash.ReplayError):
        ash.replay(str(truncated))


def test_a_replay_error_says_what_diverged(repo_root, tmp_path):
    source = os.path.join(repo_root, JOURNALS[0])
    lines = [line for line in open(source, encoding="utf-8") if line.strip()]

    truncated = tmp_path / "truncated.jsonl"
    truncated.write_text("".join(lines[:3]), encoding="utf-8")

    with pytest.raises(ash.ReplayError) as raised:
        ash.replay(str(truncated))

    assert str(raised.value)


def test_a_missing_journal_is_an_error(tmp_path):
    with pytest.raises(Exception):
        ash.replay(str(tmp_path / "nope.jsonl"))


def test_the_result_prints_itself_readably(repo_root):
    run = ash.replay(os.path.join(repo_root, JOURNALS[0]))

    text = repr(run)

    # A stop reason is a word rather than a value, so it is not quoted. The
    # answer is, and a long one is cut short -- which is what keeps a repr one
    # line even when the answer is not.
    assert text.startswith("Result(stop_reason=completed, answer=")
    assert "..." in text
    assert "events_consumed=" in text
    assert "\n" not in text
