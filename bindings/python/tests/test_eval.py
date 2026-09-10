"""Grading a suite of recorded runs, offline, from Python.

The eval harness and the pytest suite are two ways of asking the same question
about a committed journal. This file is where they meet: a project can keep its
grading rules in a suite file, run them from its own tests, and get back the
same numbers the CLI prints.
"""

import json
import os
import subprocess

import pytest

import ash
from conftest import CLI, REPO_ROOT

SUITE = os.path.join("examples", "suites", "core.json")


@pytest.fixture(scope="module")
def report(repo_root):
    return ash.eval.run_suite(os.path.join(repo_root, SUITE))


def test_a_suite_runs_with_no_key_and_nothing_fails(report):
    assert report.suite == "core"
    assert report.failed == 0
    assert report.passed == len(report.jobs)


def test_a_report_agrees_with_the_cli(repo_root, report, tmp_path):
    # The CLI writes the jobs and the latencies; its totals are recomputed from
    # those rather than stored, so recomputing them here is the comparison -- and
    # the one that would catch a total drifting away from the jobs under it.
    written = tmp_path / "cli.json"
    environment = {"PATH": os.environ["PATH"], "HOME": str(tmp_path)}
    subprocess.run([CLI, "eval", "--suite", os.path.join(repo_root, SUITE),
                    "--json", str(written)],
                   check=True, env=environment, stdout=subprocess.DEVNULL)

    cli = json.loads(written.read_text(encoding="utf-8"))

    assert [job["id"] for job in report.jobs] == [job["id"] for job in cli["jobs"]]
    assert report.passed == sum(1 for job in cli["jobs"] if job["passed"])
    assert report.total_prompt_tokens == sum(job["prompt_tokens"] for job in cli["jobs"])
    assert report.total_completion_tokens == sum(job["completion_tokens"] for job in cli["jobs"])
    assert report.total_cost_usd == pytest.approx(
        sum(job["cost_usd"] for job in cli["jobs"] if job["cost_usd"] is not None))
    assert report.every_cost_known == all(job["cost_usd"] is not None for job in cli["jobs"])


def test_a_job_result_is_the_reports_own_rendering(report):
    for job in report.jobs:
        assert set(job) >= {"id", "passed", "failures", "stop_reason", "prompt_tokens"}
        assert job["passed"] is True
        assert job["failures"] == []
        assert "sk-" not in json.dumps(job)


def test_a_percentile_is_a_latency_some_call_had(report):
    latencies = sorted(report.latency_percentile_us(p) for p in (50, 95, 100))

    assert latencies[0] > 0
    assert latencies == sorted(latencies)


def test_a_report_round_trips_through_a_file(report, tmp_path):
    path = tmp_path / "report.json"

    ash.eval.write_report(report, str(path))
    read_back = ash.eval.read_report(str(path))

    assert read_back.suite == report.suite
    assert read_back.passed == report.passed
    assert read_back.total_prompt_tokens == report.total_prompt_tokens
    assert read_back.latency_percentile_us(50) == report.latency_percentile_us(50)


def test_a_report_against_itself_has_not_regressed(report):
    diff = ash.eval.diff_reports(report, report)

    assert diff.regressed is False
    assert diff.notes == []
    assert all(delta["kind"] == "unchanged" for delta in diff.jobs)


def test_a_job_that_disappears_is_a_regression(report, tmp_path):
    # Deleting the job that fails is the cheapest way to make a suite look
    # better, so a report that simply lacks a job is not an improvement.
    written = tmp_path / "full.json"
    ash.eval.write_report(report, str(written))
    whole = json.loads(written.read_text(encoding="utf-8"))
    whole["jobs"] = whole["jobs"][:1]
    trimmed = tmp_path / "trimmed.json"
    trimmed.write_text(json.dumps(whole), encoding="utf-8")

    shorter = ash.eval.read_report(str(trimmed))
    diff = ash.eval.diff_reports(report, shorter)

    assert diff.regressed is True
    removed = [delta for delta in diff.jobs if delta["kind"] == "REMOVED"]
    assert len(removed) == 1
    assert any("gone" in note for note in removed[0]["notes"])
    assert any("passing" in note for note in diff.notes)


def test_a_parsed_suite_is_a_suite(repo_root, report):
    suite = ash.eval.load_suite(os.path.join(repo_root, SUITE))

    assert suite.name == "core"
    assert len(suite) == 2
    assert suite.jobs[0]["journal"].endswith("openai.jsonl")
    assert ash.eval.run_suite(suite).passed == report.passed


def test_workers_do_not_change_the_report(repo_root, report):
    parallel = ash.eval.run_suite(os.path.join(repo_root, SUITE), jobs=0)

    assert parallel.passed == report.passed
    assert parallel.total_prompt_tokens == report.total_prompt_tokens
    assert [job["id"] for job in parallel.jobs] == [job["id"] for job in report.jobs]
    assert parallel.latency_percentile_us(95) == report.latency_percentile_us(95)


def test_a_suite_that_is_not_there_is_an_error(tmp_path):
    with pytest.raises(Exception):
        ash.eval.load_suite(str(tmp_path / "nope.json"))
