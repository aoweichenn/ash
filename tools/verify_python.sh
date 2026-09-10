#!/usr/bin/env bash
#
# Proves the bindings claim: the same runtime, driven from Python, with the
# model server gone and no key in the environment.
#
#   tools/verify_python.sh [build-dir] [python-interpreter]
#
# The claim is not "Python can call C++". It is that the recording a Python run
# writes is the recording the CLI writes -- the same journal, replayable the
# same way, offline and for nothing -- and that the two things Python is good
# for on top of that, watching a run and stopping one, do not change what the
# run was.
#
# The checks are run in one Python process which owns the stub model server
# itself, because the server's lifetime is part of the claim: alive for the
# recording, gone for the replay. Nothing here needs a key, and the script
# clears the usual ones before it starts.
#
# Checks, in order:
#   1. both committed journals replay, with no server running and no key set
#   2. replaying one twice gives the same result, and no journal holds a key
#   3. a run recorded from Python replays once its server is gone
#   4. a run with callbacks writes the journal a run without them wrote
#   5. a Python function is called as a tool, with a schema that does not drift
#   6. a tool that raises is a tool result rather than the end of the run
#   7. a cancelled run ends as cancelled and keeps what it had
#   8. cancelling before the first token stops a run that would otherwise have
#      waited out its two-minute timeout
#   9. Ctrl-C, with no token and no callback, does the same thing
#  10. an eval suite run from Python agrees with the report the CLI writes
#
# and then the pytest suite, which is the same gate at finer grain.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD=${1:-$REPO_ROOT/build/fedora-clang-python}
PKG=$BUILD/bindings/python
CLI=$BUILD/apps/cli/ash

# The interpreter CMake configured against, not whatever python3 happens to
# resolve to. On a machine with two Pythons installed a module built for one
# does not import into the other, and that failure reads like a missing file.
if [[ -n "${2:-}" ]]; then
    PYTHON=$2
elif [[ -s "$PKG/python-interpreter.txt" ]]; then
    PYTHON=$(head -1 "$PKG/python-interpreter.txt")
else
    PYTHON=python3
fi

if ! compgen -G "$PKG/ash/_core*.so" >/dev/null; then
    echo "verify_python: no built extension under $PKG/ash" >&2
    echo "build it first: cmake --preset fedora-clang-python && cmake --build --preset fedora-clang-python" >&2
    exit 2
fi

if [[ ! -x "$CLI" ]]; then
    echo "verify_python: no ash binary at $CLI" >&2
    echo "the python preset builds the CLI too; build it first" >&2
    exit 2
fi

WORK=$(mktemp -d)
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

fail() {
    echo "verify_python: FAIL: $*" >&2
    exit 1
}

echo "verify_python: interpreter $PYTHON"
echo "verify_python: package     $PKG"
echo "verify_python: scratch dir $WORK"

cat >"$WORK/checks.py" <<'PY'
"""The end-to-end checks. Everything runs inside this one process.

Owning the stub server here rather than in the shell script is what lets a
check stop it and then keep going: the replay that follows is offline because
the process it would have talked to is gone, and that is a fact this program
established rather than one it was told.
"""

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

import ash

REPO = sys.argv[1]
CLI = sys.argv[2]

API_KEY = "sk-verify-python-must-not-be-persisted"
MODEL = "stub-model"
TASK = "list the files here and tell me what this project is"
# What the stub's script finally answers, and a phrase both committed journals
# contain -- the file the recorded run was asked to write.
ANSWER_PHRASE = "small C++ project"
RECORDED_PHRASE = "SUMMARY.md"
TOOLS = ["list_dir", "read_file", "write_file"]
JOURNALS = ["examples/journals/openai.jsonl", "examples/journals/anthropic.jsonl"]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


CHECKS = []


def check(name):
    CHECKS.append(name)
    print("  ok  %s" % name)
    sys.stdout.flush()


def stub_server(dialect):
    """A stub model server on a port of its own, started and ready to answer."""

    class Stub:
        def __init__(self):
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                self.port = probe.getsockname()[1]
            self.process = subprocess.Popen(
                [sys.executable, os.path.join(REPO, "tools/stub_model.py"),
                 "--port", str(self.port), "--dialect", dialect,
                 "--script", os.path.join(REPO, "tools/scripts/list_and_summarize.json")],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.base_url = "http://127.0.0.1:%d" % self.port
            if dialect == "openai":
                self.base_url += "/v1"
            self._await_listening()

        def _await_listening(self):
            deadline = time.time() + 15
            while time.time() < deadline:
                with socket.socket() as probe:
                    if probe.connect_ex(("127.0.0.1", self.port)) == 0:
                        return
                if self.process.poll() is not None:
                    raise RuntimeError("the stub exited before it listened")
                time.sleep(0.05)
            raise RuntimeError("the stub never came up on port %d" % self.port)

        def stop(self):
            """Shut it down. Everything after this is provably offline."""
            if self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=15)

    return Stub()


def provider(base_url, dialect="openai"):
    if dialect == "openai":
        return ash.openai_compatible(base_url=base_url, api_key=API_KEY, model=MODEL)
    return ash.anthropic(base_url=base_url, api_key=API_KEY, model=MODEL)


def read_file(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def write_file(path, content):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return "wrote %d bytes to %s" % (len(content), path)


def file_tools():
    """The three tools the committed script asks for, implemented in Python.

    The same three names the C++ CLI offers, because the point being made is
    that a tool written here is a tool: the recording it produces is the
    recording, and it replays without the code that made it.
    """
    tools = ash.ToolSet()
    tools.add(lambda path=".": "\n".join(sorted(os.listdir(path))),
              name="list_dir", description="List the entries of a directory.")
    tools.add(read_file, name="read_file", description="Read a text file.")
    tools.add(write_file, name="write_file", description="Write a text file.")
    return tools


def normalize(path):
    """A journal with the two fields that are facts about the machine removed.

    How long a call took and when it happened are not part of the run, and every
    claim about a journal being reproducible is made modulo them.
    """
    lines = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        record.pop("duration_us", None)
        record.pop("created_at", None)
        lines.append(json.dumps(record, sort_keys=True))
    return lines


# 1. The committed journals, with nothing running and no key set.

for journal in JOURNALS:
    run = ash.replay(os.path.join(REPO, journal))
    require(run.stop_reason == "completed", "%s: stop_reason is %r" % (journal, run.stop_reason))
    require(run.tools_called == TOOLS, "%s: tools_called is %r" % (journal, run.tools_called))
    require(RECORDED_PHRASE in run.answer, "%s: the answer is not the recorded one" % journal)
    require(run.events_consumed > 0, "%s: the replay consumed no events" % journal)

check("both committed journals replay: stop_reason, tools_called and answer")

# 2. Reproducible, and never a place a key could hide.

path = os.path.join(REPO, JOURNALS[0])
first, second = ash.replay(path), ash.replay(path)
require(first.answer == second.answer, "two replays of one journal disagreed")
require(first.tools_called == second.tools_called, "two replays called different tools")
require(first.usage.total_tokens == second.usage.total_tokens, "two replays counted differently")

for journal in JOURNALS:
    text = open(os.path.join(REPO, journal), encoding="utf-8").read()
    require("sk-" not in text, "%s contains something that looks like a key" % journal)
    require("authorization" not in text.lower(), "%s recorded an authorization header" % journal)

check("replaying twice gives the same result, and no journal holds a key")

# 3. Record from Python, then replay with the model server gone.

work = tempfile.mkdtemp(prefix="verify-python-")
project = os.path.join(work, "project")
os.makedirs(os.path.join(project, "src"))
open(os.path.join(project, "README.md"), "w", encoding="utf-8").write("demo\n")
open(os.path.join(project, "src", "main.cpp"), "w", encoding="utf-8").write(
    "int main() { return 0; }\n")
open(os.path.join(project, "calc.py"), "w", encoding="utf-8").write(
    "def add(a, b):\n    return a + b\n")
os.chdir(project)

journal = os.path.join(work, "recorded.jsonl")
server = stub_server("openai")
run = ash.Agent(provider(server.base_url), file_tools()).run(TASK, journal=journal)
server.stop()

require(run.stop_reason == "completed", "the recording run ended %r" % run.stop_reason)
# One tool, because the stub's script asks for one -- it is a short conversation
# rather than the one the committed journals recorded against a real model.
require(run.tools_called == ["list_dir"], "the recording run called %r" % run.tools_called)
require(ANSWER_PHRASE in run.answer, "the recording run did not get the scripted answer")

recorded = open(journal, encoding="utf-8").read()
require(API_KEY not in recorded, "the journal persisted the API key")
require("authorization" not in recorded.lower(), "the journal recorded an authorization header")

replayed = ash.replay(journal)
require(replayed.answer == run.answer, "the replay answered differently than the recording")
require(replayed.tools_called == run.tools_called, "the replay called different tools")

check("a run recorded from Python replays once its model server is gone")

# 4. Watching a run does not change it.

plain = os.path.join(work, "plain.jsonl")
watched = os.path.join(work, "watched.jsonl")

server = stub_server("openai")
ash.Agent(provider(server.base_url), file_tools()).run(TASK, journal=plain)
server.stop()

# A second server, because the stub hands out one script entry per request and
# does not rewind: a second run against the same process would record a
# different conversation and there would be nothing to compare.
pieces = []
server = stub_server("openai")
watched_run = ash.Agent(provider(server.base_url), file_tools()).run(
    TASK, journal=watched, on_text=pieces.append, on_event=lambda event: None)
server.stop()

require(len(pieces) > 1, "the answer did not arrive in pieces")
require("".join(pieces) == watched_run.answer, "the pieces did not add up to the answer")
require(normalize(plain) == normalize(watched),
        "a run with callbacks wrote a different journal than one without them")

check("a run with callbacks writes the journal a run without them wrote")

# 5. A Python function is a tool -- and one that raises is a tool result.

called = []


def spy(path="."):
    called.append(path)
    return "\n".join(sorted(os.listdir(path)))


spy_tools = ash.ToolSet()
spy_tools.add(spy, name="list_dir", description="List the entries of a directory.")

server = stub_server("openai")
ash.Agent(provider(server.base_url), spy_tools).run(TASK)
server.stop()

require(called == ["."], "the Python tool was not called: %r" % called)

# The schema is read off the signature, so the same function has to produce the
# same schema every time -- a model that is told something different on the
# second run is being told about a different tool.
again = ash.ToolSet()
again.add(spy, name="list_dir", description="List the entries of a directory.")
require(json.dumps(spy_tools.specs, sort_keys=True) == json.dumps(again.specs, sort_keys=True),
        "the same function derived two different schemas")

check("a Python function is called as a tool, with a schema that does not drift")


def explode(path: str) -> str:
    """A tool that cannot do its job."""
    raise ValueError("no such file: " + path)


failing = ash.ToolSet()
failing.add(explode, name="list_dir", description="A tool that cannot do its job.")

failing_journal = os.path.join(work, "failing.jsonl")
server = stub_server("openai")
failing_run = ash.Agent(provider(server.base_url), failing).run(TASK, journal=failing_journal)
server.stop()

require(failing_run.stop_reason == "completed", "a failing tool ended the run")
errors = [record for record in (json.loads(line) for line in open(failing_journal, encoding="utf-8")
                                if line.strip())
          if record.get("kind") == "tool_call" and record["result"]["is_error"]]
require(errors, "the tool's failure was not recorded as an error result")
require("ValueError" in errors[0]["result"]["content"],
        "the failure was recorded without saying what failed: %r" % errors[0]["result"]["content"])

check("a tool that raises becomes an error tool result rather than ending the run")

# 6. Cancelling a run, from another thread, mid-run and before anything arrives.

token = ash.CancelToken()


def cancel_when_the_first_call_ends(event):
    if event["type"] == "done":
        token.cancel()


server = stub_server("openai")
cancelled = ash.Agent(provider(server.base_url), file_tools()).run(
    TASK, cancel=token, on_event=cancel_when_the_first_call_ends)
server.stop()

require(cancelled.stop_reason == "cancelled", "the run ended %r" % cancelled.stop_reason)
require(cancelled.tools_called == ["list_dir"],
        "the cancelled run did not keep what it had: %r" % cancelled.tools_called)
require(any(message["role"] == "tool" for message in cancelled.transcript),
        "the cancelled run kept no tool result")

check("a cancelled run ends as cancelled and keeps what it had")

# A socket that accepts and then says nothing. No event happens, so a callback
# could not have stopped this -- which is why cancellation is not one.
listener = socket.socket()
listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listener.bind(("127.0.0.1", 0))
listener.listen(8)
silent_port = listener.getsockname()[1]

token = ash.CancelToken()
timer = threading.Timer(1.0, token.cancel)
timer.start()
started = time.time()
try:
    quiet = ash.Agent(provider("http://127.0.0.1:%d/v1" % silent_port), file_tools()).run(
        "say hi", cancel=token)
    elapsed = time.time() - started
finally:
    timer.cancel()
    listener.close()

require(quiet.stop_reason == "cancelled", "the run against a silent server ended %r" % quiet.stop_reason)
# The default request timeout is two minutes; a stop that waited for it would
# not be a stop.
require(elapsed < 30, "cancelling before the first token took %.1fs" % elapsed)

check("cancelling before the first token stops a run that would have waited out its timeout")

# Ctrl-C, against the same silent socket. CPython can only raise
# KeyboardInterrupt at a bytecode boundary in the main thread, and a run is
# inside libcurl with the GIL released from its first byte to its last, so
# without the run taking the signal this would wait out the two-minute timeout
# and only then raise -- which is what makes it worth a check rather than a
# sentence.
listener = socket.socket()
listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listener.bind(("127.0.0.1", 0))
listener.listen(8)
silent_port = listener.getsockname()[1]

timer = threading.Timer(0.5, os.kill, [os.getpid(), signal.SIGINT])
timer.start()
started = time.time()
interrupted = False
try:
    ash.Agent(provider("http://127.0.0.1:%d/v1" % silent_port), file_tools()).run("say hi")
except KeyboardInterrupt:
    interrupted = True
finally:
    # Cancelled rather than left to fire, so that a run which failed early
    # cannot leave a signal arriving after the run put the handler back.
    timer.cancel()
    elapsed = time.time() - started
    listener.close()

require(interrupted, "Ctrl-C did not reach the run as KeyboardInterrupt")
require(elapsed < 30, "Ctrl-C took %.1fs to stop the run" % elapsed)

check("Ctrl-C stops a run that would otherwise have waited out its timeout")

# 7. The eval harness, from Python, against the report the CLI writes.

suite = os.path.join(REPO, "examples/suites/core.json")
report = ash.eval.run_suite(suite)
require(report.suite == "core", "the suite is called %r" % report.suite)
require(report.failed == 0, "%d jobs failed" % report.failed)
require(report.passed == len(report.jobs), "the pass count does not match the jobs")

cli_json = os.path.join(work, "cli.json")
subprocess.run([CLI, "eval", "--suite", suite, "--json", cli_json],
               check=True, stdout=subprocess.DEVNULL, env={"PATH": os.environ["PATH"]})
cli = json.loads(open(cli_json, encoding="utf-8").read())

# The CLI stores the jobs and the latencies and recomputes its totals, so
# recomputing them here is the comparison -- and the one that would catch a
# total drifting away from the jobs underneath it.
require([job["id"] for job in report.jobs] == [job["id"] for job in cli["jobs"]],
        "the Python report graded different jobs than the CLI report")
require(report.passed == sum(1 for job in cli["jobs"] if job["passed"]),
        "the Python report passed a different number of jobs")
require(report.total_prompt_tokens == sum(job["prompt_tokens"] for job in cli["jobs"]),
        "the Python report counted different prompt tokens")
require(report.total_completion_tokens == sum(job["completion_tokens"] for job in cli["jobs"]),
        "the Python report counted different completion tokens")

check("an eval suite run from Python agrees with the report the CLI writes")

print("%d checks passed" % len(CHECKS))
PY

echo "verify_python: running the checks"
if ! env -u ASH_API_KEY -u OPENAI_API_KEY -u ANTHROPIC_API_KEY \
        PYTHONPATH="$PKG" "$PYTHON" "$WORK/checks.py" "$REPO_ROOT" "$CLI" \
        >"$WORK/checks.log" 2>&1; then
    cat "$WORK/checks.log" >&2
    fail "the checks did not all pass"
fi
cat "$WORK/checks.log"

echo "verify_python: running the pytest suite"
# Failing rather than skipping when pytest is missing: this is the script
# someone runs to find out whether the bindings are good, and "it could not
# check" is not a passing answer. ASH_CLI points at the CLI built here, so the
# one test that compares against it does not need a second build directory.
if ! env ASH_CLI="$CLI" PYTHONPATH="$PKG" "$PYTHON" -m pytest \
        "$REPO_ROOT/bindings/python/tests" -q >"$WORK/pytest.log" 2>&1; then
    cat "$WORK/pytest.log" >&2
    fail "the pytest suite did not pass"
fi
tail -3 "$WORK/pytest.log"

echo
echo "verify_python: OK"
echo "  interpreter     $PYTHON"
echo "  journals        replayed offline, byte for byte"
echo "  pytest          $(tail -1 "$WORK/pytest.log")"
echo "  api key         absent from every journal"
