"""Fixtures the suite shares.

The suite is the example. Nothing here reaches a real endpoint: every test
either replays a journal committed to the repository or starts
tools/stub_model.py on a port of its own, so the whole thing runs offline, costs
nothing, and needs no key. That is the point of the bindings, so it would be an
odd suite that could not demonstrate it about itself.
"""

import os
import socket
import subprocess
import sys
import time

import pytest

REPO_ROOT = os.environ.get(
    "ASH_REPO",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# The CLI, for the one test that checks a Python report against the one the
# command line writes. Overridable because the bindings build their own CLI in
# their own build directory, and a test that could only run after someone had
# also built the default preset would be a test that quietly stopped running.
CLI = os.environ.get("ASH_CLI") or os.path.join(REPO_ROOT, "build", "fedora-clang",
                                                "apps", "cli", "ash")


# A key that is not a key: the stub accepts anything, and the suite asserts this
# string never reaches a journal, a report or a transcript.
API_KEY = "sk-pytest-must-not-be-persisted"
MODEL = "stub-model"

# What the stub's script finally answers, used to check that an answer survived
# a recording and a replay.
ANSWER_PHRASE = "small C++ project"


@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT


def _free_port():
    # Bound and released rather than handed out by the kernel to the stub
    # itself, because the port has to be known before the process starts.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class Stub:
    """A running stub model server, and where to point a provider at it."""

    def __init__(self, port, dialect, process):
        self.port = port
        self.dialect = dialect
        self.process = process

    @property
    def base_url(self):
        root = "http://127.0.0.1:%d" % self.port
        # The two dialects mount their endpoint differently: OpenAI-compatible
        # servers put everything under /v1, Anthropic's is at the root.
        return root + "/v1" if self.dialect == "openai" else root

    def stop(self):
        """Shut the server down.

        A test calls this to make the offline claim mean something: a replay
        that succeeds with the model server gone is a replay that did not talk
        to it. The fixture's cleanup is a no-op on a server that is already
        stopped.
        """
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=15)


@pytest.fixture
def stub():
    """Starts a stub model server and stops it when the test is over.

    One script entry is consumed per request and the stub does not rewind, so a
    test that needs two identical runs starts two stubs.
    """
    running = []

    def start(script="list_and_summarize.json", dialect="openai"):
        port = _free_port()
        path = script if os.path.isabs(script) else os.path.join(REPO_ROOT, "tools/scripts", script)
        process = subprocess.Popen(
            [sys.executable, os.path.join(REPO_ROOT, "tools/stub_model.py"),
             "--port", str(port), "--dialect", dialect, "--script", path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        running.append(process)

        deadline = time.time() + 15
        while time.time() < deadline:
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", port)) == 0:
                    return Stub(port, dialect, process)
            if process.poll() is not None:
                raise RuntimeError("the stub exited before it listened on port %d" % port)
            time.sleep(0.05)
        raise RuntimeError("the stub never came up on port %d" % port)

    yield start

    for process in running:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=15)


@pytest.fixture
def provider():
    """Builds a provider pointed at a stub."""
    import ash

    def build(port, dialect="openai"):
        root = "http://127.0.0.1:%d" % port
        if dialect == "openai":
            return ash.openai_compatible(base_url=root + "/v1", api_key=API_KEY, model=MODEL)
        return ash.anthropic(base_url=root, api_key=API_KEY, model=MODEL)

    return build


@pytest.fixture
def scratch_project(tmp_path, monkeypatch):
    """The little project the committed script describes, and the cwd inside it.

    The tools the script calls are real ones, so the recording is of a real
    directory. A replay does not read any of this -- the tools are rebuilt from
    the journal -- which is exactly why a replay can run anywhere.
    """
    project = tmp_path / "project"
    (project / "src").mkdir(parents=True)
    (project / "README.md").write_text("demo\n", encoding="utf-8")
    (project / "src" / "main.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")
    (project / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    monkeypatch.chdir(project)
    return project


@pytest.fixture
def file_tools():
    """The three tools the committed script asks for, implemented in Python."""
    import ash

    tools = ash.ToolSet()
    tools.add(lambda path=".": "\n".join(sorted(os.listdir(path))),
              name="list_dir", description="List the entries of a directory.")
    tools.add(read_file, name="read_file", description="Read a text file.")
    tools.add(write_file, name="write_file", description="Write a text file.")
    return tools


def read_file(path: str) -> str:
    """Read a text file and return its contents."""
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def write_file(path: str, content: str) -> str:
    """Write content to a text file."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return "wrote %d bytes to %s" % (len(content), path)


def normalize(path):
    """A journal with the two fields that are facts about the machine removed.

    How long a call took and when it happened are not part of the run, and every
    claim about a journal being reproducible is made modulo them.
    """
    import json

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
