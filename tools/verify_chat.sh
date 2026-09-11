#!/usr/bin/env bash
#
# Proves the session is a conversation and not a loop of one-shot runs with a
# prompt in front of it.
#
#   tools/verify_chat.sh [path-to-ash-binary]
#
# The load-bearing claim is what the second turn sends. A session that forgot
# the first exchange would look exactly the same from outside -- same prompt,
# same streamed answer, same tool lines -- so the printed output cannot tell the
# two apart. The stub writes every request it receives to its own file and the
# checks below read the bodies, which is the only place the difference shows.
#
# Checks, in order:
#   1. the second turn's request carries the first turn's exchange, in order,
#      under exactly one system message
#   2. tool activity is printed before the answer it produced
#   3. a command really ran: its output is in the request that followed it,
#      rather than only in what the agent said about it
#   4. a denied tool call reaches the model as an ordinary failed result and the
#      turn carries on, instead of the run ending
#   5. /mode changes what is confirmed, and /exit leaves with 0
#   6. an end of input leaves with 0
#   7. Ctrl-C during a command ends the turn, keeping what the command had
#      already produced, and the session takes the next question
#   8. Ctrl-C during a model call does the same, on the other cancellation path
#   9. the API key is in neither the output nor any request body

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ASH=${1:-$REPO_ROOT/build/fedora-clang/apps/cli/ash}
SCRIPT=${SCRIPT:-$REPO_ROOT/tools/scripts/chat_session.json}
PORT=${PORT:-8144}

API_KEY="sk-session-must-not-be-persisted"

# The phrases the scripted model answers with, used to find its answers in the
# output without matching the prompt or the tool lines around them.
ANSWER_ONE="There is a README.md and a src directory here."
ANSWER_THREE="The build printed two lines"
ANSWER_FOUR="Understood -- I have not removed anything."

if [[ ! -x "$ASH" ]]; then
    echo "verify_chat: no ash binary at $ASH" >&2
    echo "build it first: cmake --build --preset fedora-clang" >&2
    exit 2
fi
ASH=$(realpath "$ASH")

require() {
    for tool in "$@"; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            echo "verify_chat: $tool is not installed" >&2
            exit 2
        fi
    done
}
require python3

WORK=$(mktemp -d)
STUB_PID=""
ASH_PID=""
cleanup() {
    for pid in "$ASH_PID" "$STUB_PID"; do
        if [[ -n "$pid" ]]; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    rm -rf "$WORK"
}
trap cleanup EXIT

fail() {
    echo "verify_chat: FAIL: $*" >&2
    exit 1
}

wait_for_port() {
    local port=$1
    for _ in $(seq 1 60); do
        if python3 -c "
import socket, sys
s = socket.socket()
sys.exit(0 if s.connect_ex(('127.0.0.1', $port)) == 0 else 1)
" 2>/dev/null; then
            return 0
        fi
        sleep 0.1
    done
    return 1
}

start_stub() {
    local script=$1
    shift
    python3 "$REPO_ROOT/tools/stub_model.py" --port "$PORT" --script "$script" "$@" \
        >"$WORK/stub.log" 2>&1 &
    STUB_PID=$!
    wait_for_port "$PORT" || fail "the stub never came up"
}

stop_stub() {
    kill "$STUB_PID" 2>/dev/null || true
    wait "$STUB_PID" 2>/dev/null || true
    STUB_PID=""
}

# Sends the session a Ctrl-C and waits for it to come back, reporting how long
# that took in INTERRUPT_ELAPSED. Twenty seconds is the bound on the wait: a
# session that ignored the signal would sit on the command's own thirty.
interrupt_and_time() {
    kill -0 "$ASH_PID" 2>/dev/null || fail "the session ended before the signal was sent"
    local start end
    start=$(date +%s.%N)
    kill -INT "$ASH_PID"

    local alive=1
    for _ in $(seq 1 200); do
        if ! kill -0 "$ASH_PID" 2>/dev/null; then
            alive=0
            break
        fi
        sleep 0.1
    done
    if [[ "$alive" == "1" ]]; then
        kill -KILL "$ASH_PID" 2>/dev/null || true
        ASH_PID=""
        fail "$1: still running twenty seconds after the Ctrl-C"
    fi

    wait "$ASH_PID" 2>/dev/null || fail "$1: the session exited non-zero"
    ASH_PID=""
    end=$(date +%s.%N)
    INTERRUPT_ELAPSED=$(python3 -c "print(f'{$end - $start:.2f}')")
}

echo "verify_chat: scratch dir $WORK"

# A fixed little project, so tool output is stable.
mkdir -p "$WORK/project/src"
printf 'demo\n' >"$WORK/project/README.md"
: >"$WORK/project/src/main.cpp"
cd "$WORK/project"

# The request-body checks. They are a program of their own rather than a
# heredoc each, because a heredoc here cannot be attached to the same command
# as its own failure message: the shell starts the body on the next physical
# line, so a `|| fail "..."` written after the redirect is swallowed as the
# check's first line. One file with named checks has no such seam.
cat >"$WORK/checks.py" <<'PY'
"""What the session actually sent, read back from the stub's request files.

Each check opens a dumped request and answers whether one claim about it holds.
The request files are the point: a session that forgot the previous turn would
print exactly the same output as one that remembered it, so the bodies are the
only place the difference is visible.
"""

import json
import os
import sys


def messages(dump, number):
    path = os.path.join(dump, f"request-{number:04d}.json")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)["messages"]


def second_turn_roles(dump):
    """The first request of the second turn, as a role sequence.

    Turn one takes two requests -- the tool call and the answer to it -- so the
    third is where turn two begins. The order is the part that carries the
    meaning: a history that arrived scrambled would still hold every phrase.
    """
    roles = [message["role"] for message in messages(dump, 3)]
    return roles == ["system", "user", "assistant", "tool", "assistant", "user"]


def second_turn_carries_the_first(dump):
    text = json.dumps(messages(dump, 3))
    return "list the files here" in text and "add a version file" in text


def shell_output_came_back(dump):
    """The command's real output, in the request that followed it.

    The sixth request is the model's reply to the tool result, so the result is
    in there. `building` cannot have come from the agent: nothing wrote it but
    the echo the command ran.
    """
    results = [
        str(message.get("content", ""))
        for message in messages(dump, 6)
        if message["role"] == "tool"
    ]
    return any("building" in result and "ok" in result for result in results)


def refusal_came_back(dump):
    """A denied call reached the model as a failed result.

    That the eighth request exists at all is the other half of the claim: the
    turn carried on after the refusal instead of the run ending.
    """
    return any(
        message["role"] == "tool" and "denied" in str(message.get("content", "")).lower()
        for message in messages(dump, 8)
    )


CHECKS = {
    "second-turn-roles": second_turn_roles,
    "second-turn-carries-the-first": second_turn_carries_the_first,
    "shell-output-came-back": shell_output_came_back,
    "refusal-came-back": refusal_came_back,
}

sys.exit(0 if CHECKS[sys.argv[1]](sys.argv[2]) else 1)
PY


# ---------------------------------------------------------------------------
# The session itself, fed from a pipe.
#
# The order below is the order the session reads: a line is the question, and
# the line after it is the answer to the approval that question provokes. That
# makes the whole run deterministic rather than a race between the reader and
# the agent.
# ---------------------------------------------------------------------------

start_stub "$SCRIPT" --dump-requests "$WORK/dump"

printf '%s\n' \
    'list the files here' \
    'add a version file' \
    'run the build' \
    'y' \
    'remove everything' \
    'n' \
    '/mode' \
    '/exit' |
    "$ASH" --mode edits --base-url "http://127.0.0.1:$PORT/v1" --api-key "$API_KEY" \
        >"$WORK/session.out" 2>&1 ||
    fail "the session exited non-zero: $(cat "$WORK/session.out")"

stop_stub

[[ -f "$WORK/dump/request-0003.json" ]] ||
    fail "the stub received fewer requests than the conversation needed"

# 1. The second turn's opening request carries the whole of the first turn.
python3 "$WORK/checks.py" second-turn-roles "$WORK/dump" ||
    fail "the second turn's request is not the first turn's exchange followed by the second question"
python3 "$WORK/checks.py" second-turn-carries-the-first "$WORK/dump" ||
    fail "the second turn's request does not carry the first turn's words"

# 2. Tool activity before the answer it produced. The loop only reports its
# steps at the end, so this is the check that the printing is live.
tool_line=$(grep -nF "  -> list_dir" "$WORK/session.out" | head -1 | cut -d: -f1 || true)
answer_line=$(grep -nF "$ANSWER_ONE" "$WORK/session.out" | head -1 | cut -d: -f1 || true)
[[ -n "$tool_line" ]] || fail "the tool call was never printed"
[[ -n "$answer_line" ]] || fail "the agent's answer was never printed"
[[ "$tool_line" -lt "$answer_line" ]] ||
    fail "the answer was printed before the tool call that produced it"

# 3. The command ran, rather than the agent saying it did.
python3 "$WORK/checks.py" shell-output-came-back "$WORK/dump" ||
    fail "the shell command's output is not in the request that followed it"

# 4. A refusal is a result, not an ending.
python3 "$WORK/checks.py" refusal-came-back "$WORK/dump" ||
    fail "a denied tool call did not come back to the model as a failed result"
grep -qF "$ANSWER_FOUR" "$WORK/session.out" ||
    fail "the turn did not finish after the refusal"

# 5. /mode moved the session on, and /exit left. The exit status was checked by
# the pipeline above.
grep -qF "yolo -- nothing is confirmed" "$WORK/session.out" ||
    fail "/mode did not change the mode"

# 6. An end of input is a way out too, and not an error.
printf '' | "$ASH" --mode edits --base-url "http://127.0.0.1:$PORT/v1" --api-key "$API_KEY" \
    >"$WORK/eof.out" 2>&1 ||
    fail "an end of input did not leave with 0"

# 7. Ctrl-C during a command. The turn asks for one that would take thirty
# seconds; the session is signalled once the command is under way, which is
# known from the activity line rather than guessed at with a sleep.
cat >"$WORK/slow.json" <<'JSON'
[
  {"usage": {"prompt": 300, "completion": 20},
   "tool_calls": [{"id": "s1", "name": "run_shell",
                   "arguments": {"command": "echo starting; sleep 30"}}]},
  {"usage": {"prompt": 340, "completion": 12},
   "content": "the answer to the second question"}
]
JSON

start_stub "$WORK/slow.json"

printf 'run it\nand now?\n' |
    "$ASH" --mode yolo --base-url "http://127.0.0.1:$PORT/v1" --api-key "$API_KEY" \
        >"$WORK/interrupted.out" 2>&1 &
ASH_PID=$!

for _ in $(seq 1 100); do
    if grep -qF "run_shell" "$WORK/interrupted.out" 2>/dev/null; then
        break
    fi
    kill -0 "$ASH_PID" 2>/dev/null || fail "the session ended before the command started"
    sleep 0.1
done
grep -qF "run_shell" "$WORK/interrupted.out" || fail "the shell command never started"
# The activity line is printed before the command is spawned, and the output it
# has produced by the time of the signal is the thing being checked for later.
sleep 1

interrupt_and_time "a command"
# Well inside the thirty seconds it asked for, which is what says the signal
# ended the turn rather than the command ending on its own.
python3 -c "import sys; sys.exit(0 if $INTERRUPT_ELAPSED < 10 else 1)" ||
    fail "the interrupted turn ran for ${INTERRUPT_ELAPSED}s, so it was not interrupted"
grep -qF "starting" "$WORK/interrupted.out" ||
    fail "the output the command produced before the interrupt was lost"
grep -qiF "cancel" "$WORK/interrupted.out" ||
    fail "nothing in the output says the command was cancelled"
grep -qF "the answer to the second question" "$WORK/interrupted.out" ||
    fail "the session did not survive the interrupt"

stop_stub

# 8. Ctrl-C during a model call, which is the other path: here the cancellation
# comes from the provider giving up a transfer that is in progress, rather than
# from a tool returning a result. The endpoint accepts the connection and says
# nothing, so the run is parked in libcurl with no frame in sight.
#
# Readiness is announced by the server rather than probed for. A probe is a TCP
# connection of its own, and this server prints a line for every connection it
# accepts -- so probing would leave the very `accepted` line the check below
# waits for already on disk, the signal would be sent before the turn began, and
# the turn's own clear_interrupt() would discard it. The session would then run
# normally and the check would report a hang that never happened.
python3 -c "
import socket
server = socket.socket()
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(('127.0.0.1', $PORT))
server.listen(8)
print('listening', flush=True)
held = []
while True:
    connection, _ = server.accept()
    held.append(connection)
    print('accepted', flush=True)
" >"$WORK/hang.log" 2>&1 &
STUB_PID=$!

for _ in $(seq 1 100); do
    grep -qF "listening" "$WORK/hang.log" 2>/dev/null && break
    kill -0 "$STUB_PID" 2>/dev/null || fail "the silent endpoint never came up"
    sleep 0.1
done
grep -qF "listening" "$WORK/hang.log" || fail "the silent endpoint never came up"

printf 'are you there\n' |
    "$ASH" --mode yolo --base-url "http://127.0.0.1:$PORT/v1" --api-key "$API_KEY" \
        >"$WORK/parked.out" 2>&1 &
ASH_PID=$!

for _ in $(seq 1 100); do
    if grep -qF "accepted" "$WORK/hang.log" 2>/dev/null; then
        break
    fi
    kill -0 "$ASH_PID" 2>/dev/null || fail "the session ended before it called the model"
    sleep 0.1
done
grep -qF "accepted" "$WORK/hang.log" || fail "the session never reached the model call"

interrupt_and_time "a model call"
grep -qiF "cancel" "$WORK/parked.out" ||
    fail "a turn parked on the model was not reported as cancelled"

stop_stub

# 9. The key is nowhere it could be read back from. The request bodies are what
# the stub wrote down, which is every byte the session sent to the endpoint.
if grep -qF "$API_KEY" "$WORK/session.out" "$WORK/interrupted.out" "$WORK/parked.out"; then
    fail "the API key reached the output"
fi
for dumped in "$WORK"/dump/*.json; do
    if grep -qF "$API_KEY" "$dumped"; then
        fail "the API key reached a request body: $dumped"
    fi
done

echo
echo "verify_chat: OK"
echo "  multi-turn      the second turn's request carries the first exchange"
echo "  printing        tool activity before the answer it produced"
echo "  shell           the command's real output reached the model"
echo "  refusal         a denied call came back as a failed result"
echo "  commands        /mode changed, /exit and an end of input both left 0"
echo "  Ctrl-C          a command and a model call, both ended the turn"
echo "  api key         absent from the output and from every request body"
