#!/usr/bin/env bash
#
# Proves the streaming claim end to end, over a real socket against the stub.
#
#   tools/verify_streaming.sh [path-to-ash-binary]
#
# The claim is that streaming is a view of a call and not a second kind of call.
# That is a claim about two runs agreeing, so this runs the same task twice --
# once plain and once with --stream -- and diffs the journals they wrote. A
# divergence means a streamed recording is a recording of something else, and
# every eval that compares a streamed run to a plain one is measuring the
# recording format instead of the model.
#
# Checks, in order, for each dialect:
#   1. a streamed run and a plain run write byte-identical journals
#   2. the answer arrives on stdout, exactly once, and not a second time from
#      the summary printer
#   3. tool activity is printed before the answer it produced
#   4. token counts survive the round trip, which only holds if the request
#      asked the endpoint for usage in the first place
#   5. the journal still never contains the API key

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ASH=${1:-$REPO_ROOT/build/fedora-clang/apps/cli/ash}
SCRIPT=${SCRIPT:-$REPO_ROOT/tools/scripts/list_and_summarize.json}
PORT=${PORT:-8124}

API_KEY="sk-streaming-must-not-be-persisted"

# A phrase that only the script's final answer contains, used to check that the
# answer was printed and printed once.
ANSWER_PHRASE="small C++ project"
# The tool the script asks for, used to check that its activity came first.
TOOL_NAME="list_dir"

if [[ ! -x "$ASH" ]]; then
    echo "verify_streaming: no ash binary at $ASH" >&2
    echo "build it first: cmake --build --preset fedora-clang" >&2
    exit 2
fi
ASH=$(realpath "$ASH")

WORK=$(mktemp -d)
STUB_PID=""
cleanup() {
    if [[ -n "$STUB_PID" ]]; then
        kill "$STUB_PID" 2>/dev/null || true
        wait "$STUB_PID" 2>/dev/null || true
    fi
    rm -rf "$WORK"
}
trap cleanup EXIT

fail() {
    echo "verify_streaming: FAIL: $*" >&2
    exit 1
}

start_stub() {
    local dialect=$1
    python3 "$REPO_ROOT/tools/stub_model.py" \
        --port "$PORT" \
        --dialect "$dialect" \
        --script "$SCRIPT" >"$WORK/stub-$dialect.log" 2>&1 &
    STUB_PID=$!

    for _ in $(seq 1 50); do
        if python3 -c "
import socket, sys
s = socket.socket()
sys.exit(0 if s.connect_ex(('127.0.0.1', $PORT)) == 0 else 1)
" 2>/dev/null; then
            return
        fi
        sleep 0.1
    done
    fail "the $dialect stub never came up"
}

stop_stub() {
    kill "$STUB_PID" 2>/dev/null || true
    wait "$STUB_PID" 2>/dev/null || true
    STUB_PID=""
}

# Journals differ in exactly two fields that are properties of the machine
# rather than of the call: how long it took, and when it happened. Everything
# else is the run, and everything else has to match.
normalize() {
    python3 - "$1" <<'PY'
import json, sys
for line in open(sys.argv[1], encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    record = json.loads(line)
    record.pop("duration_us", None)
    record.pop("created_at", None)
    print(json.dumps(record, sort_keys=True))
PY
}

TASK="list the files here and tell me what this project is"

# One run against a freshly started stub.
#
# The stub hands out one script entry per request and does not reset, so a
# second run against the same process would begin wherever the first one left
# off and record an entirely different conversation. Restarting it is what makes
# the two journals comparable at all.
record_run() {
    local dialect=$1 provider=$2 base_url=$3 journal=$4 out=$5
    shift 5

    start_stub "$dialect"
    if ! "$ASH" run --provider "$provider" --base-url "$base_url" --api-key "$API_KEY" \
        --journal "$journal" "$@" "$TASK" >"$out" 2>&1; then
        stop_stub
        fail "$dialect run failed: $(cat "$out")"
    fi
    stop_stub
}

run_dialect() {
    local dialect=$1
    local provider=$2
    local base_url=$3

    echo "verify_streaming: $dialect: recording a plain run"
    record_run "$dialect" "$provider" "$base_url" \
        "$WORK/$dialect-plain.jsonl" "$WORK/$dialect-plain.out"

    echo "verify_streaming: $dialect: recording a streamed run"
    record_run "$dialect" "$provider" "$base_url" \
        "$WORK/$dialect-streamed.jsonl" "$WORK/$dialect-streamed.out" --stream

    normalize "$WORK/$dialect-plain.jsonl" >"$WORK/$dialect-plain.norm"
    normalize "$WORK/$dialect-streamed.jsonl" >"$WORK/$dialect-streamed.norm"
    diff -u "$WORK/$dialect-plain.norm" "$WORK/$dialect-streamed.norm" >"$WORK/$dialect.diff" ||
        fail "$dialect: a streamed run wrote a different journal than a plain one:
$(cat "$WORK/$dialect.diff")"

    local out="$WORK/$dialect-streamed.out"

    # Printed exactly once. Twice would mean the summary printer did not know
    # the answer had already gone out, which is the specific bug that makes a
    # streamed CLI print everything double.
    local count
    count=$(grep -cF "$ANSWER_PHRASE" "$out" || true)
    [[ "$count" == "1" ]] || fail "$dialect: the answer appears $count times, expected once"

    # The tool ran before the answer was written, so its line has to come
    # first. The loop only reports its steps at the end, so this is the check
    # that the live printing is actually live.
    local tool_line answer_line
    tool_line=$(grep -nF "  -> $TOOL_NAME" "$out" | head -1 | cut -d: -f1 || true)
    answer_line=$(grep -nF "$ANSWER_PHRASE" "$out" | head -1 | cut -d: -f1 || true)
    [[ -n "$tool_line" ]] || fail "$dialect: the tool call was never printed"
    [[ "$tool_line" -lt "$answer_line" ]] ||
        fail "$dialect: the answer was printed before the tool call that produced it"

    # Tokens survive streaming only because the request asks for them. The stub
    # sends a usage frame only when it was asked, exactly as the real endpoint
    # does, so a client that forgot the option records zeros here -- which would
    # otherwise go unnoticed until someone compared an eval report against a
    # streamed run and found every cost at zero.
    python3 - "$WORK/$dialect-streamed.jsonl" <<'PY' || fail "$dialect: the streamed journal recorded no completion tokens"
import json, sys

counts = []
for line in open(sys.argv[1], encoding="utf-8"):
    if not line.strip():
        continue
    record = json.loads(line)
    if record.get("kind") == "model_call":
        counts.append(record["response"]["usage"]["completion_tokens"])

sys.exit(0 if counts and all(count > 0 for count in counts) else 1)
PY

    if grep -qF "$API_KEY" "$WORK/$dialect-streamed.jsonl"; then
        fail "$dialect: the journal persisted the API key"
    fi

    echo "verify_streaming: $dialect: OK"
}

echo "verify_streaming: scratch dir $WORK"

# A fixed little project for the agent to inspect, so tool output is stable.
mkdir -p "$WORK/project/src"
printf 'demo\n' >"$WORK/project/README.md"
: >"$WORK/project/src/main.cpp"
cd "$WORK/project"

run_dialect openai openai "http://127.0.0.1:$PORT/v1"
run_dialect anthropic anthropic "http://127.0.0.1:$PORT"

echo
echo "verify_streaming: OK"
echo "  dialects        openai, anthropic"
echo "  journals        identical with and without --stream"
echo "  api key         absent from the journal"
