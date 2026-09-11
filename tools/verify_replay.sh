#!/usr/bin/env bash
#
# Proves the headline claim: a recorded run replays byte-identically, offline,
# with no API key and no network.
#
#   tools/verify_replay.sh [path-to-ash-binary]
#
# Checks, in order:
#   1. a run records a journal against a stub endpoint
#   2. the stub is killed, so every replay below is provably offline
#   3. N replays produce byte-identical output
#   4. the replayed transcript matches the recorded one
#   5. the journal never contains the API key

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ASH=${1:-$REPO_ROOT/build/fedora-clang/apps/cli/ash}
PORT=${PORT:-8123}
REPLAYS=${REPLAYS:-100}

# A value distinctive enough that finding it in the journal cannot be chance.
API_KEY="sk-verify-must-not-be-persisted"

if [[ ! -x "$ASH" ]]; then
    echo "verify_replay: no ash binary at $ASH" >&2
    echo "build it first: cmake --build --preset fedora-clang" >&2
    exit 2
fi

# Resolved now, because the script later changes into the scratch directory and
# a relative path would stop pointing at the binary.
ASH=$(realpath "$ASH")

# A check that cannot run is not a check, and it must not look like one that
# ran. Without this, a missing `cmp` is caught by the comparison below as a
# non-zero status and reported as replays that differ -- this script saying the
# output is not deterministic when it never read the output at all. Being
# wrong in that direction is worse than not running.
require() {
    for tool in "$@"; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            echo "verify_replay: $tool is not installed" >&2
            exit 2
        fi
    done
}
require cmp diff python3

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
    echo "verify_replay: FAIL: $*" >&2
    exit 1
}

# The transcript lines plus the summary, with mode-specific banners removed, so
# a recording and a replay can be compared on substance alone.
extract_body() {
    sed -n '/^  -> /,$p' "$1" | grep -v '^ash: replay verified' || true
}

echo "verify_replay: scratch dir $WORK"

# A fixed little project for the agent to inspect, so tool output is stable.
mkdir -p "$WORK/project/src"
printf 'demo\n' >"$WORK/project/README.md"
: >"$WORK/project/src/main.cpp"
cd "$WORK/project"

python3 "$REPO_ROOT/tools/stub_model.py" \
    --port "$PORT" \
    --script "$REPO_ROOT/tools/scripts/list_and_summarize.json" \
    --record "$WORK/requests.json" >"$WORK/stub.log" 2>&1 &
STUB_PID=$!

for _ in $(seq 1 50); do
    if python3 -c "
import socket, sys
s = socket.socket()
sys.exit(0 if s.connect_ex(('127.0.0.1', $PORT)) == 0 else 1)
" 2>/dev/null; then
        break
    fi
    sleep 0.1
done

echo "verify_replay: recording a run"
"$ASH" run \
    --journal "$WORK/demo.jsonl" \
    --api-key "$API_KEY" \
    --base-url "http://127.0.0.1:$PORT/v1" \
    "list the files here and tell me what this project is" >"$WORK/run.out" 2>&1 ||
    fail "the recording run failed: $(cat "$WORK/run.out")"

# From here on there is no model server: replaying must not need one.
kill "$STUB_PID" 2>/dev/null || true
wait "$STUB_PID" 2>/dev/null || true
STUB_PID=""

echo "verify_replay: replaying $REPLAYS times with the stub shut down"
"$ASH" replay "$WORK/demo.jsonl" >"$WORK/replay-1.out" 2>&1 ||
    fail "the first replay failed: $(cat "$WORK/replay-1.out")"

for i in $(seq 2 "$REPLAYS"); do
    "$ASH" replay "$WORK/demo.jsonl" >"$WORK/replay-$i.out" 2>&1 ||
        fail "replay $i failed: $(cat "$WORK/replay-$i.out")"
    cmp -s "$WORK/replay-1.out" "$WORK/replay-$i.out" ||
        fail "replay $i is not byte-identical to replay 1"
done

extract_body "$WORK/run.out" >"$WORK/run.body"
extract_body "$WORK/replay-1.out" >"$WORK/replay.body"
diff -u "$WORK/run.body" "$WORK/replay.body" >"$WORK/body.diff" ||
    fail "the replayed transcript differs from the recording:
$(cat "$WORK/body.diff")"

if grep -qF "$API_KEY" "$WORK/demo.jsonl"; then
    fail "the journal persisted the API key"
fi

if [[ ! -s "$WORK/run.body" ]]; then
    fail "the transcript body is empty, so the comparison proved nothing"
fi

echo
cat "$WORK/replay-1.out"
echo
echo "verify_replay: OK"
echo "  journal lines   $(wc -l <"$WORK/demo.jsonl")"
echo "  identical runs  $REPLAYS"
echo "  api key         absent from the journal"
