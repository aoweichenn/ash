# ash

A deterministic agent runtime in C++20. Record a run once, replay it forever —
offline, with no API key, byte for byte.

```
$ ash run --journal demo.jsonl "list the files here and summarize this project"
ash: openai-compatible / deepseek-chat -> https://api.deepseek.com/v1
ash: recording to demo.jsonl
ash: task: list the files here and summarize this project

  -> list_dir {"path":"."}
  <- README.md ...

This directory holds README.md and a src/ folder containing main.cpp, so it is a small C++ project.

ash: stop=completed steps=1 tokens=354 (prompt 310, completion 44)

$ ash replay demo.jsonl          # no network, no key, no cost
ash: replaying demo.jsonl (openai-compatible / deepseek-chat)
...
ash: replay verified, 3 events consumed
```

## Why

A live agent run is not reproducible. The model is nondeterministic, the tools
touch the world, and the result costs money every time you re-run it. So agent
code gets tested by hand, and the tests that do exist mock the model into
something that no longer resembles one.

`ash` takes the position that the thing worth making deterministic is not the
model but the *run*. Every model call and every tool call is written to an
append-only journal as it happens. Replaying that journal re-executes the agent
loop for real — your control flow, your tool dispatch, your context assembly —
while every side effect is answered from the log. A replay is therefore:

- **offline** — it never opens a socket
- **free** — it never calls a paid API
- **deterministic** — the same journal produces the same bytes, always
- **verifying** — if the run asks a question the journal cannot answer, that is
  a divergence, and `ash` fails with the two hashes that disagree

That last point is what makes it a test rather than a recording. Changing your
system prompt, your tool schemas, or your message assembly changes the request
hash, and the replay tells you exactly where it drifted.

## Build

The only system dependencies are libcurl and OpenSSL.

```bash
sudo dnf install -y libcurl-devel openssl-devel     # or: apt install libcurl4-openssl-dev libssl-dev
cmake --preset fedora-clang
cmake --build --preset fedora-clang
ctest --preset fedora-clang
```

Everything else — nlohmann/json, spdlog, Catch2 — is fetched by CMake at
configure time, pinned to a tag.

To reproduce the headline claim, including the offline replay and the
byte-identity check across 100 runs:

```bash
./tools/verify_replay.sh build/fedora-clang/apps/cli/ash
```

`examples/journals/` holds two recordings made against a real endpoint — one per
wire dialect. They are replayed by the test suite, which means a change to how
the system prompt, the tool declarations, or the messages are assembled fails
the build with the two request hashes that disagree. The recordings are
regression tests that cost nothing to run:

```bash
./build/fedora-clang/apps/cli/ash replay examples/journals/openai.jsonl
```

Because the journals are the input, a whole suite of them can be graded offline
— no key, no network, no budget — which is what makes it safe to run on every
push:

```bash
./build/fedora-clang/apps/cli/ash eval --suite examples/suites/core.json
```

```
  job                           result   steps  tokens        cost   latency
  summary-openai-compatible     pass         2    2283   $0.001006    3.739s
  summary-anthropic             pass         3    1168   $0.000653    4.023s

  suite core: 2/2 passed
  tokens  3451 (prompt 2758, completion 693)
  cost    $0.001659
  latency p50 0.915s  p95 1.487s  (7 recorded model calls)
```

`--jobs` replays several jobs at once. It changes how long the suite takes and
nothing else — the report is assembled in suite order and every number in it
comes from a recording, so a parallel run and a serial one produce the same
bytes:

```bash
ash eval --suite examples/suites/core.json --jobs 1 --json serial.json
ash eval --suite examples/suites/core.json --jobs 8 --json parallel.json
cmp serial.json parallel.json          # byte for byte, and against the baseline
```

`--json` writes that report to a file, and `--baseline` reads an earlier one
back and reports what moved, exiting non-zero on a regression:

```bash
ash eval --suite examples/suites/core.json --json baseline.json
ash eval --suite examples/suites/core.json --baseline baseline.json
```

The report is a pure function of the journals, so it can be committed: there is
one under `eval/baseline/`, and CI grades every push against it. A code change
that breaks a check, and a fixture that was quietly re-recorded, both surface
as a `REGRESSED` job and a non-zero exit.

The suite also runs under AddressSanitizer and UndefinedBehaviorSanitizer, with
leak detection on — the journal and the `stop_source` plumbing hand ownership
around enough that a leak is a real failure mode, not a hypothetical one:

```bash
cmake --preset fedora-clang-asan
cmake --build --preset fedora-clang-asan
ctest --preset fedora-clang-asan
```

## Design

**Interception sits at the seam, not the wire.** The journal records
`ModelProvider::chat` and `Tool::invoke` — the two places where a run touches
something it cannot recompute. Recording at the HTTP byte layer would make the
journal a transcript of one transport, unreadable and brittle; recording at the
seam keeps it a transcript of the *run*, so it survives a provider change and
can be diffed by hand.

**Recording is a decorator, not a mode.** `RecordingProvider` and
`ReplayingProvider` wrap the real provider; `RecordingTool` and `ReplayingTool`
wrap the real tools. The agent loop has no idea which mode it is in, which is
why a recorded run and a replayed run take the same code path — and why the
replay is a real test of the loop rather than a re-enactment of a log.

**Determinism is per actor, not per thread.** Chasing bit-identical thread
interleaving is a research project. Instead each logical task gets an `actor_id`
and its own monotonic sequence, and replay orders by `(actor, seq)`. The order
in which concurrent tasks happened to interleave is simply not part of the
format, so it never has to be reproduced. Journals are already keyed this way,
so concurrency needs no format change later.

**A batch may be wider than the pool.** `when_all` starts each task on a worker
and the task gives that worker back the moment it suspends, so a hundred calls
over twenty threads make progress twenty at a time rather than deadlocking at
twenty. The same reasoning runs through `LimitedProvider`: it caps how many
requests are in flight against one endpoint, and its `AsyncSemaphore` parks the
*coroutine* rather than the thread. A `std::counting_semaphore` acquired inside
a coroutine holds the very thread that would have to release it, which deadlocks
under exactly the load the limit was added to survive. A test drives three calls
through one permit on a single worker — that wedges the blocking version and
passes for this one, so the test cannot pass by accident.

**Credentials cannot reach the journal.** The API key lives in `ProviderConfig`,
which is not part of any request. A test walks every field written to the
journal and asserts none of them is credential-shaped; another greps the file
for a known key. This is a property of the design, not a redaction step that
could be forgotten.

**The core stays embeddable.** `include/ash` may not contain `pybind11`, CLI
code, eval types, or anything that prints to the console — it returns data and
lets a consumer decide what to do with it. The eval harness therefore lives in
a top-level `eval/` and consumes the runtime like any other program, rather
than forcing every embedder to compile a suite parser and a price table. A test
enforces the boundary, and the test is checked against a deliberate violation
so it cannot pass vacuously.

## Status

Early. The vertical slice works end to end; the interesting parts are ahead.

Done:

- `Task<T>` coroutines, a bounded thread pool, `stop_token` cancellation plumbing
- Two provider adapters: the OpenAI `/chat/completions` dialect (DeepSeek,
  Moonshot, Qwen, and most others) and the Anthropic Messages dialect
- A libcurl HTTP client with connection sharing
- Three filesystem tools and a coroutine agent loop with a step budget
- An append-only journal, recording and replaying decorators, and `ash replay`
- Two real recordings committed under `examples/journals/`, replayed by the suite
- An eval harness over JSON suites: per-job checks, pass/fail, token and cost
  totals, and latency percentiles, behind `ash eval`. It replays committed
  journals, so it runs offline and costs nothing
- Report files and `--baseline`, which diffs a run against an earlier one and
  exits non-zero on a regression — including a job that was quietly deleted.
  CI grades every push against a committed baseline and archives the table
- A price table, and per-call timings written into the journal, so a replayed
  run still reports the cost and latency the original run really had
- `when_all` and `Nursery`: a batch can be wider than the pool, a failure is
  reported in task order rather than in whatever order the scheduler picked, and
  the first failure asks its siblings to stop
- `--jobs N` on the eval, replaying a suite across workers while producing a
  byte-identical report
- `LimitedProvider` and `AsyncSemaphore`, which cap in-flight requests per
  endpoint by parking the coroutine and not the thread. Tested, but nothing in
  the CLI fans out yet, so today it is a component waiting for its caller
- 82 tests, both GCC and Clang, `-Werror`, zero warnings, and clean under
  ASan + UBSan

Next:

- A concurrent runner that fans a task out across actors, which is what
  `LimitedProvider` is waiting for
- SSE streaming with a bounded channel and backpressure
- Python bindings via pybind11
- Structured traces, a viewer, and per-provider cost accounting
- Threading `stop_token` into libcurl's progress callback

## Layout

```
include/ash/     the public surface, and only that
src/             core/ model/ tool/ record/ io/
eval/            the eval harness -- a consumer of the runtime, never part of it
apps/cli/        the ash command
tests/           Catch2 suite
tools/           the offline stub server and the replay determinism check
```

## Reading

The design here owes a lot to [openai/codex](https://github.com/openai/codex),
[block/goose](https://github.com/block/goose), Aider, OpenHands, and Anthropic's
engineering writing on agents, tools, and context. This is an independent
implementation; none of their code is used.

## License

Apache-2.0. The patent grant is the reason: this is infrastructure that a
company might want to depend on, and the explicit grant removes a question
their legal review would otherwise have to ask.
