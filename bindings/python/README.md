# ash Python bindings

The runtime, from Python. For what the bindings do and why, see the Python
section of the top-level `README.md`; this file is the development loop.

## Building

```bash
cmake --preset fedora-clang-python
cmake --build --preset fedora-clang-python
```

`ASH_BUILD_PYTHON` is off by default, and the pybind11 fetch is behind it, so a
checkout that never sets it does not need Python headers. The preset turns it
on, and also builds the CLI — one test compares a report the bindings produced
against the report the CLI writes.

The module lands in `build/fedora-clang-python/bindings/python/ash/`, next to
the `__init__.py` copied there from `ash/__init__.py`. That directory is the
package; its parent is what goes on `PYTHONPATH`:

```bash
PYTHONPATH=build/fedora-clang-python/bindings/python python3 -c "
import ash
print(ash.__version__)
print(ash.replay('examples/journals/openai.jsonl').tools_called)"
```

## Which interpreter

CMake finds one interpreter and builds the module for it, and a module built for
one Python does not import into another — the failure reads like a missing file,
which is why it is worth being deliberate about. The interpreter that was chosen
is printed at configure time and recorded in
`build/fedora-clang-python/bindings/python/python-interpreter.txt`;
`tools/verify_python.sh` reads that file rather than trusting `python3`, and
takes a second argument to override it.

To point the build at a different one:

```bash
cmake --preset fedora-clang-python -DPython_EXECUTABLE=/usr/bin/python3
```

Nothing in this repository hardcodes a `.so` suffix or a personal interpreter
path: the suffix comes from `pybind11_add_module`, and the interpreter path
comes from whichever one CMake found on the machine doing the build.

## Running the tests

The suite is a plain pytest suite with no fixtures outside `conftest.py`, and it
starts the stub model server itself:

```bash
PYTHONPATH=build/fedora-clang-python/bindings/python \
ASH_CLI=build/fedora-clang-python/apps/cli/ash \
    python3 -m pytest bindings/python/tests -q
```

`ASH_CLI` is only read by the one test that compares the bindings' eval report
against the CLI's; unset, it falls back to the default preset's build directory.
`ASH_REPO` overrides where the suite looks for the committed journals and the
stub's script, and is only needed if you run the suite from outside the
repository.

It also runs under `ctest`, which is what CI uses:

```bash
ctest --preset fedora-clang-python -R python_bindings --output-on-failure
```

`tools/verify_python.sh` is the end-to-end gate — it records a run from Python,
shuts the model server down, replays it, and checks the claims the bindings are
for. Run it before pushing anything that touches this directory.

## How it is put together

`include/ash` is the public surface and may not contain pybind11; everything
here is outside it, and a test in `tests/boundary_test.cpp` fails the build if
that changes. What the bindings may use is therefore exactly what any other
embedder gets.

- `convert.*` — the one place Python objects and the runtime's JSON meet. Types
  that have their own `to_json` go through it, so what Python sees is what a
  journal would contain.
- `types.*` — `Usage`, `Result`, the `Cancelled` exception, and the guard that
  makes dropping a Python reference safe while the interpreter is shutting down.
- `provider.*`, `tools.*` — the provider factories, `@ash.tool`, `ToolSet`, and
  the schema derivation.
- `stream.*` — `CancelToken` and the sink that forwards events to a Python
  callback.
- `run.*` — `Agent`, `run()`, and `replay()`.
- `eval.*` — the eval harness.

Two rules that are easy to break and hard to debug, both enforced by comments at
the place they matter. A blocking call into the runtime is made with the GIL
released, and every callback re-acquires it — so a callback runs on the thread
inside libcurl's write callback and must not block. And nothing holding a Python
reference may cross a boundary where the GIL has been released, because the
deleter needs it.
