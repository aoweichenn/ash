#!/usr/bin/env python3
"""A scripted model server for offline development and CI.

Serves a scripted sequence of responses so the full agent path -- CLI,
provider, HTTP, JSON codec, tool loop -- can be exercised with no API key and
no network. Responses are written once in a dialect-neutral form and rendered
into either wire format, which also makes this a readable record of how the
two provider dialects differ.

    tools/stub_model.py --port 8099 --script tools/scripts/demo.json
    tools/stub_model.py --dialect anthropic --script tools/scripts/demo.json

`--record` writes every request to one file when the server shuts down.
`--dump-requests DIR` writes each one to its own file as it arrives, which is
what a check needs when the request it cares about is the third of six and the
session is still running: the later turns are the evidence, and waiting for the
end to look at them is not possible from inside the run being checked.

A request with `"stream": true` is answered as Server-Sent Events instead, in
the frame sequence the matching real endpoint uses. Text and tool arguments are
cut into small pieces rather than sent whole, because a stub that answered a
streaming request with one frame would exercise the client's framing code not at
all -- the whole difficulty is that a read can land anywhere.

Two details are deliberately faithful rather than convenient:
  * the OpenAI dialect sends its usage frame only when the request asked for it
    with `stream_options.include_usage`, exactly as the real endpoint does, so a
    client that forgets to ask records zero tokens instead of silently passing;
  * the Anthropic dialect numbers content blocks as they open, so a tool call
    that follows text is block 1 and not block 0.

Script format (a JSON list; the last entry repeats once exhausted):

    [{"content": "final answer",
      "tool_calls": [{"id": "call_1", "name": "list_dir",
                      "arguments": {"path": "."}}],
      "usage": {"prompt": 120, "completion": 18}}]
"""

import argparse
import json
import os
import signal
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer


def dump_request(directory, number, request):
    """One file per request, named by the order it arrived in."""
    path = os.path.join(directory, f"request-{number:04d}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(request, handle, indent=2)


def render_openai(entry, model, index):
    calls = entry.get("tool_calls") or []
    message = {"role": "assistant", "content": entry.get("content") or None}
    if calls:
        message["tool_calls"] = [
            {
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["name"],
                    # OpenAI-compatible endpoints carry arguments as a string.
                    "arguments": json.dumps(call.get("arguments", {})),
                },
            }
            for call in calls
        ]

    usage = entry.get("usage", {})
    prompt = usage.get("prompt", 0)
    completion = usage.get("completion", 0)
    return {
        "id": f"chatcmpl-stub-{index}",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": entry.get("finish_reason", "stop"),
            }
        ],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    }


def render_anthropic(entry, model, index):
    blocks = []
    if entry.get("content"):
        blocks.append({"type": "text", "text": entry["content"]})
    for call in entry.get("tool_calls") or []:
        blocks.append(
            {
                "type": "tool_use",
                "id": call["id"],
                "name": call["name"],
                # Anthropic carries arguments as a real JSON object.
                "input": call.get("arguments", {}),
            }
        )

    usage = entry.get("usage", {})
    return {
        "id": f"msg_stub_{index}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks,
        "stop_reason": "tool_use" if entry.get("tool_calls") else "end_turn",
        "usage": {
            "input_tokens": usage.get("prompt", 0),
            "output_tokens": usage.get("completion", 0),
        },
    }


RENDERERS = {"openai": render_openai, "anthropic": render_anthropic}


def pieces(text, size):
    """Cut text into fixed-size pieces.

    Small pieces on purpose: an argument JSON split mid-key is the case the
    client's stitching has to survive, and one big piece would never produce it.
    """
    return [text[at : at + size] for at in range(0, len(text), size)]


def frame(name, payload):
    """One SSE frame, as bytes.

    `payload` is a dict to be encoded, or a string to be sent verbatim -- the
    OpenAI sentinel is not JSON and must not be quoted into looking like it.
    """
    body = payload if isinstance(payload, str) else json.dumps(payload)
    head = f"event: {name}\n" if name else ""
    return f"{head}data: {body}\n\n".encode("utf-8")


def stream_openai(entry, model, index, include_usage):
    ident = f"chatcmpl-stub-{index}"
    calls = entry.get("tool_calls") or []

    def chunk(delta, finish_reason=None):
        return {
            "id": ident,
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    yield frame(None, chunk({"role": "assistant", "content": ""}))
    for piece in pieces(entry.get("content") or "", 3):
        yield frame(None, chunk({"content": piece}))

    for position, call in enumerate(calls):
        # The id and the name come first and the arguments follow, because the
        # endpoint has to open the call before it can stream into it.
        yield frame(
            None,
            chunk(
                {
                    "tool_calls": [
                        {
                            "index": position,
                            "id": call["id"],
                            "type": "function",
                            "function": {"name": call["name"], "arguments": ""},
                        }
                    ]
                }
            ),
        )
        for piece in pieces(json.dumps(call.get("arguments", {})), 7):
            yield frame(
                None,
                chunk({"tool_calls": [{"index": position, "function": {"arguments": piece}}]}),
            )

    yield frame(None, chunk({}, entry.get("finish_reason") or ("tool_calls" if calls else "stop")))

    if include_usage:
        usage = entry.get("usage", {})
        prompt = usage.get("prompt", 0)
        completion = usage.get("completion", 0)
        # Deliberately its own frame with no choices, which is where the real
        # endpoint puts it and why the count is read apart from the text.
        yield frame(
            None,
            {
                "id": ident,
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                    "total_tokens": prompt + completion,
                },
            },
        )

    yield frame(None, "[DONE]")


def stream_anthropic(entry, model, index, include_usage):  # noqa: ARG001 - parity with the other dialect
    calls = entry.get("tool_calls") or []
    usage = entry.get("usage", {})

    yield frame(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": f"msg_stub_{index}",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "usage": {"input_tokens": usage.get("prompt", 0), "output_tokens": 0},
            },
        },
    )

    block = 0
    if entry.get("content"):
        yield frame(
            "content_block_start",
            {"type": "content_block_start", "index": block,
             "content_block": {"type": "text", "text": ""}},
        )
        for piece in pieces(entry["content"], 3):
            yield frame(
                "content_block_delta",
                {"type": "content_block_delta", "index": block,
                 "delta": {"type": "text_delta", "text": piece}},
            )
        yield frame("content_block_stop", {"type": "content_block_stop", "index": block})
        block += 1

    for call in calls:
        yield frame(
            "content_block_start",
            {"type": "content_block_start", "index": block,
             "content_block": {"type": "tool_use", "id": call["id"],
                               "name": call["name"], "input": {}}},
        )
        for piece in pieces(json.dumps(call.get("arguments", {})), 7):
            yield frame(
                "content_block_delta",
                {"type": "content_block_delta", "index": block,
                 "delta": {"type": "input_json_delta", "partial_json": piece}},
            )
        yield frame("content_block_stop", {"type": "content_block_stop", "index": block})
        block += 1

    # The prompt count was reported at the start and the completion count is
    # reported here, which is the whole reason a usage update is partial.
    yield frame(
        "message_delta",
        {"type": "message_delta",
         "delta": {"stop_reason": "tool_use" if calls else "end_turn"},
         "usage": {"output_tokens": usage.get("completion", 0)}},
    )
    yield frame("message_stop", {"type": "message_stop"})


STREAMERS = {"openai": stream_openai, "anthropic": stream_anthropic}


class StubHandler(BaseHTTPRequestHandler):
    script = []
    dialect = "openai"
    requests = []
    calls = 0
    dump_dir = ""

    # Keep-alive, so the streaming path can be written as chunked and the
    # non-streaming path stays exactly as it was.
    protocol_version = "HTTP/1.1"

    def write_json(self, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def write_event_stream(self, frames):
        # Chunked, and flushed per frame, so the client really does receive the
        # response in pieces -- one that arrived as a single write would let a
        # broken incremental parser pass every test this server can run.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for chunk in frames:
            self.wfile.write(b"%X\r\n" % len(chunk) + chunk + b"\r\n")
            self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def do_POST(self):  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            request = json.loads(raw or b"{}")
        except json.JSONDecodeError as error:
            self.send_error(400, f"invalid JSON: {error}")
            return

        StubHandler.requests.append(request)
        index = min(StubHandler.calls, len(self.script) - 1)
        StubHandler.calls += 1
        if StubHandler.dump_dir:
            dump_request(StubHandler.dump_dir, StubHandler.calls, request)

        entry = StubHandler.script[index]
        model = request.get("model", "stub")

        if request.get("stream"):
            include_usage = bool(request.get("stream_options", {}).get("include_usage"))
            self.write_event_stream(
                STREAMERS[StubHandler.dialect](entry, model, StubHandler.calls, include_usage)
            )
            return

        self.write_json(RENDERERS[StubHandler.dialect](entry, model, StubHandler.calls))

    def log_message(self, *args):
        pass  # keep stdout clean for the driver script


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--script", required=True, help="JSON file with a list of responses")
    parser.add_argument("--dialect", choices=sorted(RENDERERS), default="openai")
    parser.add_argument("--record", help="write every received request here on shutdown")
    parser.add_argument("--dump-requests", metavar="DIR",
                        help="write each received request to its own file in DIR, as it arrives")
    args = parser.parse_args()

    with open(args.script, encoding="utf-8") as handle:
        StubHandler.script = json.load(handle)
    if not StubHandler.script:
        parser.error(f"script {args.script} is empty")
    StubHandler.dialect = args.dialect
    if args.dump_requests:
        os.makedirs(args.dump_requests, exist_ok=True)
        StubHandler.dump_dir = args.dump_requests

    # Translate SIGTERM into the same path as Ctrl-C so --record always flushes.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    server = HTTPServer(("127.0.0.1", args.port), StubHandler)
    print(
        f"stub_model: {len(StubHandler.script)} responses, {args.dialect} dialect, port {args.port}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if args.record:
            with open(args.record, "w", encoding="utf-8") as handle:
                json.dump(StubHandler.requests, handle, indent=2)
            print(f"stub_model: wrote {len(StubHandler.requests)} requests", file=sys.stderr)


if __name__ == "__main__":
    main()
