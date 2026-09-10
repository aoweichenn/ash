#!/usr/bin/env python3
"""A scripted model server for offline development and CI.

Serves a scripted sequence of responses so the full agent path -- CLI,
provider, HTTP, JSON codec, tool loop -- can be exercised with no API key and
no network. Responses are written once in a dialect-neutral form and rendered
into either wire format, which also makes this a readable record of how the
two provider dialects differ.

    tools/stub_model.py --port 8099 --script tools/scripts/demo.json
    tools/stub_model.py --dialect anthropic --script tools/scripts/demo.json

Script format (a JSON list; the last entry repeats once exhausted):

    [{"content": "final answer",
      "tool_calls": [{"id": "call_1", "name": "list_dir",
                      "arguments": {"path": "."}}],
      "usage": {"prompt": 120, "completion": 18}}]
"""

import argparse
import json
import signal
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer


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


class StubHandler(BaseHTTPRequestHandler):
    script = []
    dialect = "openai"
    requests = []
    calls = 0

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

        payload = RENDERERS[StubHandler.dialect](
            StubHandler.script[index], request.get("model", "stub"), StubHandler.calls
        )
        body = json.dumps(payload).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # keep stdout clean for the driver script


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--script", required=True, help="JSON file with a list of responses")
    parser.add_argument("--dialect", choices=sorted(RENDERERS), default="openai")
    parser.add_argument("--record", help="write every received request here on shutdown")
    args = parser.parse_args()

    with open(args.script, encoding="utf-8") as handle:
        StubHandler.script = json.load(handle)
    if not StubHandler.script:
        parser.error(f"script {args.script} is empty")
    StubHandler.dialect = args.dialect

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
