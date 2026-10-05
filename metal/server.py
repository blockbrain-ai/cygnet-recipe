#!/usr/bin/env python3
"""Local OpenAI-compatible transport for Cygnet's unchanged shim on Apple Metal.

This is deliberately a small API profile, not a general chat server. It accepts
exactly the request produced by cygnet_shim.call_vllm. The native worker reads
masked first-answer logits without generating a token. The compatibility
message contains the highest-probability token's text; completion_tokens is 0.
Calibration and duplicate-letter aggregation remain in the upstream shim.

    python3 metal/server.py --model .models/12b/gemma-4-12b-it-qat-q4_0.gguf
    SHIM_VLLM=http://127.0.0.1:8890/v1/chat/completions \\
        SHIM_TEMPERATURE=3.4 python3 shim/cygnet_shim.py
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import queue
import signal
import select
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
import uuid

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "cygnet"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MAX_BODY = 1024 * 1024
MAX_WORKER_LINE = 2 * 1024 * 1024
MAX_ERROR = 1000


class InputError(ValueError):
    """An input outside the deliberately limited API profile."""


class WorkerError(RuntimeError):
    """The native worker is unavailable or violated its protocol."""


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON keys are not accepted")
        result[key] = value
    return result


def _constant(value):
    raise ValueError("non-finite JSON numbers are not accepted")


def _float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite JSON numbers are not accepted")
    return number


def read_json(data):
    return json.loads(data, object_pairs_hook=_pairs, parse_constant=_constant, parse_float=_float)


def parse_request(body):
    """Validate, then translate the upstream shim's exact OpenAI request."""
    required = {
        "model", "messages", "max_tokens", "temperature", "logprobs",
        "top_logprobs", "chat_template_kwargs", "structured_outputs",
    }
    if not isinstance(body, dict) or set(body) != required:
        raise InputError("expected exactly the fields used by Cygnet's call_vllm")
    if body["model"] != MODEL_NAME:
        raise InputError("model must be 'cygnet'")
    for key, expected in (("max_tokens", 1), ("top_logprobs", 20)):
        if type(body[key]) is not int or body[key] != expected:
            raise InputError(f"{key} must be {expected}")
    if type(body["temperature"]) not in (int, float) or body["temperature"] != 1:
        raise InputError("temperature must be 1; calibration belongs in the Cygnet shim")
    if body["logprobs"] is not True:
        raise InputError("logprobs must be true")
    if body["chat_template_kwargs"] != {"enable_thinking": False} or (
        body["chat_template_kwargs"].get("enable_thinking") is not False
    ):
        raise InputError("chat_template_kwargs must be {'enable_thinking': false}")
    messages = body["messages"]
    if not isinstance(messages, list) or len(messages) != 2:
        raise InputError("messages must contain exactly one system and one user text message")
    for message, role in zip(messages, ("system", "user")):
        if not isinstance(message, dict) or set(message) != {"role", "content"} or message["role"] != role:
            raise InputError("messages must contain exactly one system and one user text message")
        content = message["content"]
        if not isinstance(content, str) or not content.strip() or "\0" in content:
            raise InputError("message content must be non-empty text without NUL characters")
        try:
            content.encode("utf-8")
        except UnicodeError as exc:
            raise InputError("message content must be valid UTF-8") from exc
    structured = body["structured_outputs"]
    if not isinstance(structured, dict) or set(structured) != {"choice"}:
        raise InputError("structured_outputs must contain only choice")
    letters = structured["choice"]
    if not isinstance(letters, list) or not 1 <= len(letters) <= len(LETTERS) or letters != list(LETTERS[:len(letters)]):
        raise InputError("choice must contain consecutive option letters starting at A (1–26 options)")
    return {
        "system": messages[0]["content"], "prompt": messages[1]["content"],
        "letters": letters, "top_logprobs": 20,
    }


class NativeWorker:
    """One owned process, protected from concurrent loading and serially queried."""

    def __init__(self, command, *, context_size, lock_path, startup_timeout=180.0, request_timeout=180.0):
        self.context_size = context_size
        self.request_timeout = request_timeout
        self._serial = threading.Lock()
        self._close_lock = threading.Lock()
        self._closed = False
        self._process = None
        self._lock_file = None
        self._reader = None
        self._replies = queue.Queue(maxsize=2)
        self.ready = None
        try:
            lock_path = Path(lock_path)
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            self._lock_file = lock_path.open("a+")
            try:
                fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise WorkerError(f"another Metal worker owns {lock_path}") from exc
            self._lock_file.seek(0)
            self._lock_file.truncate()
            self._lock_file.write(f"{os.getpid()}\n")
            self._lock_file.flush()
            # Stderr is inherited, never piped into an unread buffer. Stdout is protocol only.
            self._process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=None, start_new_session=True,
                # Keep the model lease alive in the child if this Python process
                # is killed during inference. Otherwise a replacement could load
                # another model while the orphan still occupies unified memory.
                pass_fds=(self._lock_file.fileno(),),
            )
            os.set_blocking(self._process.stdin.fileno(), False)
            self._reader = threading.Thread(target=self._read_output, name="metal-worker-output", daemon=True)
            self._reader.start()
            ready = self._receive(startup_timeout)
            if isinstance(ready, dict) and ready.get("ready") is False and "error" in ready:
                raise WorkerError(f"native worker startup failed: {str(ready['error'])[:MAX_ERROR]}")
            if not isinstance(ready, dict) or ready.get("ready") is not True or type(ready.get("context_size")) is not int:
                raise WorkerError("native worker did not report readiness and context_size")
            if ready["context_size"] != context_size:
                raise WorkerError("native worker reported a different context_size")
            self.ready = ready
        except BaseException:
            self.close()
            raise

    def _read_output(self):
        try:
            while not self._closed:
                line = self._process.stdout.readline(MAX_WORKER_LINE + 1)
                if not line:
                    item = WorkerError("native worker exited before responding")
                elif len(line) > MAX_WORKER_LINE or not line.endswith(b"\n"):
                    item = WorkerError("native worker returned an oversized or incomplete protocol line")
                else:
                    try:
                        item = read_json(line)
                    except (ValueError, UnicodeError, RecursionError):
                        item = WorkerError("native worker returned invalid JSON")
                try:
                    self._replies.put_nowait(item)
                except queue.Full:
                    return  # Extra unsolicited messages fail the next request's ID check.
                if isinstance(item, Exception):
                    return
        except (OSError, ValueError):
            try:
                self._replies.put_nowait(WorkerError("native worker output closed"))
            except queue.Full:
                pass

    def _receive(self, timeout):
        try:
            reply = self._replies.get(timeout=timeout)
        except queue.Empty as exc:
            raise WorkerError("native worker timed out; restart the Metal server") from exc
        if isinstance(reply, Exception):
            raise reply
        return reply

    def _write(self, wire, deadline):
        # A stuck child must not trap the HTTP thread on a full stdin pipe.
        pending = memoryview(wire)
        fd = self._process.stdin.fileno()
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([], [fd], [], remaining)[1]:
                raise WorkerError("native worker timed out reading its request; restart the Metal server")
            try:
                written = os.write(fd, pending)
            except BlockingIOError:
                continue
            pending = pending[written:]

    def is_alive(self):
        return not self._closed and self._process is not None and self._process.poll() is None

    def infer(self, request):
        # Queueing counts against the deadline, so concurrency cannot create unbounded waits.
        deadline = time.monotonic() + self.request_timeout
        if not self._serial.acquire(timeout=self.request_timeout):
            raise WorkerError("Metal worker is busy; request timed out waiting for the model")
        try:
            if not self.is_alive():
                raise WorkerError("native worker is not running; restart the Metal server")
            request_id = uuid.uuid4().hex
            message = dict(request, id=request_id)
            wire = (json.dumps(message, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
            if len(wire) > MAX_WORKER_LINE:
                raise InputError("request exceeds the native worker size limit")
            try:
                self._write(wire, deadline)
                reply = self._receive(max(0.001, deadline - time.monotonic()))
                if not isinstance(reply, dict) or reply.get("id") != request_id:
                    raise WorkerError("native worker response ID did not match the request")
                if "error" in reply:
                    error = str(reply["error"])[:MAX_ERROR]
                    if reply.get("error_type") == "input":
                        raise InputError(error)
                    raise WorkerError(f"native inference failed: {error}")
                self._validate_reply(reply)
                return reply
            except InputError:
                raise
            except (OSError, WorkerError, ValueError) as exc:
                self.close()
                if isinstance(exc, WorkerError):
                    raise
                raise WorkerError("native worker communication failed; restart the Metal server") from exc
        finally:
            self._serial.release()

    def _validate_reply(self, reply):
        if reply.get("ok") is not True or type(reply.get("prompt_tokens")) is not int or not 0 < reply["prompt_tokens"] <= self.context_size:
            raise WorkerError("native worker returned invalid token usage")
        if reply.get("completion_tokens", 0) != 0:
            raise WorkerError("native worker unexpectedly generated output tokens")
        top = reply.get("top_logprobs")
        if not isinstance(top, list) or not 1 <= len(top) <= 20:
            raise WorkerError("native worker returned an invalid top_logprobs list")
        token_ids = set()
        for item in top:
            if not isinstance(item, dict):
                raise WorkerError("native worker returned an invalid logprob record")
            token, token_id, logprob = item.get("token"), item.get("token_id"), item.get("logprob")
            if not isinstance(token, str) or not token or len(token) > 256 or type(token_id) is not int or token_id < 0 or token_id in token_ids:
                raise WorkerError("native worker returned an invalid or repeated token ID")
            if type(logprob) not in (int, float) or not math.isfinite(logprob) or logprob > 1e-6:
                raise WorkerError("native worker returned an invalid log probability")
            try:
                token.encode("utf-8")
            except UnicodeError as exc:
                raise WorkerError("native worker returned invalid UTF-8 token text") from exc
            token_ids.add(token_id)

    def close(self):
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            process = self._process
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                for stream in (process.stdin, process.stdout):
                    if stream:
                        stream.close()
            if self._reader and self._reader is not threading.current_thread():
                self._reader.join(timeout=1)
            if self._lock_file:
                self._lock_file.close()  # Keep the inode: unlinking creates a lock race.
                self._lock_file = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def completion(reply):
    top = reply["top_logprobs"]
    best = max(top, key=lambda item: item["logprob"])
    return {
        "id": "chatcmpl-metal-" + reply["id"], "object": "chat.completion",
        "created": int(time.time()), "model": MODEL_NAME,
        "choices": [{
            "index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": best["token"]},
            "logprobs": {"content": [{
                "token": best["token"], "logprob": best["logprob"],
                "bytes": list(best["token"].encode("utf-8")),
                "top_logprobs": top,  # Never collapse duplicate decoded strings into a dict.
            }]},
        }],
        "usage": {"prompt_tokens": reply["prompt_tokens"], "completion_tokens": 0, "total_tokens": reply["prompt_tokens"]},
        "metal": {"readout": "prefill_logits", "kv_cache": reply.get("kv_cache"), "timings_ms": reply.get("timings_ms")},
    }


class MetalServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, worker):
        if address[0] != "127.0.0.1":
            raise ValueError("this local adapter binds only to 127.0.0.1")
        if not worker.is_alive():
            raise WorkerError("load the native worker before starting HTTP")
        self.worker = worker
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    server_version = "CygnetMetal/1"
    sys_version = ""

    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, *_):
        pass

    def _send(self, status, body):
        data = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.close_connection = True
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass

    def _error(self, status, message):
        self._send(status, {"error": {"message": str(message)[:MAX_ERROR], "type": "invalid_request_error" if status < 500 else "server_error"}})

    def _local_request(self):
        port = self.server.server_address[1]
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if port == 80:
            hosts.update(("127.0.0.1", "localhost"))
        if len(self.headers.get_all("Host", [])) != 1 or self.headers.get("Host") not in hosts:
            self._error(403, "Host must refer to this local server")
            return False
        origin = self.headers.get("Origin")
        if origin is not None:
            try:
                parsed = urlsplit(origin)
                valid = parsed.scheme == "http" and parsed.netloc in hosts and not parsed.path and not parsed.query and not parsed.fragment
            except ValueError:
                valid = False
            if not valid or len(self.headers.get_all("Origin", [])) != 1:
                self._error(403, "cross-origin requests are not allowed")
                return False
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            self._error(403, "cross-site requests are not allowed")
            return False
        return True

    def do_GET(self):
        if not self._local_request():
            return
        if self.path != "/v1/models":
            self._error(404, "endpoint not found")
        elif not self.server.worker.is_alive():
            self._error(503, "native worker is not running; restart the Metal server")
        else:
            self._send(200, {"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local", "max_model_len": self.server.worker.context_size}]})

    def do_POST(self):
        if not self._local_request():
            return
        if self.path != "/v1/chat/completions":
            self._error(404, "endpoint not found")
            return
        if self.headers.get("Transfer-Encoding") or self.headers.get("Content-Encoding"):
            self._error(400, "transfer and content encodings are not supported")
            return
        lengths = self.headers.get_all("Content-Length", [])
        if not lengths:
            self._error(411, "Content-Length is required")
            return
        try:
            if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit():
                raise ValueError
            length = int(lengths[0])
        except ValueError:
            self._error(400, "invalid Content-Length")
            return
        if length > MAX_BODY:
            self._error(413, "request exceeds the 1 MiB size limit")
            return
        if self.headers.get_content_type() != "application/json":
            self._error(415, "Content-Type must be application/json")
            return
        try:
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise InputError("incomplete request body")
            try:
                body = read_json(raw.decode("utf-8"))
            except (ValueError, UnicodeError, RecursionError) as exc:
                raise InputError("request must be valid UTF-8 JSON without duplicate keys or non-finite numbers") from exc
            reply = self.server.worker.infer(parse_request(body))
            self._send(200, completion(reply))
        except (InputError, TimeoutError, ConnectionResetError) as exc:
            self._error(400, exc)
        except WorkerError as exc:
            self._error(503, exc)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="local Gemma 4 12B IT QAT Q4_0 GGUF")
    parser.add_argument("--worker", type=Path, default=ROOT / ".runtime/build/bin/cygnet-metal-worker")
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--startup-timeout", type=float, default=180)
    parser.add_argument("--request-timeout", type=float, default=180)
    args = parser.parse_args(argv)
    if not 512 <= args.ctx_size <= 16384 or args.ctx_size % 256 or not 1 <= args.threads <= 64 or not 1 <= args.port <= 65535:
        parser.error("ctx-size must be a multiple of 256 in 512–16384, threads 1–64, and port 1–65535")
    if not all(math.isfinite(t) and t > 0 for t in (args.startup_timeout, args.request_timeout)):
        parser.error("timeouts must be finite and positive")
    if not args.model.is_file() or not args.worker.is_file():
        parser.error("model and worker must exist; see metal/README.md for setup")
    command = [str(args.worker.resolve()), "--model", str(args.model.resolve()), "--ctx-size", str(args.ctx_size), "--threads", str(args.threads)]

    def stop(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        with NativeWorker(command, context_size=args.ctx_size, lock_path=ROOT / ".runtime/cygnet-metal.lock",
                          startup_timeout=args.startup_timeout, request_timeout=args.request_timeout) as worker:
            with MetalServer(("127.0.0.1", args.port), worker) as server:
                print(f"Cygnet Metal ready: http://127.0.0.1:{server.server_port} (context {worker.context_size}, no token generation)", flush=True)
                server.serve_forever()
    except KeyboardInterrupt:
        return 0
    except (OSError, WorkerError) as exc:
        print(f"Cygnet Metal: {str(exc)[:MAX_ERROR]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
