"""Pure-ASGI client-disconnect detector for POST /api/relax.

Why this exists: ``Request.is_disconnected()`` never reports True inside
handlers of an app that uses ``@app.middleware("http")`` (BaseHTTPMiddleware):
the call awaits ``receive()`` inside an already-cancelled anyio scope. This
wrapper sits outermost, buffers the (small JSON) request body, and watches the
real ``receive`` channel in a background task. On ``http.disconnect`` it sets a
per-request ``threading.Event`` stored at ``scope["state"]["client_disconnected"]``
so the (worker-thread) relaxation can stop cooperatively.
"""
import asyncio
import json
import os
import threading

RELAX_PATH = "/api/relax"
STATE_KEY = "client_disconnected"
# The relax request is a few hundred bytes of JSON; this wrapper buffers it
# before any rate limiting, so it must bound what it accepts.
MAX_RELAX_BODY_BYTES = int(os.getenv("MAX_RELAX_BODY_BYTES", str(1024 * 1024)))


async def _send_413(send, limit):
    payload = json.dumps(
        {"detail": f"Request body too large. Maximum size: {limit} bytes"}).encode()
    await send({"type": "http.response.start", "status": 413,
                "headers": [(b"content-type", b"application/json"),
                            (b"content-length", str(len(payload)).encode()),
                            (b"connection", b"close")]})
    await send({"type": "http.response.body", "body": payload})


class RelaxDisconnectMiddleware:
    def __init__(self, app, path: str = RELAX_PATH, max_body: int = None):
        self.app = app
        self.path = path
        self.max_body = MAX_RELAX_BODY_BYTES if max_body is None else max_body

    def _matches(self, scope) -> bool:
        if scope["type"] != "http" or scope.get("method") != "POST":
            return False
        path = scope.get("path", "")
        root = scope.get("root_path", "") or ""
        # Behind a proxy mount, path may or may not include root_path.
        if root and path.startswith(root + "/"):
            path = path[len(root):]
        return path == self.path

    async def __call__(self, scope, receive, send):
        if not self._matches(scope):
            await self.app(scope, receive, send)
            return

        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    break
                if declared > self.max_body:
                    await _send_413(send, self.max_body)
                    return
                break

        # Buffer the body, bounded (chunked bodies have no Content-Length).
        body = bytearray()
        more = True
        while more:
            message = await receive()
            if message["type"] == "http.disconnect":
                # Client left before sending the body; nothing to serve.
                return
            body += message.get("body", b"")
            if len(body) > self.max_body:
                await _send_413(send, self.max_body)
                return
            more = message.get("more_body", False)

        event = threading.Event()
        scope.setdefault("state", {})[STATE_KEY] = event
        disconnected = asyncio.get_running_loop().create_future()
        replayed = False

        async def inner_receive():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await asyncio.shield(disconnected)

        async def watch():
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    event.set()
                    if not disconnected.done():
                        disconnected.set_result(message)
                    return

        task = asyncio.create_task(watch())
        try:
            await self.app(scope, inner_receive, send)
        finally:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
