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
import threading

RELAX_PATH = "/api/relax"
STATE_KEY = "client_disconnected"


class RelaxDisconnectMiddleware:
    def __init__(self, app, path: str = RELAX_PATH):
        self.app = app
        self.path = path

    async def __call__(self, scope, receive, send):
        if (scope["type"] != "http" or scope.get("method") != "POST"
                or scope.get("path") != self.path):
            await self.app(scope, receive, send)
            return

        # Buffer the whole body (same as FastAPI's own JSON parsing would).
        body = bytearray()
        more = True
        while more:
            message = await receive()
            if message["type"] == "http.disconnect":
                # Client left before sending the body; nothing to serve.
                return
            body += message.get("body", b"")
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
