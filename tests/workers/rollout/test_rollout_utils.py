import asyncio

import pytest
from fastapi import FastAPI

from verl.workers.rollout.utils import run_unvicorn


@pytest.mark.asyncio
async def test_run_unvicorn_starts_listening_server():
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"ok": True}

    port, server_task = await run_unvicorn(app, server_args=None, server_address="127.0.0.1")
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        await writer.drain()
        response = await reader.read()
        writer.close()
        await writer.wait_closed()
        assert b"200 OK" in response
        assert b'{"ok":true}' in response
    finally:
        server_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await server_task
