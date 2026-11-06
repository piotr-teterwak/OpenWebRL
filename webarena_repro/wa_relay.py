#!/usr/bin/env python3
"""Egress for processes inside a site's loopback-only network namespace (model API calls only).

    wa_relay.py out --sock PATH [--to tillicum-dtn01:3128]   host side: UNIX socket -> the cluster HTTP proxy
    wa_relay.py in  --sock PATH [--listen 127.0.0.1:3128]    in the namespace: loopback TCP -> UNIX socket

Pathname UNIX sockets are not network-namespace scoped, so the pair gives the agent an HTTP(S) proxy at
127.0.0.1:3128 inside the namespace. Bytes are copied unchanged (HTTPS stays end-to-end TLS through the proxy's
CONNECT). Only the agent's model client is pointed at it; the browser gets no proxy. Pattern from Zixian's
v12 owrl_uds_relay.py. Stdlib only.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path


async def pipe(reader, writer):
    try:
        while data := await reader.read(1 << 16):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['out', 'in'])
    ap.add_argument('--sock', required=True)
    ap.add_argument('--to', default=os.environ.get('WA_UPSTREAM_PROXY', 'tillicum-dtn01:3128'))
    ap.add_argument('--listen', default='127.0.0.1:3128')
    ap.add_argument('--ready')
    a = ap.parse_args()
    sock = Path(a.sock)
    if a.mode == 'out':
        host, port = a.to.rsplit(':', 1)

        async def handle(r, w):
            try:
                ur, uw = await asyncio.open_connection(host, int(port))
            except OSError:
                w.close(); return
            await asyncio.gather(pipe(r, uw), pipe(ur, w))
        if sock.exists():
            sock.unlink()
        old = os.umask(0o077)
        try:
            server = await asyncio.start_unix_server(handle, path=str(sock))
        finally:
            os.umask(old)
    else:
        host, port = a.listen.rsplit(':', 1)

        async def handle(r, w):
            try:
                ur, uw = await asyncio.open_unix_connection(str(sock))
            except OSError:
                w.close(); return
            await asyncio.gather(pipe(r, uw), pipe(ur, w))
        server = await asyncio.start_server(handle, host, int(port))
    if a.ready:
        Path(a.ready).write_text(json.dumps(dict(mode=a.mode, sock=str(sock), pid=os.getpid())))
    async with server:
        await server.serve_forever()


if __name__ == '__main__':
    asyncio.run(main())
