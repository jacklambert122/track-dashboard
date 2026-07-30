from __future__ import annotations

import argparse
import errno
import os
import signal
import socket
import subprocess
import time

import panel as pn

from .entry import DashboardEntry

DEFAULT_PORT = 5006


def ensure_port_available(address: str, port: int) -> None:
    """Fail with a useful message when another server already owns the port."""
    try:
        with socket.create_server((address, port)):
            pass
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            raise SystemExit(
                f"Cannot start Track Dashboard: {address}:{port} is already in use. "
                "Stop the existing server or choose another port with --port."
            ) from exc
        raise


def listener_pids(port: int) -> list[int]:
    """Return processes listening on a TCP port."""
    result = subprocess.run(
        ["lsof", "-t", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode not in {0, 1}:
        raise SystemExit(f"Could not inspect port {port}: {result.stderr.strip()}")
    return sorted(
        {
            int(line)
            for line in result.stdout.splitlines()
            if line.isdigit() and int(line) != os.getpid()
        }
    )


def replace_default_server(address: str, port: int) -> None:
    """Stop an existing default-port server and wait for its socket to close."""
    try:
        ensure_port_available(address, port)
        return
    except SystemExit:
        pids = listener_pids(port)

    if not pids:
        ensure_port_available(address, port)

    for pid in pids:
        os.kill(pid, signal.SIGTERM)

    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            ensure_port_available(address, port)
            return
        except SystemExit:
            time.sleep(0.1)

    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    time.sleep(0.1)
    ensure_port_available(address, port)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Track Dashboard server.")
    parser.add_argument(
        "input_file",
        nargs="?",
        help="Optional .csv or .parquet input file; defaults to example data.",
    )
    parser.add_argument("--track-id-col", default="track_id")
    parser.add_argument("--address", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    try:
        dashboard = DashboardEntry(
            args.input_file,
            track_id_col=args.track_id_col,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    if args.port == DEFAULT_PORT:
        replace_default_server(args.address, args.port)
    else:
        ensure_port_available(args.address, args.port)

    pn.extension("tabulator", sizing_mode="stretch_width")
    pn.serve(
        dashboard.view(),
        title="Track Dashboard",
        show=True,
        address=args.address,
        port=args.port,
    )
