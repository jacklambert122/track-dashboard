from __future__ import annotations

import errno
import signal
from unittest.mock import patch

import pytest

from track_dashboard.cli import (
    ensure_port_available,
    listener_pids,
    replace_default_server,
)


def test_ensure_port_available_accepts_unused_port() -> None:
    with patch("track_dashboard.cli.socket.create_server") as create_server:
        ensure_port_available("127.0.0.1", 5006)

    create_server.assert_called_once_with(("127.0.0.1", 5006))


def test_ensure_port_available_rejects_port_in_use() -> None:
    error = OSError(errno.EADDRINUSE, "Address already in use")
    with patch("track_dashboard.cli.socket.create_server", side_effect=error):
        with pytest.raises(SystemExit, match=r"already in use.*--port"):
            ensure_port_available("127.0.0.1", 5006)


def test_listener_pids_parses_unique_processes() -> None:
    with patch("track_dashboard.cli.os.getpid", return_value=99):
        with patch("track_dashboard.cli.subprocess.run") as run:
            run.return_value.returncode = 0
            run.return_value.stdout = "42\n99\n42\n"

            assert listener_pids(5006) == [42]


def test_replace_default_server_terminates_existing_listener() -> None:
    with (
        patch(
            "track_dashboard.cli.ensure_port_available",
            side_effect=[SystemExit(), None],
        ),
        patch("track_dashboard.cli.listener_pids", return_value=[42]),
        patch("track_dashboard.cli.os.kill") as kill,
    ):
        replace_default_server("127.0.0.1", 5006)

    kill.assert_called_once_with(42, signal.SIGTERM)
