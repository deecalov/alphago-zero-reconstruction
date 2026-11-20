"""Exercise the GTP client over a real fake-engine subprocess."""

from __future__ import annotations

import os
import sys
import time
from collections import deque
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import IO

import pytest

from alphago_zero.gtp_client import (
    GTPClient,
    GTPCommandError,
    GTPConnectionError,
    GTPError,
    GTPProtocolError,
    GTPTimeoutError,
    split_command,
)

FAKE = Path(__file__).with_name("fake_gtp_engine.py")
STARTUP = 30.0  # Allow extra startup time for the first command on loaded hosts.

MakeClient = Callable[..., GTPClient]


@pytest.fixture
def make_client() -> Iterator[MakeClient]:
    """Build fake-engine clients and close them all after the test."""
    clients: list[GTPClient] = []

    def make(*flags: str, **kwargs: object) -> GTPClient:
        kwargs.setdefault("startup_timeout", STARTUP)
        client = GTPClient([sys.executable, str(FAKE), *flags], name="fake", **kwargs)  # type: ignore[arg-type]
        clients.append(client)
        return client

    yield make
    for client in clients:
        client.close()


# ----------------------------------------------------------------------
# Command splitting
# ----------------------------------------------------------------------
def test_split_command_posix_keeps_quoted_arguments_together() -> None:
    assert split_command("pachi -t =2000 'a b' \"c d\"", windows=False) == ["pachi", "-t", "=2000", "a b", "c d"]


def test_split_command_windows_keeps_backslashes_and_strips_quotes() -> None:
    line = r'C:\tools\pachi.exe -t =2000 "C:\my dir\data" ' + "'D:\\x y\\z'"
    assert split_command(line, windows=True) == [r"C:\tools\pachi.exe", "-t", "=2000", r"C:\my dir\data", r"D:\x y\z"]


def test_split_command_posix_swallows_unquoted_backslashes() -> None:
    """Windows path separators would be treated as escapes by POSIX shlex."""
    assert split_command(r"C:\tools\pachi.exe", windows=False) == ["C:toolspachi.exe"]


def test_split_command_defaults_to_the_current_platform() -> None:
    line = r'C:\tools\pachi.exe "a b"'
    assert split_command(line) == split_command(line, windows=(os.name == "nt"))


@pytest.mark.parametrize("text", ["", "   "])
def test_split_command_rejects_an_empty_command(text: str) -> None:
    with pytest.raises(ValueError, match="empty"):
        split_command(text, windows=False)


def test_split_command_rejects_unbalanced_quotes() -> None:
    with pytest.raises(ValueError, match="could not parse"):
        split_command("pachi 'oops", windows=False)


# ----------------------------------------------------------------------
# Commands and responses
# ----------------------------------------------------------------------
def test_command_returns_the_reply_text_without_the_marker(make_client: MakeClient) -> None:
    client = make_client("--name", "Pachi")
    assert client.command("name") == "Pachi"
    assert client.command("protocol_version") == "2"


def test_command_with_an_empty_reply_returns_an_empty_string(make_client: MakeClient) -> None:
    assert make_client().command("clear_board") == ""


def test_multiline_reply_is_returned_whole(make_client: MakeClient) -> None:
    assert make_client().command("list_commands").splitlines() == ["boardsize", "clear_board", "genmove", "play"]


def test_error_reply_raises_a_command_error_and_the_engine_stays_usable(make_client: MakeClient) -> None:
    client = make_client()
    with pytest.raises(GTPCommandError) as excinfo:
        client.command("no_such_command")
    assert excinfo.value.command == "no_such_command"
    assert excinfo.value.message == "unknown command"
    assert excinfo.value.engine == "fake"
    assert "no_such_command" in str(excinfo.value) and "unknown command" in str(excinfo.value)
    assert client.alive
    assert client.command("name") == "FakeGo"  # Subsequent responses remain aligned.


def test_a_rejected_command_is_a_gtp_error_but_not_a_connection_error(make_client: MakeClient) -> None:
    with pytest.raises(GTPCommandError) as excinfo:
        make_client("--reject", "komi").command("komi 7.5")
    assert isinstance(excinfo.value, GTPError)
    assert not isinstance(excinfo.value, GTPConnectionError)


def test_reply_id_is_stripped(make_client: MakeClient) -> None:
    assert make_client("--framing", "ids").command("7 name") == "FakeGo"


def test_crlf_line_endings_do_not_leak_into_the_reply(make_client: MakeClient) -> None:
    client = make_client("--framing", "crlf")
    assert client.command("name") == "FakeGo"
    assert client.command("list_commands").splitlines() == ["boardsize", "clear_board", "genmove", "play"]


def test_extra_blank_lines_between_replies_are_skipped(make_client: MakeClient) -> None:
    client = make_client("--framing", "extra-blank")
    assert [client.command("name"), client.command("version"), client.command("name")] == [
        "FakeGo",
        "1.0",
        "FakeGo",
    ]


def test_replies_stay_aligned_over_many_commands(make_client: MakeClient) -> None:
    client = make_client()
    for _ in range(200):
        assert client.command("name") == "FakeGo"


# ----------------------------------------------------------------------
# Drain noisy engine stderr independently of protocol output.
# ----------------------------------------------------------------------
def test_a_stderr_flood_does_not_block_the_engine(make_client: MakeClient) -> None:
    """An unread stderr pipe would eventually fill and block the engine."""
    client = make_client("--stderr-lines", "20000")  # About 1.7 MB per command.
    started = time.monotonic()
    assert client.command("name", timeout=STARTUP) == "FakeGo"
    assert client.command("version", timeout=STARTUP) == "1.0"
    assert time.monotonic() - started < STARTUP


def test_stderr_tail_keeps_the_last_lines_and_is_bounded(make_client: MakeClient) -> None:
    client = make_client("--stderr-lines", "50", stderr_lines=10)
    client.command("name")
    deadline = time.monotonic() + 10
    while len(client.stderr_tail) < 10 and time.monotonic() < deadline:
        time.sleep(0.05)
    tail = client.stderr_tail
    assert len(tail) == 10
    assert all(line.startswith("# engine log") and line.endswith("x") for line in tail)  # Without a trailing newline.


# ----------------------------------------------------------------------
# Engine failure cases
# ----------------------------------------------------------------------
def test_a_crash_mid_command_is_a_connection_error_with_the_exit_code(make_client: MakeClient) -> None:
    client = make_client("--crash-on", "genmove:1", default_timeout=10)
    client.command("name")
    with pytest.raises(GTPConnectionError, match="3") as excinfo:
        client.command("genmove B")
    assert not isinstance(excinfo.value, GTPTimeoutError)
    assert not client.alive


def test_a_hung_engine_times_out_and_is_killed(make_client: MakeClient) -> None:
    client = make_client("--hang-on", "genmove:1")
    client.command("name")
    started = time.monotonic()
    with pytest.raises(GTPTimeoutError, match="genmove"):
        client.command("genmove B", timeout=0.5)
    assert time.monotonic() - started < 10
    assert not client.alive  # A late response desynchronizes command framing. Close the engine.
    with pytest.raises(GTPConnectionError, match="not running"):
        client.command("name")


def test_errors_name_the_engine_they_come_from(make_client: MakeClient) -> None:
    """Prefix errors with the engine name to identify failures in a multi-process match."""
    client = make_client("--crash-on", "genmove:1", "--reject", "komi", default_timeout=10)
    client.command("name")
    with pytest.raises(GTPCommandError, match="^fake: "):
        client.command("komi 7.5")
    with pytest.raises(GTPConnectionError, match="^fake: "):
        client.command("genmove B")


def test_connection_error_text_carries_the_last_stderr_lines(make_client: MakeClient) -> None:
    """Include the last stderr lines in the connection error."""
    client = make_client("--crash-on", "genmove:1", "--stderr-lines", "3")
    client.command("name")
    with pytest.raises(GTPConnectionError, match="engine log"):
        client.command("genmove B")


def test_the_error_waits_for_the_stderr_of_a_dead_engine(
    make_client: MakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wait for trailing stderr when stdout closes so the diagnostic survives reader scheduling."""
    original = GTPClient._pump_stderr

    def slow_pump(stream: IO[str], tail: deque[str]) -> None:
        time.sleep(0.5)
        original(stream, tail)

    monkeypatch.setattr(GTPClient, "_pump_stderr", staticmethod(slow_pump))
    client = make_client("--crash-on", "genmove:1", "--stderr-lines", "3")
    client.command("name")
    with pytest.raises(GTPConnectionError, match="engine log"):
        client.command("genmove B")


def test_alive_turns_false_when_the_engine_exits_on_its_own(make_client: MakeClient) -> None:
    client = make_client()
    client.command("quit")
    deadline = time.monotonic() + 10
    while client.alive and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not client.alive


def test_timeout_is_a_connection_error_subtype(make_client: MakeClient) -> None:
    assert issubclass(GTPTimeoutError, GTPConnectionError)
    assert issubclass(GTPProtocolError, GTPConnectionError)


def test_a_line_outside_the_protocol_is_a_protocol_error(make_client: MakeClient) -> None:
    client = make_client("--garbage-before", "version")
    client.command("name")
    with pytest.raises(GTPProtocolError, match="this is not a GTP response"):
        client.command("version")
    assert not client.alive  # Unexpected stdout invalidates command-response alignment.


def test_a_marker_glued_to_text_is_not_a_reply_header(make_client: MakeClient) -> None:
    """Reject '=oops' as a malformed header instead of consuming it as success."""
    client = make_client("--garbage-before", "version", "--garbage-text", "=oops")
    client.command("name")
    with pytest.raises(GTPProtocolError, match="=oops"):
        client.command("version")


def test_a_command_to_an_engine_that_has_already_exited_is_a_connection_error(make_client: MakeClient) -> None:
    client = make_client()
    client.command("quit")
    deadline = time.monotonic() + 10
    while client.alive and time.monotonic() < deadline:
        time.sleep(0.05)
    with pytest.raises(GTPConnectionError):
        client.command("name")


def test_the_first_command_gets_the_startup_timeout(make_client: MakeClient) -> None:
    """Allow the engine startup timeout on the first command."""
    client = make_client("--startup-delay", "1.0", default_timeout=0.2, startup_timeout=STARTUP)
    assert client.command("name") == "FakeGo"


def test_the_startup_timeout_applies_only_to_the_first_command(make_client: MakeClient) -> None:
    client = make_client("--hang-on", "version:1", default_timeout=0.5, startup_timeout=STARTUP)
    client.command("name")
    started = time.monotonic()
    with pytest.raises(GTPTimeoutError):
        client.command("version")
    assert time.monotonic() - started < 10


def test_a_missing_executable_is_a_connection_error_naming_the_program() -> None:
    client = GTPClient(["definitely-not-an-engine-xyz", "--gtp"], name="ghost")
    with pytest.raises(GTPConnectionError, match="definitely-not-an-engine-xyz"):
        client.start()
    assert not client.alive


# ----------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------
@pytest.mark.parametrize("command", ["name\nquit", "name\rquit"])
def test_a_multiline_command_is_rejected_before_anything_is_sent(
    make_client: MakeClient, tmp_path: Path, command: str
) -> None:
    """A newline would turn one request into multiple GTP commands."""
    log = tmp_path / "engine.log"
    client = make_client("--log", str(log))
    client.command("name")
    with pytest.raises(ValueError, match="one line"):
        client.command(command)
    assert client.command("version") == "1.0"  # The engine remains live and receives no extra command.
    assert log.read_text(encoding="utf-8").splitlines() == ["name", "version"]


@pytest.mark.parametrize("timeout", [0, -1.0])
def test_a_non_positive_command_timeout_is_rejected(make_client: MakeClient, timeout: float) -> None:
    client = make_client()
    with pytest.raises(ValueError, match="positive"):
        client.command("name", timeout=timeout)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"default_timeout": 0.0}, "positive"),
        ({"startup_timeout": -1.0}, "positive"),
        ({"quit_timeout": 0.0}, "positive"),
        ({"stderr_lines": 0}, "at least 1"),
    ],
)
def test_constructor_validates_its_limits(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        GTPClient([sys.executable], **kwargs)  # type: ignore[arg-type]


def test_an_empty_launch_command_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        GTPClient([])


# ----------------------------------------------------------------------
# Process lifecycle
# ----------------------------------------------------------------------
def test_close_sends_quit_and_is_idempotent(make_client: MakeClient, tmp_path: Path) -> None:
    log = tmp_path / "engine.log"
    client = make_client("--log", str(log))
    client.command("name")
    client.close()
    client.close()
    assert not client.alive
    assert log.read_text(encoding="utf-8").splitlines() == ["name", "quit"]


def test_close_kills_an_engine_that_ignores_quit(make_client: MakeClient) -> None:
    client = make_client("--hang-on", "quit:1", quit_timeout=0.5)
    client.command("name")
    started = time.monotonic()
    client.close()
    assert time.monotonic() - started < 15
    assert not client.alive


def test_close_without_start_is_harmless() -> None:
    GTPClient([sys.executable, str(FAKE)], name="idle").close()


def test_pid_is_reported_only_while_the_engine_runs(make_client: MakeClient) -> None:
    client = make_client()
    assert client.pid is None
    client.command("name")
    assert isinstance(client.pid, int)
    client.close()
    assert client.pid is None


def test_start_does_not_touch_a_running_engine(make_client: MakeClient) -> None:
    client = make_client()
    client.start()
    pid = client.pid
    assert pid is not None
    client.start()
    assert client.pid == pid
    assert client.command("name") == "FakeGo"


def test_command_after_close_is_a_connection_error(make_client: MakeClient) -> None:
    client = make_client()
    client.command("name")
    client.close()
    with pytest.raises(GTPConnectionError, match="not running"):
        client.command("name")


def test_context_manager_starts_and_stops_the_engine() -> None:
    with GTPClient([sys.executable, str(FAKE)], name="ctx", startup_timeout=STARTUP) as client:
        assert client.alive
        assert client.command("name") == "FakeGo"
    assert not client.alive


def test_command_starts_the_engine_lazily(make_client: MakeClient) -> None:
    client = make_client()
    assert not client.alive
    assert client.command("name") == "FakeGo"
    assert client.alive


def test_restart_replaces_a_dead_engine(make_client: MakeClient, tmp_path: Path) -> None:
    log = tmp_path / "engine.log"
    client = make_client("--crash-on", "genmove:1", "--log", str(log))
    client.command("name")
    with pytest.raises(GTPConnectionError):
        client.command("genmove B")
    client.restart()
    assert client.alive
    assert client.command("name") == "FakeGo"
    assert log.read_text(encoding="utf-8").splitlines() == ["name", "genmove B", "name"]


def test_restart_of_a_healthy_engine_gives_a_fresh_process(make_client: MakeClient, tmp_path: Path) -> None:
    log = tmp_path / "engine.log"
    client = make_client("--log", str(log))
    client.command("name")
    old_pid = client.pid
    client.restart()
    client.command("name")
    assert client.pid is not None and client.pid != old_pid
    assert log.read_text(encoding="utf-8").splitlines() == ["name", "quit", "name"]
