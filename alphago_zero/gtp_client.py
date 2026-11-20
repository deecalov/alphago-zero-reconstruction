"""Control an external GTP engine over a subprocess. Read a complete response for every command and retain
recent stderr for diagnostics.
"""

from __future__ import annotations

import os
import queue
import re
import shlex
import subprocess
import threading
import time
from collections import deque
from collections.abc import Sequence
from types import TracebackType
from typing import IO, Final, Self

_HEADER: Final = re.compile(r"([=?])(\d*)(?:\s+(.*))?\Z")
_KILL_WAIT: Final = 5.0  # Grace period after terminating the process.
_STDERR_DRAIN_WAIT: Final = 1.0  # Time allowed to drain stderr after exit.
_ERROR_TAIL_LINES: Final = 5  # Number of diagnostic stderr lines to retain.


class GTPError(Exception):
    """Base class for errors while communicating with a GTP engine."""


class GTPCommandError(GTPError):
    """The engine rejected a command with a '?' response and can still accept commands."""

    def __init__(self, engine: str, command: str, message: str) -> None:
        super().__init__(f"{engine}: command {command!r} rejected by engine: {message}")
        self.engine = engine
        self.command = command
        self.message = message


class GTPConnectionError(GTPError):
    """The engine exited, timed out or returned a malformed response."""


class GTPTimeoutError(GTPConnectionError):
    """A complete response did not arrive before the deadline."""


class GTPProtocolError(GTPConnectionError):
    """The response stream violates GTP framing."""


def _strip_quotes(token: str) -> str:
    """Remove a pair of matching outer quotes left by shlex in Windows mode."""
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "'\"":
        return token[1:-1]
    return token


def split_command(text: str, *, windows: bool | None = None) -> list[str]:
    """Split a launch command without invoking a shell. Preserve Windows path separators and handle quoted
    arguments.
    """
    if windows is None:
        windows = os.name == "nt"
    try:
        tokens = shlex.split(text, posix=not windows)
    except ValueError as error:
        raise ValueError(f"could not parse engine command: {error}") from error
    if windows:
        tokens = [_strip_quotes(token) for token in tokens]
    if not tokens:
        raise ValueError("empty launch command")
    return tokens


class GTPClient:
    """Run an engine from an argument vector and communicate over GTP streams."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        name: str = "engine",
        default_timeout: float = 60.0,
        startup_timeout: float = 120.0,
        stderr_lines: int = 200,
        quit_timeout: float = 5.0,
        cwd: str | os.PathLike[str] | None = None,
    ) -> None:
        if not command:
            raise ValueError("empty engine command")
        if min(default_timeout, startup_timeout, quit_timeout) <= 0:
            raise ValueError("timeouts must be positive")
        if stderr_lines < 1:
            raise ValueError("stderr_lines must be at least 1")
        self._command = [str(part) for part in command]
        self.name = name
        self._default_timeout = default_timeout
        self._startup_timeout = startup_timeout
        self._quit_timeout = quit_timeout
        self._stderr_lines = stderr_lines
        self._cwd = cwd
        self._proc: subprocess.Popen[str] | None = None
        self._ever_started = False
        self._fresh = False  # The engine has not received its first command yet.
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=stderr_lines)
        self._threads: list[threading.Thread] = []
        self._stderr_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Process state
    # ------------------------------------------------------------------
    @property
    def alive(self) -> bool:
        """Return whether the engine process exists and is still running."""
        return self._proc is not None and self._proc.poll() is None

    @property
    def pid(self) -> int | None:
        """Return the PID, or None if the engine has not started."""
        return self._proc.pid if self.alive and self._proc is not None else None

    @property
    def stderr_tail(self) -> list[str]:
        """Return the most recent lines from the bounded stderr buffer."""
        return list(self._stderr)

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Process lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the engine if it is not already running."""
        if self.alive:
            return
        self._terminate()
        try:
            proc = subprocess.Popen(
                self._command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=self._cwd,
            )
        except OSError as error:
            raise GTPConnectionError(f"{self.name}: failed to start {self._command[0]!r}: {error}") from error
        self._ever_started = True
        self._fresh = True
        self._proc = proc
        self._lines = queue.Queue()
        self._stderr = deque(maxlen=self._stderr_lines)
        assert proc.stdout is not None and proc.stderr is not None
        stdout_thread = threading.Thread(target=self._pump_stdout, args=(proc.stdout, self._lines), daemon=True)
        self._stderr_thread = threading.Thread(target=self._pump_stderr, args=(proc.stderr, self._stderr), daemon=True)
        self._threads = [stdout_thread, self._stderr_thread]
        for thread in self._threads:
            thread.start()

    def close(self) -> None:
        """Send quit and terminate an unresponsive process. Repeated calls are safe."""
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                assert proc.stdin is not None
                proc.stdin.write("quit\n")
                proc.stdin.flush()
                proc.wait(timeout=self._quit_timeout)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
        self._terminate()

    def restart(self) -> None:
        """Restart the subprocess with a fresh board and clocks."""
        self.close()
        self.start()

    def _terminate(self) -> None:
        """Terminate the process, drain reader threads and close the pipes."""
        proc, self._proc = self._proc, None
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=_KILL_WAIT)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=_KILL_WAIT)
        for thread in self._threads:
            thread.join(timeout=_KILL_WAIT)
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # Protocol commands
    # ------------------------------------------------------------------
    def command(self, line: str, *, timeout: float | None = None) -> str:
        """Send one command and return its successful response. Distinguish a rejected command from a broken
        connection.
        """
        if "\n" in line or "\r" in line:
            raise ValueError("a GTP command must fit on one line")
        proc = self._running_process()
        limit = self._default_timeout if timeout is None else timeout
        if limit <= 0:
            raise ValueError("timeout must be positive")
        if self._fresh:
            limit = max(limit, self._startup_timeout)
        deadline = time.monotonic() + limit
        try:
            assert proc.stdin is not None
            proc.stdin.write(line + "\n")
            proc.stdin.flush()
        except (OSError, ValueError) as error:
            raise self._fail(GTPConnectionError, f"engine is not accepting commands ({error})") from error
        succeeded, text = self._read_reply(line, deadline)
        self._fresh = False
        if not succeeded:
            raise GTPCommandError(self.name, line, text)
        return text

    def _running_process(self) -> subprocess.Popen[str]:
        if self._proc is None and not self._ever_started:
            self.start()
        if self._proc is None:
            raise GTPConnectionError(f"{self.name}: engine is not running (closed or terminated after failure)")
        return self._proc

    def _read_reply(self, line: str, deadline: float) -> tuple[bool, str]:
        """Read the response header, continuation lines and terminating blank line."""
        got_header = False
        succeeded = True
        lines: list[str] = []
        while True:
            remaining = deadline - time.monotonic()
            try:
                if remaining <= 0:
                    raise queue.Empty
                item = self._lines.get(timeout=remaining)
            except queue.Empty:
                raise self._fail(GTPTimeoutError, f"no response to {line!r} within the timeout") from None
            if item is None:
                raise self._fail(GTPConnectionError, f"engine exited without answering {line!r}{self._exit_note()}")
            if not got_header:
                if not item.strip():
                    continue  # Ignore extra blank lines between responses.
                matched = _HEADER.match(item.strip())
                if matched is None:
                    raise self._fail(GTPProtocolError, f"response to {line!r} does not start with \'=\' or \'?\': {item!r}")
                got_header = True
                succeeded = matched.group(1) == "="
                first = (matched.group(3) or "").strip()
                if first:
                    lines.append(first)
            elif item.strip():
                lines.append(item.rstrip())
            else:
                return succeeded, "\n".join(lines)

    def _exit_note(self) -> str:
        """Return the exit status, allowing a brief interval for the process to finish."""
        if self._proc is None:
            return ""
        try:
            code = self._proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            return ""
        return f" (process exited with code {code})"

    def _fail(self, error_type: type[GTPConnectionError], message: str) -> GTPConnectionError:
        """Stop the engine and attach the last stderr lines to the reported error."""
        proc = self._proc
        if proc is not None and proc.poll() is not None and self._stderr_thread is not None:
            self._stderr_thread.join(timeout=_STDERR_DRAIN_WAIT)
        tail = self.stderr_tail[-_ERROR_TAIL_LINES:]
        note = "\nengine stderr (last lines):\n  " + "\n  ".join(tail) if tail else ""
        self._terminate()
        return error_type(f"{self.name}: {message}{note}")

    # ------------------------------------------------------------------
    # Output reader threads
    # ------------------------------------------------------------------
    @staticmethod
    def _pump_stdout(stream: IO[str], lines: queue.Queue[str | None]) -> None:
        try:
            for raw in iter(stream.readline, ""):
                lines.put(raw.rstrip("\r\n"))
        except (OSError, ValueError):
            pass
        finally:
            lines.put(None)  # End of stream indicates process exit.

    @staticmethod
    def _pump_stderr(stream: IO[str], tail: deque[str]) -> None:
        try:
            for raw in iter(stream.readline, ""):
                tail.append(raw.rstrip("\r\n"))
        except (OSError, ValueError):
            pass
