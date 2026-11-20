"""Standalone fake GTP engine for subprocess and external-match tests."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

COLUMNS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"  # GTP omits column I.


def _split_counted(spec: str | None) -> tuple[str, int] | None:
    """Parse a command trigger and optional occurrence number, defaulting to the first occurrence."""
    if not spec:
        return None
    name, _, count = spec.partition(":")
    return name.lower(), int(count) if count else 1


class FakeEngine:
    """State and command handlers for a fake engine."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.board_size = 9
        self.occupied: set[str] = set()
        self.own_played = 0
        self.game_number = 0
        self.seen: dict[str, int] = {}
        self.crash_on = _split_counted(args.crash_on)
        self.hang_on = _split_counted(args.hang_on)
        self.resign_games = {int(item) for item in args.resign_games.split(",") if item}
        self.scores = [item for item in args.final_scores.split(",") if item] or ["B+0.5"]
        self.out = sys.stdout.buffer

    # ------------------------------------------------------------------
    # Input and output
    # ------------------------------------------------------------------
    def _write(self, text: str) -> None:
        newline = "\r\n" if self.args.framing == "crlf" else "\n"
        self.out.write(text.replace("\n", newline).encode("utf-8"))
        self.out.flush()

    def _reply(self, command_id: str, body: str, ok: bool = True) -> None:
        prefix = ("=" if ok else "?") + (command_id if self.args.framing == "ids" else "")
        text = f"{prefix} {body}" if body else prefix
        self._write(text + "\n\n" + ("\n" if self.args.framing == "extra-blank" else ""))

    def _log(self, line: str) -> None:
        if self.args.log:
            with Path(self.args.log).open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    # ------------------------------------------------------------------
    # Game state
    # ------------------------------------------------------------------
    def _vertex_text(self, vertex: str) -> str:
        if self.args.case == "lower":
            return vertex.lower()
        return vertex.upper()

    def _own_order(self) -> list[str]:
        rows = range(1, self.board_size + 1)
        ordered_rows = rows if self.args.zone == "bottom" else reversed(rows)
        return [f"{COLUMNS[col]}{row}" for row in ordered_rows for col in range(self.board_size)]

    def _genmove(self) -> str:
        if self.args.illegal_genmove:
            return self.args.illegal_genmove
        if self.game_number in self.resign_games and self.own_played == 0:
            return "resign"
        if self.own_played >= self.args.own_moves:
            return "PASS" if self.args.case == "upper" else "pass"
        for vertex in self._own_order():
            if vertex not in self.occupied:
                self.occupied.add(vertex)
                self.own_played += 1
                return self._vertex_text(vertex)
        return "pass"

    def _final_score(self) -> str:
        return self.scores[(max(self.game_number, 1) - 1) % len(self.scores)]

    # ------------------------------------------------------------------
    # Command dispatch
    # ------------------------------------------------------------------
    def _maybe_break(self, name: str) -> None:
        count = self.seen[name] = self.seen.get(name, 0) + 1
        if self.crash_on == (name, count):
            os._exit(3)
        if self.hang_on == (name, count):
            time.sleep(3600)

    def handle(self, line: str) -> bool:
        """Process one line and return True after quit."""
        self._log(line)
        tokens = line.split()
        if not tokens:
            return False
        command_id = tokens.pop(0) if tokens[0].isdigit() else ""
        if not tokens:
            return False
        name, rest = tokens[0].lower(), tokens[1:]
        for _ in range(self.args.stderr_lines):
            print("# engine log: " + "x" * 80, file=sys.stderr)
        sys.stderr.flush()
        if name == self.args.garbage_before:
            self._write(self.args.garbage_text + "\n")
        self._maybe_break(name)
        if name in {item.lower() for item in self.args.reject}:
            self._reply(command_id, "rejected", ok=False)
            return False
        return self._dispatch(command_id, name, rest)

    def _dispatch(self, command_id: str, name: str, rest: list[str]) -> bool:
        if name == "quit":
            self._reply(command_id, "")
            return True
        if name == "boardsize":
            if self.args.unacceptable_size:
                self._reply(command_id, "unacceptable size", ok=False)
                return False
            self.board_size = int(rest[0])
            self.occupied.clear()
            self.own_played = 0
        elif name == "clear_board":
            self.game_number += 1
            self.occupied.clear()
            self.own_played = 0
        elif name == "play":
            vertex = rest[1].upper()
            if vertex == self.args.reject_play_vertex or vertex in self.occupied:
                self._reply(command_id, "illegal move", ok=False)
                return False
            if vertex != "PASS":
                self.occupied.add(vertex)
        elif name == "genmove":
            self._reply(command_id, self._genmove())
            return False
        elif name == "final_score":
            self._reply(command_id, self._final_score())
            return False
        elif name in {"name", "version", "protocol_version", "list_commands", "showboard", "known_command"}:
            self._reply(command_id, self._info(name, rest))
            return False
        elif name not in {"komi", "time_settings", "time_left"}:
            self._reply(command_id, "unknown command", ok=False)
            return False
        self._reply(command_id, "")
        return False

    def _info(self, name: str, rest: list[str]) -> str:
        if name == "name":
            return self.args.name
        if name == "version":
            return self.args.engine_version
        if name == "protocol_version":
            return "2"
        if name == "known_command":
            return "true"
        return "boardsize\nclear_board\ngenmove\nplay"  # list_commands and showboard use multiline responses.

    def run(self) -> None:
        if self.args.startup_delay:
            time.sleep(self.args.startup_delay)
        for raw in sys.stdin.buffer:
            if self.handle(raw.decode("utf-8", errors="replace").strip()):
                break


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="FakeGo")
    parser.add_argument("--engine-version", default="1.0")
    parser.add_argument("--zone", choices=("top", "bottom"), default="bottom")
    parser.add_argument("--own-moves", type=int, default=3)
    parser.add_argument("--final-scores", default="")
    parser.add_argument("--resign-games", default="")
    parser.add_argument("--log", default="")
    parser.add_argument("--stderr-lines", type=int, default=0)
    parser.add_argument("--crash-on", default="")
    parser.add_argument("--hang-on", default="")
    parser.add_argument("--reject", nargs="*", default=[])
    parser.add_argument("--reject-play-vertex", default="")
    parser.add_argument("--garbage-before", default="")
    parser.add_argument("--garbage-text", default="this is not a GTP response")
    parser.add_argument("--illegal-genmove", default="")
    parser.add_argument("--startup-delay", type=float, default=0.0)
    parser.add_argument("--unacceptable-size", action="store_true")
    parser.add_argument("--framing", choices=("plain", "crlf", "ids", "extra-blank"), default="plain")
    parser.add_argument("--case", choices=("upper", "lower"), default="upper")
    return parser


if __name__ == "__main__":
    FakeEngine(_parser().parse_args()).run()
