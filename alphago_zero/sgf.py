from __future__ import annotations

"""Import and export FF[4] Smart Game Format records for Go."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .game import BLACK, WHITE, GoState

_SGF_LETTERS = "abcdefghijklmnopqrstuvwxyz"


def _action_to_sgf(action: int, board_size: int) -> str:
    """Encode a board action as an SGF coordinate, using the empty value for pass."""
    if action == board_size * board_size:
        return ""
    row = action // board_size
    col = action % board_size
    return f"{_SGF_LETTERS[col]}{_SGF_LETTERS[row]}"


def format_result(winner: int, resigned: bool, score_margin: float | None) -> str:
    """Format a result such as B+R or W+3.5, with an optional margin."""
    side = "B" if winner == BLACK else "W"
    if resigned:
        return f"{side}+R"
    if score_margin is None:
        return f"{side}+"
    margin = f"{score_margin:g}"
    return f"{side}+{margin}"


def _escape_text(text: str) -> str:
    """Escape backslashes and closing brackets in text supplied by engines and users."""
    return text.replace("\\", "\\\\").replace("]", "\\]")


def game_to_sgf(
    *,
    board_size: int,
    komi: float,
    moves: list[int],
    winner: int | None = None,
    resigned: bool = False,
    score_margin: float | None = None,
    black_name: str = "black",
    white_name: str = "white",
    comment: str | None = None,
    result: str | None = None,
) -> str:
    """Build the SGF record for one game from its move list and metadata."""
    if result is None:
        if winner is None:
            raise ValueError("game_to_sgf requires winner or result")
        result = format_result(winner, resigned, score_margin)
    header_props = [
        "GM[1]",  # Go
        "FF[4]",  # SGF file format version 4.
        f"SZ[{board_size}]",
        f"KM[{komi:g}]",
        f"PB[{_escape_text(black_name)}]",
        f"PW[{_escape_text(white_name)}]",
        f"RE[{_escape_text(result)}]",
    ]
    if comment:
        header_props.append(f"C[{_escape_text(comment)}]")

    nodes = []
    for index, action in enumerate(moves):
        color = "B" if index % 2 == 0 else "W"
        nodes.append(f";{color}[{_action_to_sgf(action, board_size)}]")

    return "(;" + "".join(header_props) + "".join(nodes) + ")"


def save_sgf(path: Path, sgf_text: str) -> None:
    """Create parent directories and write the SGF file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sgf_text, encoding="utf-8")


# ----------------------------------------------------------------------
# SGF import
# ----------------------------------------------------------------------


class SGFParseError(ValueError):
    """Malformed SGF or a game that cannot be replayed under the selected rules."""


@dataclass(slots=True)
class ParsedGame:
    """Parsed root properties, setup stones and main-line moves as (color, action) pairs."""

    board_size: int
    komi: float
    result: str | None
    black_name: str
    white_name: str
    handicap: int
    setup_black: tuple[int, ...]
    setup_white: tuple[int, ...]
    setup_clear: tuple[int, ...]
    first_to_play: int | None  # Optional PL property. Otherwise infer the first player from setup.
    moves: tuple[tuple[int, int], ...]
    root_comment: str | None
    warnings: tuple[str, ...]


# Record unsupported root and setup properties in the warnings.
_KNOWN_PROPERTIES = {"GM", "FF", "SZ", "KM", "RE", "PB", "PW", "HA", "PL", "C", "AB", "AW", "AE", "B", "W"}

# One parsed node maps property names to lists of values.
_Node = dict[str, list[str]]


@dataclass(slots=True)
class _Parser:
    """Parse SGF trees recursively, following the main line and counting skipped variations."""

    text: str
    pos: int = 0
    skipped_variations: int = field(default=0)

    def _fail(self, message: str) -> SGFParseError:
        return SGFParseError(f"SGF (offset {self.pos}): {message}")

    def _skip_space(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def _peek(self) -> str:
        self._skip_space()
        return self.text[self.pos] if self.pos < len(self.text) else ""

    def parse_main_line(self) -> list[_Node]:
        """Parse the first game tree and return its main-line nodes."""
        if self._peek() != "(":
            raise self._fail("expected '(' at the start of a game tree")
        nodes = self._parse_tree()
        return nodes

    def _parse_tree(self) -> list[_Node]:
        self.pos += 1  # '('
        nodes: list[_Node] = []
        while self._peek() == ";":
            nodes.append(self._parse_node())
        children: list[list[_Node]] = []
        while self._peek() == "(":
            children.append(self._parse_tree())
        if self._peek() != ")":
            raise self._fail("expected ')' at the end of a game tree")
        self.pos += 1
        if children:
            # Follow the first variation and count the other branches.
            self.skipped_variations += len(children) - 1
            nodes.extend(children[0])
        return nodes

    def _parse_node(self) -> _Node:
        self.pos += 1  # ';'
        node: _Node = {}
        while True:
            self._skip_space()
            if self.pos >= len(self.text) or self.text[self.pos] in ";()":
                return node
            if not self.text[self.pos].isalpha():
                raise self._fail(f"expected property identifier, found {self.text[self.pos]!r}")
            start = self.pos
            while self.pos < len(self.text) and self.text[self.pos].isalpha():
                self.pos += 1
            ident = self.text[start : self.pos].upper()
            values: list[str] = []
            while self._peek() == "[":
                values.append(self._parse_value())
            if not values:
                raise self._fail(f"property {ident} has no \'[...]\' value")
            node.setdefault(ident, []).extend(values)

    def _parse_value(self) -> str:
        self.pos += 1  # '['
        chars: list[str] = []
        while self.pos < len(self.text):
            char = self.text[self.pos]
            if char == "\\" and self.pos + 1 < len(self.text):
                escaped = self.text[self.pos + 1]
                if escaped != "\n":  # An escaped line break is a soft wrap and contributes no character.
                    chars.append(escaped)
                self.pos += 2
                continue
            if char == "]":
                self.pos += 1
                return "".join(chars)
            chars.append(char)
            self.pos += 1
        raise self._fail("unterminated property value (missing ']')")


def _sgf_point_to_action(value: str, board_size: int, *, context: str) -> int:
    """Decode an SGF point or a supported pass notation."""
    token = value.strip()
    if token == "":
        return board_size * board_size
    if len(token) != 2:
        raise SGFParseError(f"{context}: invalid coordinate {value!r}")
    col = _SGF_LETTERS.index(token[0]) if token[0] in _SGF_LETTERS else -1
    row = _SGF_LETTERS.index(token[1]) if token[1] in _SGF_LETTERS else -1
    if col < 0 or row < 0:
        # Uppercase SGF coordinates continue after z on boards larger than 26.
        if token[0].isupper() and token[1].isupper():
            col = 26 + (ord(token[0]) - ord("A"))
            row = 26 + (ord(token[1]) - ord("A"))
        elif token.lower() == "tt":
            return board_size * board_size
        else:
            raise SGFParseError(f"{context}: invalid coordinate {value!r}")
    if col >= board_size or row >= board_size:
        if token.lower() == "tt":
            return board_size * board_size
        raise SGFParseError(f"{context}: coordinate {value!r} outside the board {board_size}x{board_size}")
    return row * board_size + col


def parse_sgf(text: str) -> ParsedGame:
    """Parse text into a ParsedGame before replaying its moves."""
    parser = _Parser(text=text)
    nodes = parser.parse_main_line()
    if not nodes:
        raise SGFParseError("SGF: game tree contains no nodes")

    warnings: list[str] = []
    if parser.skipped_variations:
        warnings.append(f"skipped variations outside the main line: {parser.skipped_variations}")
    if parser.text[parser.pos :].strip():
        warnings.append("skipped content after the first game tree")

    def first_value(node: _Node, ident: str) -> str | None:
        values = node.get(ident)
        return values[0] if values else None

    root = nodes[0]
    board_size = 19  # SGF default when SZ is absent.
    sz_value = first_value(root, "SZ")
    if sz_value is not None:
        parts = sz_value.split(":")
        try:
            sizes = [int(part) for part in parts]
        except ValueError as exc:
            raise SGFParseError(f"SZ: invalid board size {sz_value!r}") from exc
        if len(sizes) == 2 and sizes[0] != sizes[1]:
            raise SGFParseError(f"SZ: only square boards are supported, got {sz_value!r}")
        board_size = sizes[0]
        if board_size < 1:
            raise SGFParseError(f"SZ: board size must be >= 1, got {board_size}")

    komi = 0.0  # SGF default when KM is absent.
    km_value = first_value(root, "KM")
    if km_value is not None:
        try:
            komi = float(km_value)
        except ValueError as exc:
            raise SGFParseError(f"KM: invalid komi {km_value!r}") from exc

    handicap = 0
    ha_value = first_value(root, "HA")
    if ha_value is not None:
        try:
            handicap = int(ha_value)
        except ValueError as exc:
            raise SGFParseError(f"HA: invalid handicap {ha_value!r}") from exc

    setup_black: list[int] = []
    setup_white: list[int] = []
    setup_clear: list[int] = []
    first_to_play: int | None = None
    moves: list[tuple[int, int]] = []
    seen_unknown: set[str] = set()
    moves_started = False

    for node_index, node in enumerate(nodes):
        for ident, values in node.items():
            if ident not in _KNOWN_PROPERTIES and ident not in seen_unknown:
                seen_unknown.add(ident)
                warnings.append(f"unsupported property {ident} skipped")
            if ident in ("AB", "AW", "AE"):
                if moves_started:
                    raise SGFParseError(
                        f"node {node_index}: setup stones ({ident}) after play begins are unsupported"
                    )
                target = setup_black if ident == "AB" else setup_white if ident == "AW" else setup_clear
                target.extend(
                    _sgf_point_to_action(value, board_size, context=f"node {node_index} ({ident})")
                    for value in values
                )
            elif ident == "PL" and not moves_started:
                token = values[0].strip().upper()
                if token not in ("B", "W"):
                    raise SGFParseError(f"PL: expected B or W, got {values[0]!r}")
                first_to_play = BLACK if token == "B" else WHITE
            elif ident in ("B", "W"):
                moves_started = True
                color = BLACK if ident == "B" else WHITE
                for value in values:
                    action = _sgf_point_to_action(
                        value, board_size, context=f"move {len(moves) + 1} ({ident})"
                    )
                    moves.append((color, action))

    return ParsedGame(
        board_size=board_size,
        komi=komi,
        result=first_value(root, "RE"),
        black_name=first_value(root, "PB") or "black",
        white_name=first_value(root, "PW") or "white",
        handicap=handicap,
        setup_black=tuple(setup_black),
        setup_white=tuple(setup_white),
        setup_clear=tuple(setup_clear),
        first_to_play=first_to_play,
        moves=tuple(moves),
        root_comment=first_value(root, "C"),
        warnings=tuple(warnings),
    )


def replay_states(
    game: ParsedGame,
    *,
    superko_rule: str = "positional",
    max_moves: int | None = None,
) -> list[GoState]:
    """Replay the setup and main line, returning the initial position and one state after each move."""
    size = game.board_size
    board = np.zeros((size, size), dtype=np.int8)
    for action in game.setup_black:
        if action < size * size:
            board[action // size, action % size] = BLACK
    for action in game.setup_white:
        if action < size * size:
            board[action // size, action % size] = WHITE
    for action in game.setup_clear:
        if action < size * size:
            board[action // size, action % size] = 0

    if game.first_to_play is not None:
        to_play = game.first_to_play
    elif game.setup_black and not game.setup_white:
        to_play = WHITE  # Black's handicap stones are already on the board, so White plays first.
    else:
        to_play = BLACK

    if max_moves is None:
        max_moves = max(2 * size * size, len(game.moves) + 2)
    state = GoState.from_board(
        board, to_play=to_play, komi=game.komi, max_moves=max_moves, superko_rule=superko_rule
    )
    states = [state]
    color_names = {BLACK: "black", WHITE: "white"}
    for index, (color, action) in enumerate(game.moves, start=1):
        if color != state.to_play:
            raise SGFParseError(
                f"move {index}: expected a move by {color_names[state.to_play]}, but SGF gives {color_names[color]}"
            )
        if not state.is_legal(action):
            raise SGFParseError(f"move {index}: illegal move (occupied, suicide or ko)")
        state = state.play(action)
        states.append(state)
    return states


def replay_game(
    game: ParsedGame,
    *,
    superko_rule: str = "positional",
    max_moves: int | None = None,
) -> GoState:
    """Return the final position after SGF replay."""
    return replay_states(game, superko_rule=superko_rule, max_moves=max_moves)[-1]


def load_sgf(path: Path, *, superko_rule: str = "positional") -> GoState:
    """Read an SGF file and return its final game state."""
    return replay_game(parse_sgf(path.read_text(encoding="utf-8")), superko_rule=superko_rule)
