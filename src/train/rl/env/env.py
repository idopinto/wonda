# @title (Collapsible) 2048 Game Implementation
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
import random
import copy
from unsloth import execute_with_time_limit


def _compress_and_merge_row_left(row: List[int]) -> Tuple[List[int], int, bool]:
    n = len(row)
    tiles = [x for x in row if x != 0]
    gained = 0
    i = 0
    merged = []
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            v = tiles[i] * 2
            gained += v
            merged.append(v)
            i += 2
        else:
            merged.append(tiles[i])
            i += 1
    merged += [0] * (n - len(merged))
    changed = merged != row
    return merged, gained, changed


def _move_left(board: List[List[int]]) -> Tuple[List[List[int]], int, bool]:
    changed_any = False
    total_gain = 0
    new_board = []
    for row in board:
        new_row, gained, changed = _compress_and_merge_row_left(row)
        new_board.append(new_row)
        total_gain += gained
        changed_any = changed_any or changed
    return new_board, total_gain, changed_any


def _move_right(board: List[List[int]]) -> Tuple[List[List[int]], int, bool]:
    changed_any = False
    total_gain = 0
    new_board = []
    for row in board:
        rev = list(reversed(row))
        new_rev, gained, changed = _compress_and_merge_row_left(rev)
        new_row = list(reversed(new_rev))
        new_board.append(new_row)
        total_gain += gained
        changed_any = changed_any or changed
    return new_board, total_gain, changed_any


def _transpose(board: List[List[int]]) -> List[List[int]]:
    return [list(row) for row in zip(*board)]


def _move_up(board: List[List[int]]) -> Tuple[List[List[int]], int, bool]:
    t = _transpose(board)
    moved, gain, changed = _move_left(t)
    return _transpose(moved), gain, changed


def _move_down(board: List[List[int]]) -> Tuple[List[List[int]], int, bool]:
    t = _transpose(board)
    moved, gain, changed = _move_right(t)
    return _transpose(moved), gain, changed


def _empty_cells(board: List[List[int]]) -> List[Tuple[int, int]]:
    size = len(board)
    return [(r, c) for r in range(size) for c in range(size) if board[r][c] == 0]


def _can_move(board: List[List[int]]) -> bool:
    if _empty_cells(board):
        return True
    size = len(board)
    for r in range(size):
        for c in range(size - 1):
            if board[r][c] == board[r][c + 1]:
                return True
    for r in range(size - 1):
        for c in range(size):
            if board[r][c] == board[r + 1][c]:
                return True
    return False


@dataclass
class GameBoard:
    size: int
    seed: Optional[int] = None
    target: int = 2048
    probability_fours: float = 0.10  # originally spawns (4) 10% of the time!
    _rng: random.Random = field(init=False, repr=False)
    _board: List[List[int]] = field(init=False, repr=False)
    _score: int = field(default=0, init=False, repr=False)
    _state: str = field(default="ongoing", init=False, repr=False)

    def __post_init__(self):
        if self.size < 2:
            raise ValueError("Board size must be at least 2.")
        self._rng = random.Random(self.seed)
        self._board = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self._add_random_tile()
        self._add_random_tile()
        self._update_state_after_change()

    class _BoardView:
        def __init__(self, game: "GameBoard"):
            self._game = game

        def __iter__(self):
            return iter(self._game._board)

        def __len__(self):
            return len(self._game._board)

        def __getitem__(self, idx):
            return self._game._board[idx]

        def __repr__(self) -> str:
            return repr(self._game._board)

        __str__ = __repr__

        def do_action(self, key: str) -> None:
            self._game.do_action(key)

        def state(self) -> str:
            return self._game.state()

        def pretty(
            self, colors: bool = True, border: bool = True, dot_for_zero: bool = True
        ) -> str:
            return self._game._render_pretty(
                colors=colors, border=border, dot_for_zero=dot_for_zero
            )

    def board(self) -> "_BoardView":
        return GameBoard._BoardView(self)

    def state(self) -> str:
        return self._state

    def score(self) -> int:
        return self._score

    def do_action(self, key: str) -> None:
        if self._state != "ongoing":
            return
        if not isinstance(key, str) or len(key) == 0:
            self._state = "failed"
            return
        k = key.strip().lower()
        if k == "q":
            self._state = "failed"
            return
        move_map = {"a": _move_left, "d": _move_right, "w": _move_up, "s": _move_down}
        if k not in move_map:
            self._state = "failed"
            return
        mover = move_map[k]
        new_board, gain, changed = mover(self._board)
        if changed:
            self._board = new_board
            self._score += gain
            self._add_random_tile()
        self._update_state_after_change()

    def _add_random_tile(self) -> bool:
        empties = _empty_cells(self._board)
        if not empties:
            return False
        r, c = self._rng.choice(empties)
        self._board[r][c] = 4 if self._rng.random() < self.probability_fours else 2
        return True

    def _update_state_after_change(self) -> None:
        if any(self.target in row for row in self._board):
            self._state = "success"
            return
        if not _can_move(self._board):
            self._state = "failed"
            return
        self._state = "ongoing"

    def _render_pretty(
        self, colors: bool = True, border: bool = True, dot_for_zero: bool = True
    ) -> str:
        """
        Pretty-print the board with colors that scale from 0 up to self.target.
        Uses ANSI 256-color codes (works in most terminals). Set colors=False to disable.
        """
        import math

        b = self._board
        mx = max((max(row) for row in b), default=0)
        cell_w = max(3, len(str(mx)))

        RESET = "\x1b[0m"

        # A smooth-ish gradient from cool → warm
        # (blue/cyan/green → yellow/orange/red). Tweak or expand as you like.
        GRAD = [
            33,
            39,
            45,
            51,
            50,
            49,
            48,
            47,
            46,
            82,
            118,
            154,
            190,
            226,
            220,
            214,
            208,
            202,
            196,
        ]
        ZERO_FG = 239  # dim gray

        def color_code(v: int) -> str:
            if not colors:
                return ""
            if v == 0:
                return f"\x1b[38;5;{ZERO_FG}m"
            # Normalize by exponent relative to target: r in [0,1]
            t = max(2, self.target)  # safety; avoid log2(1)
            # Guard: if v is not a power of two or is <1, handle gracefully
            try:
                r = max(0.0, min(1.0, math.log2(v) / math.log2(t)))
            except ValueError:
                r = 0.0
            idx = int(round(r * (len(GRAD) - 1)))
            return f"\x1b[38;5;{GRAD[idx]}m"

        def fmt(v: int) -> str:
            s = "." if (v == 0 and dot_for_zero) else str(v)
            s = s.rjust(cell_w)
            return color_code(v) + s + (RESET if colors else "")

        def hline(left: str, mid: str, right: str) -> str:
            return left + mid.join("─" * cell_w for _ in range(self.size)) + right

        rows = []
        if border:
            rows.append(hline("┌", "┬", "┐"))
        for r in range(self.size):
            content = "│".join(fmt(v) for v in b[r])
            rows.append(("│" + content + "│") if border else content)
            if border:
                rows.append(
                    hline(
                        "└" if r == self.size - 1 else "├",
                        "┴" if r == self.size - 1 else "┼",
                        "┘" if r == self.size - 1 else "┤",
                    )
                )
        return "\n".join(rows)


def _execute_strategy(strategy: Callable, game: GameBoard):
    assert callable(strategy)

    steps = 0
    while game.state() == "ongoing":
        print(game.board().pretty(), game.state())
        action = strategy(list(game.board()))
        print(f"Action: {action}")
        steps += 1
        if type(action) is not str:
            return steps, "failed"
        game.do_action(action)
    return steps, game.state()


@execute_with_time_limit(2)
def execute_strategy(strategy: Callable, game: GameBoard):
    return _execute_strategy(strategy, game)


def always_move_left(board):
    return "W"


if __name__ == "__main__":
    game = GameBoard(size=8, seed=42, target=2048, probability_fours=0.10)
    try:
        execute_strategy(always_move_left, game)
    except TimeoutError as e:
        print(f"Timed out with error = {str(e)}")
    # game = GameBoard(size=5, seed=42, target=8, probability_fours=0.10)
    # print(game.board().pretty(), game.state())
    # game.do_action("A")
    # print(game.board().pretty(), game.state())
    # game.do_action("W")
    # print(game.board().pretty(), game.state())
