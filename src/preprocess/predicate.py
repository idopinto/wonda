from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Predicate:
    """
    Project-wide predicate container.

    Historically predicates were tied to a concrete source line via `line_number`.
    The AST-based pipeline can instead use `marker_name` (e.g., INVARIANT_MARKER_1,
    TARGET_ASSERT_MARKER) and may leave `line_number=None`.
    """

    content: str
    marker_name: Optional[str] = None

    def __repr__(self):
        return f"Predicate(content={self.content}, marker_name={self.marker_name})"
