from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Predicate:
    """
    Project-wide predicate container.
    The AST-based pipeline can use `marker_name` (e.g., INVARIANT_MARKER_1)
    """

    content: str
    marker_name: Optional[str] = None

    def __repr__(self):
        return f"Predicate(content={self.content}, marker_name={self.marker_name})"

    def __eq__(self, other: Predicate) -> bool:
        return self.content == other.content and self.marker_name == other.marker_name
