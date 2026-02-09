from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Property:
    """
    A property is a pair (location, predicate) where the location is a loop
    marker (e.g. INVARIANT_MARKER_1) and the predicate is a boolean expression
    over the program variables.
    """

    content: str
    marker_name: Optional[str] = None

    def __repr__(self):
        return f"Property(content={self.content}, marker_name={self.marker_name})"

    def __eq__(self, other: Property) -> bool:
        return self.content == other.content and self.marker_name == other.marker_name
