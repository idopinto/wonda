import re


def parse_predicate(completion: str) -> tuple[str, str]:
    """
    Parse model completion to extract predicate content and line label.

    Args:
        completion: The model's final completion string.

    Returns:
        Tuple of (predicate_content, line_label).
    """
    if not completion:
        return "", ""

    assert_pos = completion.find("assert(")
    if assert_pos == -1:
        return "", ""

    # Track parentheses depth to find matching closing paren
    start_pos = assert_pos + 7  # after "assert("
    paren_count = 0
    end_pos = None

    for i in range(start_pos, len(completion)):
        if completion[i] == "(":
            paren_count += 1
        elif completion[i] == ")":
            if paren_count == 0:
                end_pos = i
                break
            paren_count -= 1

    if end_pos is None:
        return "", ""

    invariant = completion[start_pos:end_pos].strip()
    rest = completion[end_pos:]

    # Try multiple comment patterns (case-insensitive)
    patterns = [
        r"\);\s*//\s*[Ll]ine\s+([A-Za-z])",  # ); // Line A
        r"\);\s*//\s*([A-Za-z])",  # ); // A
        r"\);\s*/\*\s*[Ll]ine\s+([A-Za-z])\s*\*/",  # ); /* Line A */
    ]

    for pattern in patterns:
        match = re.search(pattern, rest)
        if match:
            return invariant, match.group(1).upper()

    return "", ""
