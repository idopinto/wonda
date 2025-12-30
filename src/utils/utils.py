def reindent_c_code(code: str, indent_size: int = 4) -> str:
    """
    Re-indent C code based on brace counting.

    This is useful for displaying code that was processed with clang-format
    using IndentWidth: 0, which removes all indentation.

    Args:
        code: C code string to re-indent.
        indent_size: Number of spaces per indentation level.

    Returns:
        Re-indented C code string.
    """
    if not code or not code.strip():
        return code

    lines = code.split("\n")
    result = []
    indent_level = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            result.append("")
            continue

        # Count braces in this line
        open_count = stripped.count("{")
        close_count = stripped.count("}")

        # Determine indent for THIS line:
        # If line starts with }, decrease indent first
        line_indent = indent_level
        if stripped.startswith("}"):
            line_indent = max(0, indent_level - 1)

        # Apply indent and add to result
        indented_line = " " * (line_indent * indent_size) + stripped
        result.append(indented_line)

        # Calculate indent for NEXT line based on net brace change
        # But handle the case where line starts with } specially
        if stripped.startswith("}"):
            # We already decreased for this line, now handle any { on same line
            # Example: "} else {" - net change from indent_level should be 0
            # We decreased by 1 for the }, now increase by open_count
            indent_level = max(0, indent_level - 1 + open_count)
        else:
            # Normal case: add opens, subtract closes
            net_change = open_count - close_count
            indent_level = max(0, indent_level + net_change)

    return "\n".join(result)
