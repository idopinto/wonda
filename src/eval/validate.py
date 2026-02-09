"""Validation utilities for model-generated loop invariants.

Provides JSON parsing of model answers, syntactic validation of C boolean
expressions (rejecting assignments and mutations), and a combined validation
entry point used by the evaluation scorer.

Also used in the preprocessing step to filter ground-truth invariants
by syntactic correctness before building training datasets.
"""
import json
import re
from typing import Optional

from src.core.property import Property

def parse_model_answer(raw_model_answer: str) -> Property:
    """
    Parse model answer string to json and extract marker and content.
    
    First tries to extract JSON from ```json ... ``` code block,
    then falls back to parsing the string directly as JSON. so both options are supported.
    """
    
    # Try to extract JSON from markdown code block first
    json_block_pattern = r"```json\s*(\{.*?\})\s*```"
    match = re.search(json_block_pattern, raw_model_answer, re.DOTALL)
    
    if match:
        json_str = match.group(1)
    else:
        # Fall back to parsing the candidate directly
        json_str = raw_model_answer
    try: 
        json_candidate = json.loads(json_str)
    except json.JSONDecodeError:
        raise ValueError(f"Failed to parse model answer: {json_str}")
    if "marker" in json_candidate and "content" in json_candidate:
        marker = json_candidate["marker"]
        content = json_candidate["content"]
        return Property(content=content, marker_name=marker)
    else:
        raise ValueError(f"Failed to parse model answer: {json_str}. model answer must contain 'marker' and 'content' fields")

def syntactic_validation(content: str) -> bool:
    """
    Validate that the candidate invariant is a syntactically correct boolean expression
    without side effects (no assignments or mutations).

    Based on paper requirements:
    - Must be a boolean condition over program state
    - Must not contain assignments (e.g., a += 1, a = 5)
    - Must be safely interpretable as a state predicate
    """
    # Early rejection of empty or whitespace-only strings
    if not content or not content.strip():
        return False

    try:
        import pycparser.c_parser
        from pycparser.c_ast import NodeVisitor

        # Parse the predicate as a C expression wrapped in an assert
        # This allows us to check its AST structure
        wrapped_expr = f"void test() {{ assert({content}); }}"
        ast = pycparser.c_parser.CParser().parse(wrapped_expr)

        # Visitor class to check for assignment operations
        class AssignmentVisitor(NodeVisitor):
            def __init__(self):
                self.has_assignment = False

            def visit_Assignment(self, node):
                self.has_assignment = True
                return False  # Don't traverse children once assignment found

        visitor = AssignmentVisitor()
        visitor.visit(ast)

        # If the predicate contains any assignment, it's invalid
        if visitor.has_assignment:
            return False

        # Reject increment/decrement operators anywhere (they mutate state)
        if "++" in content or "--" in content:
            return False

        return True

    except Exception as e:
        # If parsing fails, the predicate is likely malformed
        # Log the error but return False for safety
        print(f"Warning: Failed to parse predicate '{content}': {e}")
        return False

def validate_model_answer(raw_model_answer: str, target_marker: Optional[str] = None) -> tuple[Property, dict]:
    parsed_model_answer: Optional[Property] = None
    is_valid_dict = {
        "valid_json_format": True,
        "valid_content": False,
        "valid_marker": False,
    }
    try:
        parsed_model_answer = parse_model_answer(raw_model_answer)
    except Exception:
        is_valid_dict["valid_json_format"] = False

    if parsed_model_answer:
        is_valid_dict["valid_content"] = syntactic_validation(parsed_model_answer.content)

    if target_marker:
        # Check marker matches (only if we have a parsed answer)
        is_valid_dict["valid_marker"] = (
            parsed_model_answer is not None 
            and target_marker == parsed_model_answer.marker_name
        )
    else:
        # When not in per-marker mode, marker validation is not required
        is_valid_dict["valid_marker"] = True
    return parsed_model_answer, is_valid_dict


if __name__ == "__main__":
    content = "2*yx - 2*xy - X + 2Y - v + 2*y == 0"
    content2 = "2*y*x - 2*x*y - X + 2*Y - v + 2*y == 0"
    print(syntactic_validation(content))
    print(syntactic_validation(content2))
