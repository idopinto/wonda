# Tests

All tests run with **no external tools** (no gcc, UAutomizer, or runlim) and **no network**; they use in-memory fixtures and mocks so they are suitable for CI.

## Run tests

```bash
# Run the full test suite
uv run pytest tests/ -v

# Run a specific test file
uv run pytest tests/test_validate_model_answer.py -v

# Run tests matching a keyword
uv run pytest tests/ -v -k "decision"
```

## What to run

| Command | Description |
|--------|-------------|
| `uv run pytest tests/ -v` | **Recommended.** Run all unit tests (fast, no external deps). |
| `uv run pytest tests/ -v -k "validate or ast_program or decision or uautomizer"` | Shorter run: only validate, ast_program, decision_procedure, and uautomizer tests. |
| `uv run pytest tests/test_gt_invariant_normalization.py -v` | GT invariant normalization (parsing, pretty-printing, casts, parentheses). |
| `uv run pytest tests/test_gt_invariants.py -v` | GT invariant integration (may use more fixtures). |

## Test modules (summary)

| File | Coverage |
|------|----------|
| `test_validate.py` | Syntactic validation of C expressions (boolean, no side effects). |
| `test_validate_complex.py` | Complex expressions, malformed inputs, edge cases. |
| `test_validate_model_answer.py` | `parse_model_answer`, `validate_model_answer` (JSON format, marker, content). |
| `test_ast_program.py` | `nondet_type`, `_rewrite_verifier_builtins`, `AstProgram` (from_code, process, get_program_with_assertion), `Property`. |
| `test_decision_procedure.py` | Decision calculus `compute_decision` (DEC-FALSE, DEC-PROP, DEC-U). |
| `test_uautomizer_parsing.py` | `_parse_runlim_output`, `_parse_uautomizer_output` (no verifier binary). |
| `test_gt_invariant_normalization.py` | GT invariant parsing, AST→src, casts, parentheses. |
| `test_gt_invariants.py` | GT invariant integration / regression tests. |
