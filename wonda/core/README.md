# `wonda/core` — AST Program & Property

This module provides the foundational representations for working with C programs in the WONDA pipeline: **`AstProgram`** for AST-based preprocessing and marker-based instrumentation, and **`Property`** for location-bound boolean expressions (loop invariants and target assertions).

## Overview

The core idea is to take an SV-COMP-style C program, parse it into an AST, and then instrument it with **markers** — placeholder function calls inserted at semantically meaningful locations (loop heads and the target assertion). These markers serve as anchors: at verification time or LLM inference time, they are replaced with concrete `assume(...)` / `assert(...)` calls carrying the actual property expressions.

```
┌─────────────┐      preprocess       ┌────────────────┐
│  Raw C file  │ ───────────────────▶  │  Cleaned code  │
│  (SV-COMP)   │   gcc -E, rewrite    │  (no externs,  │
└─────────────┘   verifier builtins    │   no includes) │
                                       └───────┬────────┘
                                               │ parse (pycparser)
                                               ▼
                                       ┌────────────────┐
                                       │   Clean AST    │
                                       │  (self.ast)    │
                                       └───────┬────────┘
                                               │
                         ┌─────────────────────┼──────────────────────┐
                         │ replace 1st assert   │                      │
                         │ with TARGET_ASSERT_  │ insert loop markers  │
                         │ MARKER()             │ INVARIANT_MARKER_k() │
                         ▼                      ▼                      │
                  ┌──────────────┐     ┌────────────────┐              │
                  │  Marked AST  │     │   LLM AST      │◄─────────────┘
                  │(self.marked_ │     │ (self.llm_ast)  │  materialize
                  │     ast)     │     │ target assert   │  target back
                  └──────┬───────┘     │ visible as      │  to assert()
                         │             │ assert(target))  │
                         │             └────────┬─────────┘
                         │                      │
                         ▼                      ▼
                  ┌──────────────┐     ┌────────────────┐
                  │ Verifier-    │     │  LLM-facing    │
                  │ facing code  │     │  code          │
                  │ (pp_code,    │     │ (llm_code)     │
                  │  + PATCH)    │     │ nondet → rand  │
                  └──────────────┘     └────────────────┘
```

## Files

| File | Description |
|------|-------------|
| `ast_program.py` | `AstProgram` class — AST parsing, marker insertion, code generation |
| `property.py` | `Property` dataclass — (marker location, boolean expression) pair |

## `Property`

A `Property` binds a boolean expression to a program location identified by a marker name.

```python
@dataclass
class Property:
    content: str                         # e.g. "x >= 0 && y < n"
    marker_name: Optional[str] = None    # e.g. "INVARIANT_MARKER_1"
```

There are two kinds of markers:

| Marker | Meaning | Example |
|--------|---------|---------|
| `INVARIANT_MARKER_k` | Loop head of the k-th loop | `INVARIANT_MARKER_1`, `INVARIANT_MARKER_2` |
| `TARGET_ASSERT_MARKER` | Original location of the target assertion | `TARGET_ASSERT_MARKER` |

Properties are used to express:
- **Target assertions** — the property the program must satisfy (placed at `TARGET_ASSERT_MARKER`)
- **Candidate loop invariants** — expressions that hold at loop heads (placed at `INVARIANT_MARKER_k`)

## `AstProgram`

### Construction & Processing

```python
from wonda.core.ast_program import AstProgram

# From file
program = AstProgram().from_file_path(Path("example.c"))

# From string
program = AstProgram().from_code(c_source_string)

# Process: preprocess → parse → instrument → generate code
pp_code, llm_code = program.process()
```

`process()` performs the full pipeline:

1. **Preprocess** — runs `gcc -E -P` to expand macros, then rewrites SV-COMP builtins (`__VERIFIER_assert` → `assert`, `assume_abort_if_not` → `assume`)
2. **Parse** — builds a `pycparser` AST from the preprocessed code
3. **Prune** — removes helper function definitions (`reach_error`, `__VERIFIER_assert`, `assert`, `assume`, `assume_abort_if_not`) and `extern` declarations
4. **Markerize target** — replaces the first `assert(expr)` with `TARGET_ASSERT_MARKER()` and stores the extracted expression as a `Property`
5. **Insert loop markers** — walks the AST and inserts `INVARIANT_MARKER_k()` at the top of every loop body (`while`, `for`, `do-while`)
6. **Build LLM AST** — a copy of the marked AST where `TARGET_ASSERT_MARKER()` is materialized back to `assert(target_expr)` so the LLM can see the property it needs to help verify

After `process()`, the following attributes are populated:

| Attribute | Type | Description |
|-----------|------|-------------|
| `ast` | `c_ast.FileAST` | Clean AST with target assert replaced by marker |
| `marked_ast` | `c_ast.FileAST` | AST with both target marker and loop markers |
| `llm_ast` | `c_ast.FileAST` | Marked AST with target assert visible |
| `pp_code` | `str` | Generated code from clean AST |
| `llm_code` | `str` | LLM-facing code (`__VERIFIER_nondet_*` → `rand()`) |
| `marked_code_from_ast` | `str` | Code with all markers visible |
| `num_loops` | `int` | Number of loops found |
| `assertions` | `List[Property]` | Extracted target assertion(s) |
| `available_markers` | `List[str]` | All marker names (loop + target) |

### Key Methods

#### `get_target_assert() → Property`
Returns the target assertion extracted during processing.

#### `get_available_markers() → List[str]`
Returns all marker names: `["INVARIANT_MARKER_1", ..., "INVARIANT_MARKER_k", "TARGET_ASSERT_MARKER"]`.

#### `get_program_with_assertion(property_to_assert, assumptions, for_llm) → str`

The central method for generating instrumented programs for verification. It takes a set of properties and produces a complete C program with markers replaced by concrete `assume`/`assert` calls.

```python
# Assert candidate invariant at loop 1 (correctness check)
correctness_program = program.get_program_with_assertion(
    property_to_assert=Property(content="i >= 0", marker_name="INVARIANT_MARKER_1"),
    assumptions=[],
    for_llm=False,
)

# Assert target with invariant assumed (usefulness check)
usefulness_program = program.get_program_with_assertion(
    property_to_assert=program.get_target_assert(),
    assumptions=[Property(content="i >= 0", marker_name="INVARIANT_MARKER_1")],
    for_llm=False,
)
```

Parameters:
- `property_to_assert` — the property to assert (at its marker location)
- `assumptions` — list of properties to assume (each at their respective loop marker)
- `for_llm` — if `True`, emits LLM-facing code (nondet→rand, no PATCH); if `False`, emits verifier-facing code (with PATCH defining `assert`/`assume`)
- `preserve_property_format` — if `True` (default), preserves the original formatting of property expressions

### Internal AST Visitors

| Class | Purpose |
|-------|---------|
| `_LoopMarkerInserter` | Inserts `INVARIANT_MARKER_k()` at the top of `while`/`for`/`do-while` bodies |
| `_DeclarationPruner` | Removes specified function definitions and `extern` declarations |
| `Markerizer` | Replaces the first `assert(...)` call with `TARGET_ASSERT_MARKER()` |
| `Replacer` | Replaces a specific marker with a function call (e.g. `assert(expr)`) |
| `LoopMarkerReplacer` | Replaces loop markers with `assume`/`assert` calls based on a replacement map |
| `Cleaner` | Removes any unresolved markers (replaces with empty statements) |

## How It Fits in the WONDA Pipeline

```
                         ┌──────────────┐
                         │  C Program   │
                         └──────┬───────┘
                                │
                    AstProgram.process()
                                │
                 ┌──────────────┼──────────────┐
                 ▼                             ▼
          ┌─────────────┐              ┌─────────────┐
          │  llm_code   │              │ marked_ast  │
          │(for prompt) │              │(for verif.) │
          └──────┬──────┘              └──────┬──────┘
                 │                            │
                 ▼                            │
          ┌─────────────┐                     │
          │  LLM / SLM  │                     │
          │ generates   │                     │
          │ candidate   │                     │
          │ invariant q │                     │
          └──────┬──────┘                     │
                 │                            │
                 ▼                            ▼
          ┌────────────────────────────────────────┐
          │       get_program_with_assertion()      │
          │                                        │
          │  correctness: assert(q) at loop marker │
          │  usefulness:  assume(q), assert(target) │
          └────────────────────┬───────────────────┘
                               │
                               ▼
                      ┌─────────────────┐
                      │   UAutomizer    │
                      │   (verifier)    │
                      └────────┬────────┘
                               │
                               ▼
                      ┌─────────────────┐
                      │ Decision: TRUE  │
                      │ FALSE, UNKNOWN  │
                      └─────────────────┘
```

1. **Preprocessing** — `AstProgram` parses and instruments the C program
2. **LLM Inference** — the LLM receives `llm_code` (with `rand()` instead of nondet, target assert visible) and proposes a candidate invariant
3. **Verification** — `get_program_with_assertion()` generates two programs:
   - **Correctness query**: `assert(candidate)` at the loop marker (is the invariant valid?)
   - **Usefulness query**: `assume(candidate)` at the loop marker + `assert(target)` (does it help prove the target?)
4. **Decision** — the verifier results are combined by the decision procedure

## Demos

Runnable demos are located in `wonda/demos/`:

### AST Program Demo (no external tools needed)

```bash
uv run python -m wonda.demos.ast_program_demo
```

Creates a sample C program in-memory, processes it through the full `AstProgram` pipeline, and prints each stage of the transformation — from raw code to marked AST to instrumented verification programs. No external files or tools are needed.

### AST Tree Visualization Demo (no external tools needed)

```bash
uv run python -m wonda.demos.ast_tree_demo
```

Visualizes pycparser AST trees as Unicode box-drawing diagrams. Shows how C expressions and programs are represented internally as trees, how operator precedence is encoded in tree structure, and how normalization rewrites (cast stripping, tautology elimination) transform the tree.

### Normalization Demo (no external tools needed)

```bash
uv run python -m wonda.demos.normalization_demo
```

Demonstrates Stage 1 of the WONDA data pipeline: AST-based invariant normalization. Shows how raw verifier-generated invariants are cleaned up by stripping redundant integral casts, eliminating tautologies/contradictions, and minimizing parentheses — all while preserving semantic equivalence.

### Direct Verification Demo (requires UAutomizer + runlim)

```bash
uv run python -m wonda.demos.direct_verification_demo
uv run python -m wonda.demos.direct_verification_demo --timeout 60 --index 2
```

Loads a real evaluation program from the HuggingFace dataset, processes it through `AstProgram`, and runs full end-to-end verification with UAutomizer — including baseline verification, correctness/usefulness queries, and the decision procedure.

### Built-in CLI

```bash
uv run python -m wonda.core.ast_program <file.c>
```
