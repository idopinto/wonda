# Preprocessing Pipeline Documentation

## Overview

This document explains the AST-based program preprocessing approach, the invariant cleaning pipeline, and how these components work together to prepare data for training.

---

## 1. AST-Based Program Processing (`src/preprocess/program.py`)

### Key Concept: Marker-Based Instrumentation

The new `Program` class uses **pycparser** to build and manipulate C Abstract Syntax Trees (ASTs). Instead of string manipulation, it works at the AST level for reliable transformations.

### Core Workflow

```
Raw C Code → GCC Preprocess → Parse to AST → Transform AST → Generate Code
```

#### Step-by-Step:

1. **Preprocessing** (`preprocess()`):
   - Remove `#include` directives
   - Run `gcc -E -P -nostdinc` to expand macros and strip comments
   - Normalize SV-COMP builtins (`__VERIFIER_assert` → `assert`, etc.)

2. **AST Construction & Pruning** (`process()`):
   - Parse preprocessed code into a `pycparser` AST
   - Remove unwanted function definitions (`reach_error`, `assert`, `assume`, etc.)
   - Remove `extern` declarations

3. **Target Assert Replacement**:
   - Find the first `assert(...)` call in the program
   - Replace it with `TARGET_ASSERT_MARKER()` placeholder
   - Store the original assertion predicate

4. **Loop Marker Insertion** (`_LoopMarkerInserter`):
   - Traverse AST and insert `INVARIANT_MARKER_k()` at the top of every loop body
   - Works for `while`, `for`, and `do-while` loops
   - Each marker gets a unique index (1, 2, 3, ...)

5. **LLM Code Generation**:
   - Create a separate AST for LLM consumption
   - Replace `TARGET_ASSERT_MARKER()` back to `assert(target)`
   - Replace `__VERIFIER_nondet_T()` with `(T) rand()`

### Generated Outputs

| Output | Description |
|--------|-------------|
| `pp_code` | Preprocessed code (no markers) |
| `marked_code_from_ast` | Code with all markers (for verification) |
| `llm_code` | Code with markers + `rand()` rewrites (for LLM prompts) |

### Instrumentation API

```python
program.get_program_with_assertion(
    predicate=Predicate(content="x > 0", marker_name="INVARIANT_MARKER_1"),
    assumptions=[Predicate(content="y >= 0", marker_name="INVARIANT_MARKER_2")],
    for_llm=False  # True for LLM, False for verifier
)
```

- **`predicate`**: The assertion to check (placed at the specified marker)
- **`assumptions`**: Assumptions to add at loop markers (`assume(...)`)
- **`for_llm`**: If `True`, uses `rand()` instead of `__VERIFIER_nondet`; if `False`, adds the verification PATCH header

---

## 2. What Changed from Old `program.py` (`src/utils/program.py`)

### Old Approach (Line-Based / String Manipulation)

The old implementation used **regex and line counting**:

```python
# Old: Track line numbers and positions
self.assertion_points: Dict[int, Set[AssertionPointAttributes]] = {}
self.in_loop: Dict[int, int] = dict()
self.unclosed_brackets: Dict[int, int] = {}
```

**Problems:**
- Fragile: relied on exact string patterns like `stripped_line.split("(")[0] == "assert"`
- Error-prone bracket counting for loop detection
- Hard to maintain with complex C syntax

### New Approach (AST-Based)

The new implementation uses **pycparser AST traversal**:

```python
# New: Work with AST nodes directly
class _LoopMarkerInserter(c_ast.NodeVisitor):
    def visit_While(self, node): ...
    def visit_For(self, node): ...
    def visit_DoWhile(self, node): ...
```

**Improvements:**

| Aspect | Old | New |
|--------|-----|-----|
| Loop detection | Regex + bracket counting | AST `visit_While/For/DoWhile` |
| Assert extraction | `re.search(r"assert\((.*?)\);")` | `_is_call_named(node, "assert")` |
| Code generation | String concatenation | `c_generator.CGenerator().visit(ast)` |
| Marker placement | Line number arithmetic | Insert AST nodes into `Compound.block_items` |
| Format preservation | Lossy | Uses placeholder substitution to preserve original predicate formatting |

### API Differences

```python
# Old: Line numbers + assertion point attributes
program.get_program_with_assertion(
    predicate, assumptions, 
    assertion_points={5: "BeginningOfLoop"},  # Line-based
    forGPT=True
)

# New: Marker names (location-independent)
program.get_program_with_assertion(
    predicate=Predicate(content="...", marker_name="INVARIANT_MARKER_1"),
    assumptions=[...],
    for_llm=True
)
```

---

## 3. Invariant Cleaning Pipeline (`preprocess_gt_invariants.py`)

### Pipeline Overview

```
Load Dataset → For each invariant:
    1. Clean invariant (remove casts, tautologies)
    2. Optionally generalize verbose invariants (LLM)
    3. Verify simplified invariant is correct & useful
    4. Generate CoT reasoning
→ Save preprocessed dataset
```

### Invariant Cleaning (`clean_invariants.py`)

The `clean_invariant()` function applies AST-based transformations:

#### What Gets Removed:

1. **Integral Casts** (default):
   ```c
   // Before:
   ((long long) prime_count + 1) % 4294967296
   // After:
   (prime_count + 1) % 4294967296
   ```
   
   Removed types: `signed`, `unsigned`, `char`, `short`, `int`, `long`, `__int128`, `_Bool`

2. **Tautologies** (always true):
   ```c
   x <= x  →  removed from && chains
   x >= x  →  removed from && chains
   x == x  →  removed from && chains
   0 < 1   →  removed (constant comparison)
   1       →  true constant
   ```

3. **Contradictions** (always false):
   ```c
   x < x   →  0 (in || chains, removed)
   x > x   →  0
   x != x  →  0
   ```

#### Simplification Rules:

| Expression | Result |
|------------|--------|
| `taut && x` | `x` |
| `x && taut` | `x` |
| `contra && x` | `0` |
| `taut \|\| x` | `1` |
| `contra \|\| x` | `x` |

#### Pretty Printing:

The `invariant_ast_to_pretty_src()` function minimizes parentheses using C operator precedence:

```c
// Before (pycparser default):
((((a + b) * c) > d) && ((e || f)))
// After (pretty):
(a + b) * c > d && (e || f)
```
```
            &&  (prec 20)
           /  \
          /    \
         >      ||  (prec 10, but parenthesized)
      (prec 50)  / \
        / \     e   f
       /   \
      *     d
   (prec 70)
     / \
    +   c
 (prec 60)
   / \
  a   b
```
### Generalization Pipeline (Optional)

For verbose invariants (controlled by config):

1. **Filter**: Only process invariants with:
   - `min_disjuncts <= disjunct_count <= max_disjuncts`
   - `min_char_length <= length <= max_char_length`

2. **LLM Generalization**: Use Kimi K2 Thinking to simplify:
   ```
   Input:  (x == 1 || x == 2 || x == 3 || ... || x == 100)
   Output: 1 <= x && x <= 100
   ```

3. **Verification**: Check the simplified invariant with UAutomizer:
   - **Correctness**: `V(P, ∅, simplified)` → Is it an invariant?
   - **Usefulness**: `V(P, {simplified}, target)` → Does it help prove the target?

4. **Fallback**: If verification fails, keep the cleaned (but not LLM-generalized) invariant

### CoT Distillation (Optional)

For each invariant, generate Chain-of-Thought reasoning:
- Uses the program with markers
- Explains why the invariant holds at the target location
- Stored as `cot_reasoning` and `cot_content` fields

---

## Configuration

See `configs/preprocess/preprocess_gt_invariants.yaml`:

```yaml
pipeline:
  generalize_invariants: true  # Use LLM to simplify
  cot_distilation: true        # Generate CoT explanations

filtering:
  min_disjuncts: 5
  max_disjuncts: 100
  min_char_length: 200
  max_char_length: 10000
```

---

## Example Usage

```bash
# Preprocess all invariants
uv run -m src.preprocess.preprocess_gt_invariants

# Process limited subset
uv run -m src.preprocess.preprocess_gt_invariants dataset.limit=10

# Skip LLM generalization (just clean)
uv run -m src.preprocess.preprocess_gt_invariants pipeline.generalize_invariants=false
```
