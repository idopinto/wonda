## `clean_invariants.py` — how invariant cleaning works (internals + examples)

This repo receives invariants from tools like **UAutomizer**. Those invariants are *valid C boolean expressions*, but they’re often:

- Extremely parenthesis-heavy
- Full of redundant integral casts like `(long long)` / `(unsigned int)`
- Contain tautologies like `x <= x` or constant truths like `0 < 1`

`src/preprocess/clean_invariants.py` is the “string normalizer” that tries to make those expressions smaller and easier to read **without changing meaning** (as much as possible).

---

## What goes in / what comes out

### Input
- A string that is intended to be a **C expression**, e.g.:

```c
((((long long) prime_count + 1) % 4294967296) <= x) && (x <= x)
```

### Output
- Another C expression string, typically simpler:

```c
(prime_count + 1) % 4294967296 <= x
```

---

## The pipeline (high-level)

Internally, cleaning is a 3-step pipeline:

- **Parse** the expression string into a `pycparser` AST
- **Rewrite** the AST (remove casts + prune tautologies/contradictions)
- **Print** the AST back into a C expression string

The main entry point is `clean_invariant(...)`.

---

## Step 1: Parsing (`parse_invariant_expr`)

### Why parsing at all?
If we only used regex/string replacement, we’d risk breaking operator precedence or removing casts incorrectly. Parsing gives us structure:

- `BinaryOp("&&", left=..., right=...)`
- `Cast(to_type=..., expr=...)`
- `UnaryOp("!", expr=...)`
- etc.

### How parsing works
`pycparser` parses **C**, but it expects complete C code. So the file wraps your expression inside a tiny dummy function:

```c
void _temp() { int _dummy = <YOUR_EXPR>; }
```

Then it locates the initializer of `_dummy` and returns it as an AST expression node.

### Important behavior (robustness)
If parsing fails (e.g. the expression isn’t valid C), the code returns `None` and callers fall back to returning the **original string** trimmed.

This matters in preprocessing: a single bad invariant should not crash the whole dataset build.

---

## Step 2: AST rewriting (`clean_invariant_ast`)

This function walks the AST and applies local rewrite rules.

### 2.1 Removing casts
UAutomizer often emits casts like `(long long)x` everywhere. These are usually noise when you only care about the logical condition.

The cleaner supports:

- **`remove_integral_casts=True` (default)**  
  Removes casts where the cast type is integral-ish: `signed/unsigned/char/short/int/long/__int128/_Bool/bool`.
  Pointer casts are preserved.

- **`remove_all_casts=True`**  
  Removes *all* casts (including pointer casts). This can be risky.

#### Example
Input:

```c
((long long) x) + 1 <= (unsigned int) y
```

Output:

```c
x + 1 <= y
```

### 2.2 Removing tautologies and contradictions
When `remove_tautologies=True` (default), the rewrite removes constraints that are always true/false.

#### Tautology examples
- `x <= x` → always true
- `0 < 1` → always true
- constant `1` → treated as boolean true

#### Contradiction examples
- `x < x` → always false
- `1 < 0` → always false
- constant `0` → treated as boolean false

#### How it simplifies `&&` and `||`
These rules are applied locally:

- **AND (`&&`)**
  - `0 && expr` → `0`
  - `1 && expr` → `expr`
  - `taut && expr` → `expr`

- **OR (`||`)**
  - `1 || expr` → `1`
  - `0 || expr` → `expr`
  - `contra || expr` → `expr`

#### Example (tautology removal)
Input:

```c
x <= x && y > 0
```

Output:

```c
y > 0
```

#### Example (contradiction removal)
Input:

```c
0 || (z != z) || (a == b)
```

Output:

```c
a == b
```

### What it *doesn’t* do
This cleaner is intentionally conservative. It does **not** attempt:

- algebraic simplification (e.g. `x + 0` → `x`)
- subsumption (e.g. `x > 0 && x > 1` → `x > 1`)
- deduplication across large conjunctions/disjunctions

It focuses on:
- stripping cast noise
- pruning obvious “always true/false” pieces

---

## Step 3: Printing (normal vs “pretty”)

Once rewritten, the AST is converted back to C source.

There are two printers:

- **`invariant_ast_to_src`**: uses `pycparser`’s default generator (often very parenthesis-heavy)
- **`invariant_ast_to_pretty_src`**: custom printer that uses operator precedence to remove *unnecessary* parentheses

### Precedence (the “why” behind parentheses)

When we “pretty print”, we’re trying to remove parentheses **only when C would evaluate the expression the same way without them**.

That comes down to two rules:

- **Operator precedence**: which operator binds tighter (happens earlier).
- **Associativity**: when two operators have the same precedence, how expressions group.
  - Most binary operators in C are **left-associative** (`a - b - c` means `(a - b) - c`).
  - Assignment and `?:` have different associativity, but `clean_invariants.py` mostly deals with boolean/arithmetic ops.

#### How the code represents precedence
In `src/preprocess/clean_invariants.py`, there’s a table called `_BINOP_PRECEDENCE` that assigns a number to each binary operator:

- **Higher number = binds tighter**
- The *exact* numbers don’t matter — only their **relative ordering** does.

So `"*"` gets a higher number than `"+"` because multiplication binds tighter than addition. Likewise `"&&"` gets a higher number than `"||"` because AND binds tighter than OR.

This table is modeled after C operator precedence (simplified to only the operators we actually see in invariants):

- `||` lowest (10)
- `&&` (20)
- bitwise `|` / `^` / `&`
- comparisons (`==`, `<`, `<=`, …)
- shifts (`<<`, `>>`)
- `+`, `-`
- `*`, `/`, `%` highest among binary ops here

#### The parenthesis rules used by the pretty printer

When printing a child expression inside a parent expression:

- **If child precedence is lower than parent precedence** → we **must** add parentheses.
  - Example: `(a || b) && c` must keep parentheses because `||` is lower precedence than `&&`.

- **If child precedence is higher than parent precedence** → parentheses are not needed.
  - Example: `a + b * c` does not need parentheses around `b * c`.

- **If child and parent have the same precedence**, associativity matters:
  - For many ops, `a op b op c` is safe to print as a flat chain because it groups left.
  - But for **non-associative** ops like `-`, `/`, `%`, shifts, etc., the right side can be dangerous:
    - `a - (b - c)` is **not** the same as `a - b - c`
    - so the printer keeps parentheses when the **right child** is another binary op of the **same precedence**.

That’s exactly why the code has this special case:

```c
// preserve grouping for: a - (b - c), a << (b << c), etc.
if right_child is BinaryOp and prec(right_child) == prec(parent):
    parenthesize(right_child)
```

And for a small set of truly associative operators (`&&`, `||`, `+`, `*`, `&`, `|`, `^`) it safely **flattens chains** like:

```c
((a && b) && c)  ->  a && b && c
```

### Example (pretty printing)
Input:

```c
((a || b) && c)
```

Pretty output:

```c
(a || b) && c
```

But:

```c
(a || (b && c))
```

Pretty output keeps meaning (no extra parentheses needed because `&&` binds tighter than `||`):

```c
a || b && c
```

---

## Public API (what you typically call)

### `clean_invariant(invariant_src, pretty=True, ...)`
Most common entry point. It:

- parses
- cleans casts + tautologies
- prints (pretty if requested)

### `pretty_invariant_src(invariant_src, ...)`
Convenience wrapper that always pretty-prints, and optionally strips casts/tautologies.

---

## Where this is used in the repo

- `src/preprocess/preprocess_gt_invariants.py`
  - when `pipeline.prettify` is enabled, it calls `clean_invariant(..., pretty=True)` to normalize `inv["invariant"]`

- `src/train/sft/data_utils.py`
  - normalizes invariants before building training samples

---

## Bugs / pitfalls found (and fixed)

### 1) Parsing failures could crash preprocessing
Previously, `parse_invariant_expr(...)` did not catch parse exceptions. If `pycparser` couldn’t parse an invariant, the whole script could error out.

**Fix**: parsing is now wrapped in `try/except`, returning `None` on failure. Callers return the original invariant string (trimmed) when parsing fails.

### 2) Trailing semicolons
Some invariants may come with a trailing `;`. That breaks the “expression inside initializer” trick.

**Fix**: we strip a trailing semicolon before parsing.

---

## Practical tips

- If you see the cleaner returning the exact input string, it usually means **pycparser couldn’t parse it**.
- If you need maximum safety, keep `remove_all_casts=False` (default) so pointer casts remain.
- Use `pretty=True` for readability when writing datasets/logs.

