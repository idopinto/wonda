"""
Utilities for pruning / normalizing invariant strings coming from tools like UAutomizer.

Main use-case:
  - Parse a C-like boolean expression (the invariant) into a `pycparser` AST
  - Rewrite the AST to remove noisy / unnecessary typing casts such as `(long long)`
  - Regenerate a simplified invariant string

Example invariant (from InvBench/UAutomizer):
    ((((prime_count <= x) && ((((long long) prime_count + 1) % 4294967296) <= x)) ...
After stripping casts:
    (((prime_count <= x) && (((prime_count + 1) % 4294967296) <= x)) ...
"""

from __future__ import annotations

import re
from typing import Optional, Tuple
import json
from typing import Any
from pycparser import c_ast, c_generator, c_parser
import hydra
from omegaconf import DictConfig
from loguru import logger
from pathlib import Path
from tqdm import tqdm
import json
from typing import Any

_LIST_CHILD_RE = re.compile(r"^(?P<field>[A-Za-z_]\w*)\[(?P<idx>\d+)\]$")


def parse_invariant_expr(
    expr_src: str, *, parser: Optional[c_parser.CParser] = None
) -> c_ast.Node:
    """
    Parse an invariant string (a C expression) into a `pycparser` expression node.

    Notes:
    - This is **syntactic** parsing only; identifiers don't need to be declared.
    - We parse by wrapping the expression inside a dummy function, similar to
      `Program._parse_expr` in `src/preprocess/program.py`.
    - If parsing fails, returns `c_ast.ID(expr_src)` as a best-effort placeholder.
    """
    p = parser or c_parser.CParser()
    wrapped = f"void _temp() {{ int _dummy = {expr_src}; }}"
    temp_ast = p.parse(wrapped)
    for ext in temp_ast.ext:
        if isinstance(ext, c_ast.FuncDef):
            for stmt in ext.body.block_items or []:
                if isinstance(stmt, c_ast.Decl) and stmt.init is not None:
                    return stmt.init
    return None


def invariant_ast_to_src(expr_ast: c_ast.Node) -> str:
    """Convert a `pycparser` expression node back to C source."""
    return c_generator.CGenerator().visit(expr_ast)


def invariant_ast_to_pretty_src(expr_ast: c_ast.Node) -> str:
    """
    Convert a `pycparser` expression node back to C source with *minimal parentheses*.

    `pycparser`'s default generator can be quite parenthesis-heavy for complex expressions.
    This printer uses C operator precedence/associativity and only inserts parentheses
    when they are required to preserve meaning.
    """

    def prec(node: c_ast.Node) -> int:
        # Higher number = binds tighter.
        if isinstance(node, (c_ast.ID, c_ast.Constant)):
            return 100
        if isinstance(node, (c_ast.FuncCall, c_ast.ArrayRef, c_ast.StructRef)):
            return 90
        if isinstance(node, (c_ast.UnaryOp, c_ast.Cast)):
            return 80
        if isinstance(node, c_ast.TernaryOp):
            return 25
        if isinstance(node, c_ast.BinaryOp):
            return _BINOP_PRECEDENCE.get(node.op, 0)
        return 0

    def is_associative(op: str) -> bool:
        # We only use this to safely flatten chains like a && b && c.
        return op in {"&&", "||", "+", "*", "&", "|", "^"}

    def emit(node: c_ast.Node, parent_op: Optional[str] = None, side: str = "N") -> str:
        """
        side: 'L' if this node is the left child of a parent BinaryOp,
              'R' if right child,
              'N' for none/top-level/other contexts.
        """
        if isinstance(node, c_ast.ID):
            return node.name
        if isinstance(node, c_ast.Constant):
            return node.value
        if isinstance(node, c_ast.FuncCall):
            name = emit(node.name)
            args = ""
            if node.args and getattr(node.args, "exprs", None):
                args = ", ".join(emit(a) for a in node.args.exprs)
            return f"{name}({args})"
        if isinstance(node, c_ast.ArrayRef):
            return f"{emit(node.name)}[{emit(node.subscript)}]"
        if isinstance(node, c_ast.StructRef):
            # '.' or '->'
            return f"{emit(node.name)}{node.type}{emit(node.field)}"
        if isinstance(node, c_ast.UnaryOp):
            op = node.op
            expr_s = emit(node.expr)
            if prec(node.expr) < prec(node):
                expr_s = f"({expr_s})"
            # pycparser uses "p++"/"p--" for postfix in UnaryOp as well
            if op in {"p++", "p--"}:
                return f"{expr_s}{op[1:]}"
            return f"{op}{expr_s}"
        if isinstance(node, c_ast.Cast):
            # Keep casts that survived stripping (e.g., pointer casts if configured).
            ty = c_generator.CGenerator().visit(node.to_type)
            expr_s = emit(node.expr)
            if prec(node.expr) < prec(node):
                expr_s = f"({expr_s})"
            return f"({ty}){expr_s}"
        if isinstance(node, c_ast.TernaryOp):
            c_s = emit(node.cond)
            if prec(node.cond) < prec(node):
                c_s = f"({c_s})"
            ift = emit(node.iftrue)
            iff = emit(node.iffalse)
            return f"{c_s} ? {ift} : {iff}"
        if isinstance(node, c_ast.BinaryOp):
            op = node.op
            p = prec(node)

            # For associative ops (&&, ||, +, ...), flatten chains to reduce noise.
            if is_associative(op):

                def collect_operands(n: c_ast.Node) -> list[c_ast.Node]:
                    if isinstance(n, c_ast.BinaryOp) and n.op == op:
                        return collect_operands(n.left) + collect_operands(n.right)
                    return [n]

                parts = collect_operands(node)
                rendered_parts: list[str] = []
                for part in parts:
                    s = emit(part)
                    if prec(part) < p:
                        s = f"({s})"
                    rendered_parts.append(s)
                expr_s = f" {op} ".join(rendered_parts)
            else:
                left_s = emit(node.left)
                right_s = emit(node.right)

                if prec(node.left) < p:
                    left_s = f"({left_s})"
                if prec(node.right) < p:
                    right_s = f"({right_s})"
                # Preserve grouping for cases like a - (b - c) or a << (b << c)
                if isinstance(node.right, c_ast.BinaryOp) and prec(node.right) == p:
                    right_s = f"({right_s})"

                expr_s = f"{left_s} {op} {right_s}"

            # Parenthesis relative to parent operator (to preserve precedence).
            if parent_op is not None:
                parent_p = _BINOP_PRECEDENCE.get(parent_op, 0)
                if p < parent_p:
                    return f"({expr_s})"
                # Same precedence: if we are the right child of a left-associative parent,
                # keep parens unless the parent is associative and same op (which we flattened).
                if p == parent_p and side == "R" and parent_op != op:
                    return f"({expr_s})"
            return expr_s

        # Fallback to pycparser's generator for rare/unhandled nodes.
        return c_generator.CGenerator().visit(node)

    return emit(expr_ast)


_BINOP_PRECEDENCE: dict[str, int] = {
    "||": 10,
    "&&": 20,
    "|": 30,
    "^": 35,
    "&": 40,
    "==": 45,
    "!=": 45,
    "<": 50,
    "<=": 50,
    ">": 50,
    ">=": 50,
    "<<": 55,
    ">>": 55,
    "+": 60,
    "-": 60,
    "*": 70,
    "/": 70,
    "%": 70,
}


def _nodes_are_equal(n1: c_ast.Node, n2: c_ast.Node) -> bool:
    """Check if two AST nodes are syntactically identical."""
    if type(n1) != type(n2):
        return False
    if isinstance(n1, c_ast.ID):
        return n1.name == n2.name
    if isinstance(n1, c_ast.Constant):
        return n1.value == n2.value and n1.type == n2.type
    if isinstance(n1, c_ast.BinaryOp):
        return (n1.op == n2.op and 
                _nodes_are_equal(n1.left, n2.left) and 
                _nodes_are_equal(n1.right, n2.right))
    if isinstance(n1, c_ast.UnaryOp):
        return n1.op == n2.op and _nodes_are_equal(n1.expr, n2.expr)
    # For complex nodes, fall back to string comparison
    return invariant_ast_to_src(n1) == invariant_ast_to_src(n2)


def _try_eval_constant(node: c_ast.Node) -> Optional[int]:
    """Try to evaluate a node as an integer constant. Returns None if not a constant."""
    if isinstance(node, c_ast.Constant):
        try:
            # Handle hex, octal, etc.
            val = node.value.rstrip('uUlL')
            if val.startswith('0x') or val.startswith('0X'):
                return int(val, 16)
            elif val.startswith('0') and len(val) > 1 and val[1:].isdigit():
                return int(val, 8)
            return int(val)
        except ValueError:
            return None
    if isinstance(node, c_ast.UnaryOp) and node.op == '-':
        inner = _try_eval_constant(node.expr)
        if inner is not None:
            return -inner
    return None


def _is_tautology(node: c_ast.Node) -> bool:
    """
    Check if a node is a tautology (always true).
    
    Examples: x <= x, x >= x, x == x, 0 < 1
    
    Note: We only treat constant `1` as tautology, not arbitrary non-zero numbers
    like `10`, since those are likely arithmetic values, not boolean flags.
    """
    # Only treat explicit `1` as boolean true (not arbitrary non-zero like 10)
    if isinstance(node, c_ast.Constant):
        val = _try_eval_constant(node)
        return val == 1
    
    if isinstance(node, c_ast.BinaryOp):
        op = node.op
        
        # x <= x, x >= x, x == x are always true
        if op in {'<=', '>=', '=='} and _nodes_are_equal(node.left, node.right):
            return True
        
        # Constant comparisons: 0 < 1, 5 <= 10, etc.
        left_val = _try_eval_constant(node.left)
        right_val = _try_eval_constant(node.right)
        if left_val is not None and right_val is not None:
            if op == '<':
                return left_val < right_val
            elif op == '<=':
                return left_val <= right_val
            elif op == '>':
                return left_val > right_val
            elif op == '>=':
                return left_val >= right_val
            elif op == '==':
                return left_val == right_val
            elif op == '!=':
                return left_val != right_val
    
    return False


def _is_contradiction(node: c_ast.Node) -> bool:
    """
    Check if a node is a contradiction (always false).
    
    Examples: x < x, x > x, x != x, 1 < 0, 0
    """
    # Constant 0
    if isinstance(node, c_ast.Constant):
        val = _try_eval_constant(node)
        return val == 0
    
    if isinstance(node, c_ast.BinaryOp):
        op = node.op
        
        # x < x, x > x, x != x are always false
        if op in {'<', '>', '!='} and _nodes_are_equal(node.left, node.right):
            return True
        
        # Constant comparisons that are false
        left_val = _try_eval_constant(node.left)
        right_val = _try_eval_constant(node.right)
        if left_val is not None and right_val is not None:
            if op == '<':
                return not (left_val < right_val)
            elif op == '<=':
                return not (left_val <= right_val)
            elif op == '>':
                return not (left_val > right_val)
            elif op == '>=':
                return not (left_val >= right_val)
            elif op == '==':
                return not (left_val == right_val)
            elif op == '!=':
                return not (left_val != right_val)
    
    return False


def _make_true() -> c_ast.Constant:
    """Create a constant node representing true (1)."""
    return c_ast.Constant(type='int', value='1')


def _make_false() -> c_ast.Constant:
    """Create a constant node representing false (0)."""
    return c_ast.Constant(type='int', value='0')


def clean_invariant(
    invariant_src: str,
    *,
    remove_all_casts: bool = False,
    remove_integral_casts: bool = True,
    remove_tautologies: bool = True,
    pretty: bool = False,
    parser: Optional[c_parser.CParser] = None,
    print_ast: bool = False,
) -> str:
    """
    Parse `invariant_src`, remove noisy typing casts and tautologies, and return simplified C source.

    By default we remove *integral* casts (e.g., `(long long)`, `(unsigned int)`),
    but we keep pointer casts unless `remove_all_casts=True`.
    
    We also remove tautological constraints like `x <= x`, `0 < 1`, etc.
    """
    ast = parse_invariant_expr(invariant_src, parser=parser)
    if ast:
        if print_ast:
            print(ast)
        new_ast = clean_invariant_ast(
            ast,
            remove_all_casts=remove_all_casts,
            remove_integral_casts=remove_integral_casts,
            remove_tautologies=remove_tautologies,
        )
        return (
            invariant_ast_to_pretty_src(new_ast)
            if pretty
            else invariant_ast_to_src(new_ast)
        )
    else:
        return "ERROR: Could not parse invariant"


def pretty_invariant_src(
    invariant_src: str,
    *,
    strip_casts: bool = True,
    remove_all_casts: bool = False,
    remove_integral_casts: bool = True,
    remove_tautologies: bool = True,
    parser: Optional[c_parser.CParser] = None,
) -> str:
    """
    Convenience helper:
      parse -> (optionally) strip casts/tautologies -> pretty-print with minimal parentheses.
    """
    ast = parse_invariant_expr(invariant_src, parser=parser)
    if strip_casts or remove_tautologies:
        ast = clean_invariant_ast(
            ast,
            remove_all_casts=remove_all_casts,
            remove_integral_casts=remove_integral_casts if strip_casts else False,
            remove_tautologies=remove_tautologies,
        )
    return invariant_ast_to_pretty_src(ast)


def clean_invariant_ast(
    expr_ast: c_ast.Node,
    *,
    remove_all_casts: bool = False,
    remove_integral_casts: bool = True,
    remove_tautologies: bool = True,
) -> c_ast.Node:
    """
    Rewrite pass over a `pycparser` expression AST that removes cast nodes and tautologies.

    This is conservative by default:
    - **Integral casts** are removed (common UAutomizer noise), e.g. `(long long)x`
    - **Pointer casts** are preserved unless `remove_all_casts=True`
    - **Tautologies** like `x <= x`, `0 < 1` are removed from conjunctions
    - **Contradictions** like `x < x` in disjunctions are removed
    """

    def should_strip_cast(node: c_ast.Cast) -> bool:
        if remove_all_casts:
            return True
        if remove_integral_casts and _is_integral_typename(node.to_type):
            return True
        return False

    def transform(node: c_ast.Node) -> c_ast.Node:
        # First, recursively transform children (post-order rewrite).
        for child_name, child in node.children():
            new_child = transform(child)
            _set_child(node, child_name, new_child)

        # Strip casts
        if isinstance(node, c_ast.Cast) and should_strip_cast(node):
            return node.expr
        
        # Remove tautologies and contradictions
        if remove_tautologies:
            # Handle && with tautologies: (taut && x) -> x, (x && taut) -> x
            # Handle && with contradictions: (contra && x) -> 0
            if isinstance(node, c_ast.BinaryOp) and node.op == '&&':
                left_taut = _is_tautology(node.left)
                right_taut = _is_tautology(node.right)
                left_contra = _is_contradiction(node.left)
                right_contra = _is_contradiction(node.right)
                
                # Contradiction in && makes whole thing false
                if left_contra or right_contra:
                    return _make_false()
                # Remove tautologies from &&
                if left_taut and right_taut:
                    return _make_true()
                if left_taut:
                    return node.right
                if right_taut:
                    return node.left
            
            # Handle || with contradictions: (contra || x) -> x, (x || contra) -> x
            # Handle || with tautologies: (taut || x) -> 1
            if isinstance(node, c_ast.BinaryOp) and node.op == '||':
                left_taut = _is_tautology(node.left)
                right_taut = _is_tautology(node.right)
                left_contra = _is_contradiction(node.left)
                right_contra = _is_contradiction(node.right)
                
                # Tautology in || makes whole thing true
                if left_taut or right_taut:
                    return _make_true()
                # Remove contradictions from ||
                if left_contra and right_contra:
                    return _make_false()
                if left_contra:
                    return node.right
                if right_contra:
                    return node.left
            
            # Standalone tautology -> 1, standalone contradiction -> 0
            if _is_tautology(node):
                return _make_true()
            if _is_contradiction(node):
                return _make_false()
        
        return node

    return transform(expr_ast)


def _set_child(parent: c_ast.Node, child_name: str, new_child: c_ast.Node) -> None:
    """
    Assign `new_child` into `parent` at the location described by `child_name`.

    `pycparser` names list children like `block_items[0]`, `exprs[3]`.
    """
    m = _LIST_CHILD_RE.match(child_name)
    if m:
        field = m.group("field")
        idx = int(m.group("idx"))
        lst = getattr(parent, field)
        if lst is not None:
            lst[idx] = new_child
        return

    setattr(parent, child_name, new_child)


def _is_integral_typename(tn: c_ast.Typename) -> bool:
    """
    Return True for casts like:
      (long long) x, (unsigned int) y, (signed char) z, (_Bool) b
    and False for pointer casts, structs, etc.
    """
    # Pointer casts: Typename(type=PtrDecl(...))
    if isinstance(tn.type, c_ast.PtrDecl):
        return False

    # Simple identifier types: Typename(type=TypeDecl(type=IdentifierType(names=[...])))
    names = _extract_identifier_type_names(tn.type)
    if not names:
        return False

    integral_atoms = {
        "signed",
        "unsigned",
        "char",
        "short",
        "int",
        "long",
        "__int128",
        "_Bool",
        "bool",
    }
    return all(n in integral_atoms for n in names)


def _extract_identifier_type_names(type_node: c_ast.Node) -> Optional[Tuple[str, ...]]:
    """
    Extract IdentifierType names out of a TypeDecl chain.
    Returns e.g. ('long','long') for 'long long'.
    """
    if isinstance(type_node, c_ast.TypeDecl):
        inner = type_node.type
        if isinstance(inner, c_ast.IdentifierType):
            return tuple(inner.names)
    return None


def some_tests():
    # Example 1: UAutomizer-style invariant (lots of casts + parentheses)
    inv_uautomizer = (
        "((((prime_count <= x) && ((((long long) prime_count + 1) % 4294967296) <= x)) "
        "&& ((((long long) 2 + prime_count) % 4294967296) <= ((long long) n + 1))) "
        "|| ((prime_count == 0) && (n == 0)))"
    )

    print("=== Original (UAutomizer) ===")
    print(inv_uautomizer)
    print()

    print("=== Strip casts (keep parentheses as emitted by generator) ===")
    print(clean_invariant(inv_uautomizer))
    print()

    print("=== Pretty (minimal parentheses) + strip casts ===")
    print(pretty_invariant_src(inv_uautomizer, strip_casts=True))
    print()

    # Example 2: __int128 / unsigned __int128 casts
    inv_int128 = "(((__int128) x + 1) > 0) && ((unsigned __int128) y >= 0)"

    print("=== Original (__int128) ===")
    print(inv_int128)
    print()

    print("=== Strip casts + pretty ===")
    print(clean_invariant(inv_int128, pretty=True))

    another_inv = "(((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((sum <= ((long long) 2 * threshold)) && (1 <= b)) && (((long long) b * 43) <= sum)) && (steps <= 43)) && (threshold <= 100)) || (((((((long long) b * 15) <= sum) && (((long long) 2 + sum) <= ((long long) 2 * threshold))) && (steps <= 15)) && (1 <= b)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (1 <= b)) && (((long long) b * 45) <= sum)) && (steps <= 45)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (((long long) b * 17) <= sum)) && (1 <= b)) && (steps <= 17)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (steps <= 61)) && (1 <= b)) && (((long long) b * 61) <= sum)) && (threshold <= 100))) || (((((((long long) 2 + sum) <= ((long long) 2 * threshold)) && (steps <= 9)) && (1 <= b)) && (((long long) 9 * b) <= sum)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (((long long) 40 * b) <= sum)) && (1 <= b)) && (steps <= 40)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (steps <= 67)) && (1 <= b)) && (((long long) 67 * b) <= sum)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (steps <= 944)) && (((long long) b * 85) <= sum)) && (1 <= b)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (1 <= b)) && (((long long) 56 * b) <= sum)) && (steps <= 56)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (((long long) b * 70) <= sum)) && (steps <= 70)) && (1 <= b)) && (threshold <= 100))) || (((((((long long) 2 + sum) <= ((long long) 2 * threshold)) && (((long long) 5 * b) <= sum)) && (1 <= b)) && (steps <= 5)) && (threshold <= 100))) || ((((steps <= 0) && (0 <= sum)) && (1 <= b)) && (threshold <= 100))) || (((((((long long) 7 * b) <= sum) && (1 <= b)) && (steps <= 7)) && ((((long long) 6 * sum) + 7) <= ((long long) 7 * threshold))) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (steps <= 936)) && (1 <= b)) && (threshold <= 100)) && (((long long) 77 * b) <= sum))) || (((((((long long) 2 + sum) <= ((long long) 2 * threshold)) && (1 <= b)) && (steps <= 6)) && (((long long) 6 * b) <= sum)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (steps <= 24)) && (1 <= b)) && (((long long) b * 24) <= sum)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (((long long) b * 39) <= sum)) && (1 <= b)) && (steps <= 39)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (steps <= 948)) && (1 <= b)) && (((long long) 89 * b) <= sum)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (1 <= b)) && (((long long) b * 28) <= sum)) && (steps <= 28)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (((long long) b * 33) <= sum)) && (1 <= b)) && (steps <= 33)) && (threshold <= 100))) || ((((((((long long) 2 + sum) <= ((long long) 2 * threshold)) && (1 <= b)) && (steps <= 8)) && (8 <= threshold)) && (threshold <= 100)) && (((long long) b * 8) <= sum))) || (((((sum <= ((long long) 2 * threshold)) && (steps <= 947)) && (1 <= b)) && (((long long) 88 * b) <= sum)) && (threshold <= 100))) || (((((((long long) 2 + sum) <= ((long long) 2 * threshold)) && (1 <= b)) && (((long long) 2 * b) <= sum)) && (steps <= 2)) && (threshold <= 100))) || (((((steps <= 10) && (((long long) 2 + sum) <= ((long long) 2 * threshold))) && (1 <= b)) && (((long long) b * 10) <= sum)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (steps <= 942)) && (((long long) b * 83) <= sum)) && (1 <= b)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (steps <= 950)) && (1 <= b)) && (((long long) 91 * b) <= sum)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (((long long) 52 * b) <= sum)) && (steps <= 52)) && (1 <= b)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (1 <= b)) && (((long long) b * 75) <= sum)) && (steps <= 934)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (((long long) b * 99) <= sum)) && (1 <= b)) && (steps <= 958)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (steps <= 952)) && (1 <= b)) && (((long long) b * 93) <= sum)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (((long long) b * 74) <= sum)) && (steps <= 933)) && (1 <= b)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (1 <= b)) && (((long long) b * 18) <= sum)) && (steps <= 18)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (steps <= 946)) && (1 <= b)) && (threshold <= 100)) && (((long long) 87 * b) <= sum))) || (((((sum <= ((long long) 2 * threshold)) && (((long long) b * 73) <= sum)) && (1 <= b)) && (threshold <= 100)) && (steps <= 932))) || (((((sum <= ((long long) 2 * threshold)) && (1 <= b)) && (((long long) b * 27) <= sum)) && (steps <= 27)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (steps <= 21)) && (1 <= b)) && (((long long) 21 * b) <= sum)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (1 <= b)) && (((long long) 30 * b) <= sum)) && (steps <= 30)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (steps <= 59)) && (1 <= b)) && (((long long) 59 * b) <= sum)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (1 <= b)) && (((long long) b * 58) <= sum)) && (steps <= 58)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (steps <= 931)) && (1 <= b)) && (((long long) 72 * b) <= sum)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (steps <= 60)) && (1 <= b)) && (((long long) b * 60) <= sum)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (steps <= 941)) && (1 <= b)) && (((long long) 82 * b) <= sum)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (1 <= b)) && (((long long) 71 * b) <= sum)) && (steps <= 930)) && (threshold <= 100))) || ((((((((long long) 33 * sum) + 34) <= ((long long) threshold * 34)) && (((long long) b * 34) <= sum)) && (1 <= b)) && (steps <= 34)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (((long long) 64 * b) <= sum)) && (1 <= b)) && (threshold <= 100)) && (steps <= 64))) || (((((sum <= ((long long) 2 * threshold)) && (steps <= 35)) && (1 <= b)) && (((long long) 35 * b) <= sum)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (steps <= 957)) && (((long long) 98 * b) <= sum)) && (1 <= b)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (((long long) b * 25) <= sum)) && (1 <= b)) && (steps <= 25)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (((long long) b * 50) <= sum)) && (1 <= b)) && (steps <= 50)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (1 <= b)) && (((long long) 62 * b) <= sum)) && (steps <= 62)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (steps <= 935)) && (1 <= b)) && (((long long) 76 * b) <= sum)) && (threshold <= 100))) || (((((((long long) 2 + sum) <= ((long long) 2 * threshold)) && (steps <= 13)) && (1 <= b)) && (((long long) b * 13) <= sum)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (((long long) 97 * b) <= sum)) && (1 <= b)) && (steps <= 956)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (steps <= 951)) && (1 <= b)) && (((long long) 92 * b) <= sum)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (1 <= b)) && (steps <= 65)) && (threshold <= 100)) && (((long long) b * 65) <= sum))) || (((((sum <= ((long long) 2 * threshold)) && (1 <= b)) && (b <= sum)) && (steps <= 1)) && (threshold <= 100))) || (((((((long long) 2 + sum) <= ((long long) 2 * threshold)) && (steps <= 11)) && (((long long) b * 11) <= sum)) && (1 <= b)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (steps <= 959)) && (1 <= b)) && (((long long) b * 100) <= sum)) && (threshold <= 100))) || (((((((long long) 2 + sum) <= ((long long) 2 * threshold)) && (steps <= 12)) && (1 <= b)) && (((long long) b * 12) <= sum)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (1 <= b)) && (((long long) b * 48) <= sum)) && (steps <= 48)) && (threshold <= 100))) || ((((((69 + ((long long) 68 * sum)) <= ((long long) 69 * threshold)) && (1 <= b)) && (((long long) 69 * b) <= sum)) && (steps <= 69)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (steps <= 63)) && (((long long) b * 63) <= sum)) && (1 <= b)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (steps <= 29)) && (((long long) 29 * b) <= sum)) && (1 <= b)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (1 <= b)) && (((long long) 49 * b) <= sum)) && (steps <= 49)) && (threshold <= 100))) || (((((((long long) 32 * b) <= sum) && (sum <= ((long long) 2 * threshold))) && (steps <= 32)) && (1 <= b)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (1 <= b)) && (steps <= 66)) && (((long long) 66 * b) <= sum)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (((long long) 41 * b) <= sum)) && (1 <= b)) && (steps <= 41)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (1 <= b)) && (steps <= 68)) && (((long long) 68 * b) <= sum)) && (threshold <= 100))) || (((((((long long) 2 + sum) <= ((long long) 2 * threshold)) && (((long long) b * 4) <= sum)) && (1 <= b)) && (steps <= 4)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (1 <= b)) && (((long long) b * 80) <= sum)) && (steps <= 939)) && (threshold <= 100))) || (((((((long long) 2 + sum) <= ((long long) 2 * threshold)) && (1 <= b)) && (((long long) 14 * b) <= sum)) && (steps <= 14)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (steps <= 55)) && (1 <= b)) && (((long long) b * 55) <= sum)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (((long long) 84 * b) <= sum)) && (1 <= b)) && (steps <= 943)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (steps <= 38)) && (1 <= b)) && (((long long) b * 38) <= sum)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (steps <= 54)) && (1 <= b)) && (((long long) b * 54) <= sum)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (1 <= b)) && (steps <= 945)) && (threshold <= 100)) && (((long long) b * 86) <= sum))) || (((((sum <= ((long long) 2 * threshold)) && (steps <= 46)) && (1 <= b)) && (((long long) 46 * b) <= sum)) && (threshold <= 100))) || (((((((long long) 2 + sum) <= ((long long) 2 * threshold)) && (steps <= 16)) && (1 <= b)) && (((long long) 16 * b) <= sum)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (((long long) b * 26) <= sum)) && (1 <= b)) && (steps <= 26)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (steps <= 53)) && (1 <= b)) && (((long long) 53 * b) <= sum)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (steps <= 57)) && (1 <= b)) && (((long long) 57 * b) <= sum)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (1 <= b)) && (((long long) 81 * b) <= sum)) && (steps <= 940)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (1 <= b)) && (steps <= 953)) && (((long long) 94 * b) <= sum)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (((long long) 42 * b) <= sum)) && (1 <= b)) && (steps <= 42)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (steps <= 938)) && (((long long) b * 79) <= sum)) && (1 <= b)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (((long long) 23 * b) <= sum)) && (1 <= b)) && (steps <= 23)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (1 <= b)) && (((long long) b * 96) <= sum)) && (threshold <= 100)) && (steps <= 955))) || (((((steps <= 949) && (sum <= ((long long) 2 * threshold))) && (((long long) b * 90) <= sum)) && (1 <= b)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (steps <= 47)) && (((long long) 47 * b) <= sum)) && (1 <= b)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (steps <= 36)) && (((long long) b * 36) <= sum)) && (1 <= b)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (1 <= b)) && (steps <= 31)) && (((long long) 31 * b) <= sum)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (1 <= b)) && (steps <= 19)) && (threshold <= 100)) && (((long long) 19 * b) <= sum))) || (((((sum <= ((long long) 2 * threshold)) && (1 <= b)) && (((long long) 51 * b) <= sum)) && (threshold <= 100)) && (steps <= 51))) || (((((((long long) 3 * b) <= sum) && (((long long) 2 + sum) <= ((long long) 2 * threshold))) && (1 <= b)) && (steps <= 3)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (steps <= 937)) && (1 <= b)) && (((long long) b * 78) <= sum)) && (threshold <= 100))) || (((((steps <= 44) && (sum <= ((long long) 2 * threshold))) && (1 <= b)) && (((long long) 44 * b) <= sum)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (steps <= 37)) && (1 <= b)) && (((long long) b * 37) <= sum)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (steps <= 954)) && (1 <= b)) && (((long long) 95 * b) <= sum)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (((long long) 22 * b) <= sum)) && (1 <= b)) && (steps <= 22)) && (threshold <= 100))) || (((((sum <= ((long long) 2 * threshold)) && (((long long) b * 20) <= sum)) && (1 <= b)) && (steps <= 20)) && (threshold <= 100)))"
    print("=== Original (__int128) ===")
    print(f"Length: {len(another_inv)}  ")
    print(another_inv)
    print()
    print("=== Strip casts + pretty ===")
    print(f"Length: {len(clean_invariant(another_inv, pretty=True))}")
    print(clean_invariant(another_inv, pretty=True))

    # Test cases where parentheses MUST be preserved
    print("\n" + "=" * 60)
    print("=== PARENTHESES PRESERVATION TESTS ===")
    print("=" * 60)

    paren_tests = [
        # Arithmetic precedence - parens needed
        ("((a + b) * c)", "(a + b) * c", "Addition before multiplication"),
        ("(a / (b * c))", "a / (b * c)", "Division with grouped multiply"),
        ("(a - (b - c))", "a - (b - c)", "Right-associative subtraction"),
        ("(a % (b + c))", "a % (b + c)", "Modulo with grouped addition"),
        # Arithmetic precedence - parens NOT needed
        ("((a * b) + c)", "a * b + c", "Multiplication already higher precedence"),
        ("(a + (b * c))", "a + b * c", "Multiplication already higher precedence"),
        ("((a + b) + c)", "a + b + c", "Left-associative addition"),
        # Logical precedence - parens needed (|| vs &&)
        ("((a || b) && c)", "(a || b) && c", "OR before AND"),
        ("(a && (b || c))", "a && (b || c)", "OR grouped in AND"),
        # Logical precedence - parens NOT needed
        ("((a && b) || c)", "a && b || c", "AND already higher precedence"),
        ("(a || (b && c))", "a || b && c", "AND already higher precedence"),
        # Comparison with arithmetic
        ("((a + 1) > b)", "a + 1 > b", "Addition already higher than comparison"),
        ("(a > (b - 1))", "a > b - 1", "Subtraction already higher than comparison"),
        # Negation
        ("(!(a && b))", "!(a && b)", "Negation of conjunction - needs parens"),
        ("((!a) && b)", "!a && b", "Negation binds tighter - no parens needed"),
        # Mixed - complex cases
        ("((a + b) * (c + d))", "(a + b) * (c + d)", "Both sides need parens"),
        (
            "((a || b) && (c || d))",
            "(a || b) && (c || d)",
            "Both OR need parens in AND",
        ),
    ]

    for original, expected, description in paren_tests:
        result = clean_invariant(original, pretty=True, print_ast=False)
        status = "✅" if result == expected else "❌"
        print(f"\n{status} {description}")
        print(f"   Input:    {original}")
        print(f"   Expected: {expected}")
        print(f"   Got:      {result}")


def main() -> None:
    some_tests()
