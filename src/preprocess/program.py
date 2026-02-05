from __future__ import annotations

import copy
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pycparser import c_ast, c_generator, c_parser

from src.preprocess.predicate import Predicate

PATCH = (
    "void assert(int cond) { if (!(cond)) { ERROR : { reach_error(); abort(); } } }\n"
    "void assume(int cond) { if (!cond) { abort(); } }\n"
)

INVARIANT_MARKER_PREFIX = "INVARIANT_MARKER_"
TARGET_ASSERT_MARKER = "TARGET_ASSERT_MARKER"


def _gcc_preprocess(code: str) -> str:
    """Preprocess code with gcc to remove comments/macros/includes expansion noise."""
    # Remove includes before GCC preprocessing to avoid macro expansion
    code = re.sub(r'^\s*#include\s+[<"].*?[>"]\s*$', "", code, flags=re.MULTILINE)

    fd, path = tempfile.mkstemp(suffix=".c", prefix="gcc_preprocess_")
    os.close(fd)
    tmp_file = Path(path)
    try:
        tmp_file.write_text(code)
        command = f"gcc -E -P -nostdinc {tmp_file}"
        p = subprocess.run(command.split(), capture_output=True)
        if p.returncode != 0:
            stderr = p.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"GCC preprocessing failed: {stderr}")
        return p.stdout.decode("utf-8", errors="replace")
    finally:
        if tmp_file.exists():
            tmp_file.unlink()


def _rewrite_verifier_builtins(code: str) -> str:
    """Normalize SV-COMP verifier intrinsics to assert/assume and strip GCC extensions."""
    code = re.sub(r"__attribute__\s*\(\([^)]*\)\)", "", code)
    code = re.sub(r"printf\s*\([^)]*\)\s*;", "", code)
    code = re.sub(r"__extension__\s*", "", code)

    code = code.replace("__VERIFIER_assert", "assert")
    code = code.replace("assume_abort_if_not", "assume")
    code = re.sub(r"\n\s*\n\s*\n+", "\n\n", code).strip()
    return code


def _is_call_named(node: Any, name: str) -> bool:
    """Check if a C AST node is a function call with the specified name.

    Args:
        node: A C AST node to check
        name: The function name to match against

    Returns:
        True if the node is a FuncCall node with the given name, False otherwise
    """
    return (
        isinstance(node, c_ast.FuncCall)
        and isinstance(node.name, c_ast.ID)
        and node.name.name == name
    )


class _LoopMarkerInserter(c_ast.NodeVisitor):
    """Insert INVARIANT_MARKER_k(); at the top of every loop body."""

    def __init__(self) -> None:
        self.loop_count = 0
        self.marker_names: List[str] = []  # Track all marker names created

    def _add_marker(self, node: Any) -> None:
        self.loop_count += 1
        marker_name = f"{INVARIANT_MARKER_PREFIX}{self.loop_count}"
        self.marker_names.append(marker_name)  # Track this marker
        marker_stmt = c_ast.FuncCall(name=c_ast.ID(marker_name), args=None)

        if isinstance(node.stmt, c_ast.Compound):
            if node.stmt.block_items is None:
                node.stmt.block_items = [marker_stmt]
            else:
                node.stmt.block_items.insert(0, marker_stmt)
        else:
            node.stmt = c_ast.Compound(block_items=[marker_stmt, node.stmt])

    def visit_While(self, node: c_ast.While) -> None:
        self._add_marker(node)
        self.generic_visit(node)

    def visit_For(self, node: c_ast.For) -> None:
        self._add_marker(node)
        self.generic_visit(node)

    def visit_DoWhile(self, node: c_ast.DoWhile) -> None:
        self._add_marker(node)
        self.generic_visit(node)


class _DeclarationPruner(c_ast.NodeVisitor):
    """Remove unwanted function definitions and extern declarations."""

    def __init__(
        self,
        functions_to_remove: Optional[List[str]] = None,
        remove_externs: bool = True,
    ):
        self.functions_to_remove = set(functions_to_remove or [])
        self.remove_externs = remove_externs

    def visit_FileAST(self, node: c_ast.FileAST) -> None:
        if node.ext:
            node.ext = [ext for ext in node.ext if not self._should_remove(ext)]
        self.generic_visit(node)

    def _should_remove(self, ext: Any) -> bool:
        if self.functions_to_remove:
            if (
                isinstance(ext, c_ast.FuncDef)
                and isinstance(ext.decl, c_ast.Decl)
                and ext.decl.name in self.functions_to_remove
            ):
                return True

        if self.remove_externs:
            if isinstance(ext, c_ast.Decl) and ext.storage and "extern" in ext.storage:
                return True

        return False


class Program:
    """
    AST-based preprocessor + marker-based instrumentation.

    - Keeps the target assertion at its original location using TARGET_ASSERT_MARKER();
    - Places candidate assumes/asserts at loop markers INVARIANT_MARKER_k();
    - Emits llm_code with nondet->rand rewrites.
    """

    def __init__(self):
        self.input_path: Optional[Path] = None
        self.code: Optional[str] = None

        self.pp_code: Optional[str] = None
        self.llm_code: Optional[str] = None
        self.marked_code_from_ast: Optional[str] = None

        self.ast: Optional[c_ast.FileAST] = None
        self.marked_ast: Optional[c_ast.FileAST] = None
        self.llm_ast: Optional[c_ast.FileAST] = None
        # Number of loops (= number of INVARIANT_MARKER_k inserted)
        self.num_loops: int = 0

        # List of all available markers for validation
        self.available_markers: List[str] = []

        # Public API compatibility: one "target" assertion stored here
        self.assertions: List[Predicate] = []
        self.lemmas: List[Predicate] = []  # reserved (not used in this AST pipeline)

        self.parser = c_parser.CParser()
        self.generator = c_generator.CGenerator()

    def from_file_path(self, input_path: Path) -> "Program":
        self.input_path = input_path
        self.code = input_path.read_text().strip()
        return self

    def from_code(self, code: str) -> "Program":
        self.code = code.strip()
        return self

    def preprocess(self) -> str:
        if not self.code:
            raise ValueError("Code is not set")
        pp = _gcc_preprocess(self.code)
        pp = _rewrite_verifier_builtins(pp)
        self.pp_code = pp
        return pp

    def process(
        self, functions_to_remove: Optional[List[str]] = None, print_ast: bool = False
    ) -> Tuple[str, str]:
        """
        Build:
        - self.ast: cleaned AST with the target assert replaced by TARGET_ASSERT_MARKER();
        - self.marked_ast: same, plus INVARIANT_MARKER_k inserted into loop bodies;
        - self.llm_ast: marked_ast but with the target marker materialized back to assert(target).
        Also builds pp_code/marked_code_from_ast/llm_code strings.
        """
        if functions_to_remove is None:
            functions_to_remove = [
                "reach_error",
                "__VERIFIER_assert",
                "assert",
                "assume_abort_if_not",
                "assume",
            ]
        try:
            self.preprocess()
        except Exception as e:
            raise RuntimeError(f"Preprocessing failed: {e}") from e

        assert self.pp_code is not None

        # Parse and prune declarations
        # print("PP CODE:", self.pp_code)
        self.ast = self.parser.parse(self.pp_code)
        _DeclarationPruner(
            functions_to_remove=functions_to_remove, remove_externs=True
        ).visit(self.ast)

        # Replace the (single) target assertion with a marker at its original location
        self.assertions = []
        self._replace_first_assert_with_target_marker(self.ast)
        if not self.assertions:
            raise ValueError("No target assert(...) found in program")

        # Loop markers for placement (verification-time and LLM-time)
        self.marked_ast = copy.deepcopy(self.ast)
        inserter = _LoopMarkerInserter()
        inserter.visit(self.marked_ast)
        self.num_loops = inserter.loop_count

        # Store all available markers (loop markers + target marker)
        self.available_markers = inserter.marker_names.copy()
        if self.assertions:  # Only add if target assert was found
            self.available_markers.append(TARGET_ASSERT_MARKER)

        # LLM-facing: show the target property as assert(...) (not as a marker call)
        self.llm_ast = copy.deepcopy(self.marked_ast)
        self._replace_marker_with_call(
            self.llm_ast,
            marker_name=TARGET_ASSERT_MARKER,
            call_name="assert",
            predicate_content=self.assertions[0].content,
        )

        # Emit code strings (no PATCH here; PATCH is added by get_program_with_assertion when for_llm=False)
        self.pp_code = self.generator.visit(self.ast).strip()
        self.marked_code_from_ast = self.generator.visit(self.marked_ast).strip()
        self.llm_code = self._replace_nondet_with_rand(
            self.generator.visit(self.llm_ast).strip()
        )

        if print_ast:
            print("AST:", self.ast)
            print("MARKED AST:", self.marked_ast)
            print("LLM AST:", self.llm_ast)

        return self.pp_code, self.llm_code

    def get_target_assert(self) -> Predicate:
        return self.assertions[0]

    def get_available_markers(self) -> List[str]:
        """Return list of all available markers for validation.

        Returns:
            List of marker names including:
            - INVARIANT_MARKER_1, INVARIANT_MARKER_2, ... (loop markers)
            - TARGET_ASSERT_MARKER (target assertion marker)
        """
        return self.available_markers.copy()

    def _extract_first_arg_as_text(self, call: c_ast.FuncCall) -> str:
        try:
            if call.args is None:
                return ""
            if not isinstance(call.args, c_ast.ExprList) or not call.args.exprs:
                return ""
            return self.generator.visit(call.args.exprs[0]).strip()
        except Exception:
            return ""

    def _replace_first_assert_with_target_marker(self, ast: c_ast.FileAST) -> None:
        program_self = self

        class Markerizer(c_ast.NodeVisitor):
            def __init__(self) -> None:
                super().__init__()
                self.done = False

            def _maybe_replace_stmt(self, stmt: Any) -> Any:
                if self.done:
                    return stmt
                if _is_call_named(stmt, "assert"):
                    program_self.assertions.append(
                        Predicate(
                            content=program_self._extract_first_arg_as_text(stmt),
                            marker_name=TARGET_ASSERT_MARKER,
                        )
                    )
                    self.done = True
                    return c_ast.FuncCall(
                        name=c_ast.ID(TARGET_ASSERT_MARKER), args=None
                    )
                return stmt

            def visit_Compound(self, node: c_ast.Compound) -> None:
                if node.block_items:
                    new_items = []
                    for it in node.block_items:
                        if isinstance(it, c_ast.Label):
                            # Preserve the label, but replace its inner statement if needed
                            it.stmt = self._maybe_replace_stmt(it.stmt)
                            new_items.append(it)
                        else:
                            new_items.append(self._maybe_replace_stmt(it))
                    node.block_items = new_items
                self.generic_visit(node)

        Markerizer().visit(ast)

    def _parse_expr(self, expr_src: str) -> c_ast.Node:
        """Parse a predicate expression into a pycparser expression node (best-effort)."""
        try:
            wrapped = f"void _temp() {{ int _dummy = {expr_src}; }}"
            temp_ast = self.parser.parse(wrapped)
            for ext in temp_ast.ext:
                if isinstance(ext, c_ast.FuncDef):
                    for stmt in ext.body.block_items or []:
                        if isinstance(stmt, c_ast.Decl) and stmt.init is not None:
                            return stmt.init
        except Exception:
            pass
        try:
            wrapped = f"void _temp() {{ if ({expr_src}) {{ }} }}"
            temp_ast = self.parser.parse(wrapped)
            for ext in temp_ast.ext:
                if isinstance(ext, c_ast.FuncDef):
                    for stmt in ext.body.block_items or []:
                        if isinstance(stmt, c_ast.If):
                            return stmt.cond
        except Exception:
            pass
        return c_ast.ID(expr_src)

    def _replace_marker_with_call(
        self,
        ast: c_ast.FileAST,
        marker_name: str,
        call_name: str,
        predicate_content: str,
    ) -> None:
        call_stmt = c_ast.FuncCall(
            name=c_ast.ID(call_name),
            args=c_ast.ExprList(exprs=[self._parse_expr(predicate_content)]),
        )

        class Replacer(c_ast.NodeVisitor):
            def visit_Compound(self, node: c_ast.Compound) -> None:
                if node.block_items:
                    new_items = []
                    for item in node.block_items:
                        if (
                            isinstance(item, c_ast.FuncCall)
                            and isinstance(item.name, c_ast.ID)
                            and item.name.name == marker_name
                        ):
                            new_items.append(call_stmt)
                        elif isinstance(item, c_ast.Label):
                            if (
                                isinstance(item.stmt, c_ast.FuncCall)
                                and isinstance(item.stmt.name, c_ast.ID)
                                and item.stmt.name.name == marker_name
                            ):
                                item.stmt = call_stmt
                            new_items.append(item)
                        else:
                            new_items.append(item)
                    node.block_items = new_items
                self.generic_visit(node)

        Replacer().visit(ast)

    def _replace_loop_markers(
        self, ast: c_ast.FileAST, replacements: Dict[str, Dict[str, str]]
    ) -> None:
        program_self = self

        class LoopMarkerReplacer(c_ast.NodeVisitor):
            def _make_replacement(self, marker_name: str) -> c_ast.FuncCall:
                spec = replacements[marker_name]
                return c_ast.FuncCall(
                    name=c_ast.ID(spec["kind"]),
                    args=c_ast.ExprList(exprs=[program_self._parse_expr(spec["expr"])]),
                )

            def visit_Compound(self, node: c_ast.Compound) -> None:
                if node.block_items:
                    new_items = []
                    for item in node.block_items:
                        if (
                            isinstance(item, c_ast.FuncCall)
                            and isinstance(item.name, c_ast.ID)
                            and item.name.name in replacements
                        ):
                            new_items.append(self._make_replacement(item.name.name))
                        elif isinstance(item, c_ast.Label):
                            # Preserve labels, but replace marker inside if needed
                            if (
                                isinstance(item.stmt, c_ast.FuncCall)
                                and isinstance(item.stmt.name, c_ast.ID)
                                and item.stmt.name.name in replacements
                            ):
                                item.stmt = self._make_replacement(item.stmt.name.name)
                            new_items.append(item)
                        else:
                            new_items.append(item)
                    node.block_items = new_items
                self.generic_visit(node)

        LoopMarkerReplacer().visit(ast)

    def _remove_unresolved_markers(self, ast: c_ast.FileAST) -> None:
        class Cleaner(c_ast.NodeVisitor):
            def _is_marker_stmt(self, stmt: Any) -> bool:
                return (
                    isinstance(stmt, c_ast.FuncCall)
                    and isinstance(stmt.name, c_ast.ID)
                    and (
                        stmt.name.name.startswith(INVARIANT_MARKER_PREFIX)
                        or stmt.name.name == TARGET_ASSERT_MARKER
                    )
                )

            def visit_Compound(self, node: c_ast.Compound) -> None:
                if node.block_items:
                    new_items = []
                    for item in node.block_items:
                        if self._is_marker_stmt(item):
                            new_items.append(c_ast.EmptyStatement())
                        elif isinstance(item, c_ast.Label) and self._is_marker_stmt(
                            item.stmt
                        ):
                            item.stmt = c_ast.EmptyStatement()
                            new_items.append(item)
                        else:
                            new_items.append(item)
                    node.block_items = new_items
                self.generic_visit(node)

        Cleaner().visit(ast)

    def _replace_nondet_with_rand(self, code: str) -> str:
        out = code
        tokens = set(re.findall(r"\b__VERIFIER_nondet_\w+\b", code))
        for token in tokens:
            out = out.replace(
                token + "()",
                nondet_type(token.split("__VERIFIER_nondet_")[1]) + " rand()",
            )
        return out

    def get_program_with_assertion(
        self,
        predicate: Optional[Predicate],
        assumptions: List[Predicate],
        for_llm: bool,
        preserve_predicate_format: bool = True,
    ) -> str:
        """
        Marker-based instrumentation.

        - assumptions: each must have marker_name=INVARIANT_MARKER_k
        - predicate:
            - if marker_name==TARGET_ASSERT_MARKER: assert(target) at original target location
            - else marker_name==INVARIANT_MARKER_k: assert(candidate) at that loop marker
        - preserve_predicate_format: if True, use string replacement to preserve the original
            predicate format (avoids pycparser adding extra parentheses)
        """
        if self.marked_ast is None or self.llm_ast is None:
            raise ValueError("Call process() first")

        # Use placeholder for string-based replacement to preserve predicate format
        PLACEHOLDER = "__PREDICATE_PLACEHOLDER__"
        original_predicates: Dict[str, str] = {}  # marker -> original content

        working_ast = copy.deepcopy(self.llm_ast if for_llm else self.marked_ast)

        repls: Dict[str, Dict[str, str]] = {}
        for a in assumptions:
            if not a.marker_name or not a.marker_name.startswith(
                INVARIANT_MARKER_PREFIX
            ):
                raise ValueError("assumptions must have marker_name=INVARIANT_MARKER_k")
            if preserve_predicate_format:
                placeholder_id = f"{PLACEHOLDER}_{a.marker_name}"
                original_predicates[placeholder_id] = a.content
                repls[a.marker_name] = {"kind": "assume", "expr": placeholder_id}
            else:
                repls[a.marker_name] = {"kind": "assume", "expr": a.content}

        if predicate is not None:
            if predicate.marker_name == TARGET_ASSERT_MARKER:
                if preserve_predicate_format:
                    placeholder_id = f"{PLACEHOLDER}_{TARGET_ASSERT_MARKER}"
                    original_predicates[placeholder_id] = predicate.content
                    self._replace_marker_with_call(
                        working_ast,
                        marker_name=TARGET_ASSERT_MARKER,
                        call_name="assert",
                        predicate_content=placeholder_id,
                    )
                else:
                    self._replace_marker_with_call(
                        working_ast,
                        marker_name=TARGET_ASSERT_MARKER,
                        call_name="assert",
                        predicate_content=predicate.content,
                    )
            elif predicate.marker_name and predicate.marker_name.startswith(
                INVARIANT_MARKER_PREFIX
            ):
                if preserve_predicate_format:
                    placeholder_id = f"{PLACEHOLDER}_{predicate.marker_name}"
                    original_predicates[placeholder_id] = predicate.content
                    repls[predicate.marker_name] = {
                        "kind": "assert",
                        "expr": placeholder_id,
                    }
                else:
                    repls[predicate.marker_name] = {
                        "kind": "assert",
                        "expr": predicate.content,
                    }
            else:
                raise ValueError(
                    "predicate must have marker_name=TARGET_ASSERT_MARKER or INVARIANT_MARKER_k"
                )

        if repls:
            self._replace_loop_markers(working_ast, repls)

        if not for_llm:
            self._remove_unresolved_markers(working_ast)

        program = self.generator.visit(working_ast).strip()

        # Replace placeholders with original predicate content (preserves formatting)
        if preserve_predicate_format:
            for placeholder_id, original_content in original_predicates.items():
                program = program.replace(placeholder_id, original_content)

        if for_llm:
            program = self._replace_nondet_with_rand(program)
        else:
            program = PATCH + program
        return program


def nondet_type(type_str: str) -> str:
    if type_str == "uchar":
        return "(unsigned char)"
    if type_str == "char":
        return "(signed char)"
    if type_str == "uint":
        return "(unsigned int)"
    return f"({type_str})"


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.preprocess.program <file.c>")
        raise SystemExit(2)

    path = sys.argv[1]
    p = Program().from_file_path(Path(path))
    pp, llm = p.process(print_ast=True)
    print("----- PP (no patch) -----")
    print(pp)
    # print("----- LLM (nondet->rand) -----")
    # print(llm)

    # predicate = Predicate(content="x > 0", marker_name="INVARIANT_MARKER_1")
    # assumptions = []
    # program = p.get_program_with_assertion(predicate, assumptions, for_llm=False)
    # print("----- Program -----")
    # print(program)

    # predicate = p.get_target_assert()
    # print("----- Target Assert -----")
    # print(predicate)
    # assumptions = [
    #     # Predicate(content="y > 0", marker_name="INVARIANT_MARKER_2"),
    #     # Predicate(content="z > 0", marker_name="INVARIANT_MARKER_3"),
    # ]
    # program = p.get_program_with_assertion(predicate, assumptions, for_llm=False)
    # print("----- Program -----")
    # print(program)
    # print("----- Available Markers -----")
    print(p.get_available_markers())
