from dataclasses import dataclass
from typing import Optional, Dict, List, Any
import sys
import re
import subprocess
import tempfile
import os
import copy
from pathlib import Path
from pycparser import c_parser, c_generator, c_ast
from src.preprocess.predicate import Predicate  # type: ignore

PATCH = (
    "void assert(int cond) { if (!(cond)) { ERROR : { reach_error(); abort(); } } }\n"
    "void assume(int cond) { if (!cond) { abort(); } }\n"
)


class LoopLabeler(c_ast.NodeVisitor):
    def __init__(self):
        self.loop_count = 0

    def _add_marker(self, node):
        """Helper method to add an invariant marker to a loop body."""
        self.loop_count += 1
        # Create a marker: we use a FunctionCall as a placeholder for the assertion
        # because it's easier for the AST generator to render than a raw comment string
        marker_name = f"INVARIANT_MARKER_{self.loop_count}"
        marker = c_ast.FuncCall(name=c_ast.ID(marker_name), args=None)

        if isinstance(node.stmt, c_ast.Compound):
            # Insert at the very top of the { } block
            node.stmt.block_items.insert(0, marker)
        else:
            # Convert single line loop to compound block
            node.stmt = c_ast.Compound(block_items=[marker, node.stmt])

    def visit_While(self, node):
        self._add_marker(node)
        self.generic_visit(node)

    def visit_For(self, node):
        self._add_marker(node)
        self.generic_visit(node)

    def visit_DoWhile(self, node):
        self._add_marker(node)
        self.generic_visit(node)


class DeclarationRemover(c_ast.NodeVisitor):
    """Visitor to remove functions and extern declarations from the AST."""

    def __init__(self, functions_to_remove=None, remove_externs=True):
        """
        Args:
            functions_to_remove: Set or list of function names to remove. If None, no functions are removed.
            remove_externs: If True, remove all extern declarations.
        """
        self.functions_to_remove = (
            set(functions_to_remove) if functions_to_remove else set()
        )
        self.remove_externs = remove_externs

    def visit_FileAST(self, node):
        """Visit the root node and filter out unwanted declarations."""
        if node.ext:
            node.ext = [ext for ext in node.ext if not self._should_remove(ext)]
        self.generic_visit(node)

    def _should_remove(self, ext):
        """Check if a declaration should be removed."""
        # Remove functions by name
        if self.functions_to_remove:
            if (
                isinstance(ext, c_ast.FuncDef)
                and isinstance(ext.decl, c_ast.Decl)
                and ext.decl.name in self.functions_to_remove
            ):
                return True

        # Remove extern declarations
        if self.remove_externs:
            if isinstance(ext, c_ast.Decl) and ext.storage and "extern" in ext.storage:
                return True

        return False


class MarkerReplacer(c_ast.NodeVisitor):
    """Visitor to replace INVARIANT_MARKER_K function calls with assume/assert statements."""

    def __init__(self, predicates: Dict[str, str], parser: c_parser.CParser):
        """
        Args:
            predicates: Dictionary mapping marker names to predicate content.
                       e.g., {"INVARIANT_MARKER_1": "x > 0", "INVARIANT_MARKER_2": "y < 10"}
            parser: C parser instance for parsing predicate expressions.
        """
        self.predicates = predicates
        self.parser = parser

    def _parse_predicate_expression(self, predicate_content: str) -> c_ast.Node:
        """Parse a predicate string into an AST expression node."""
        try:
            # Wrap in a simple expression context to parse
            wrapped = f"void _temp() {{ int _dummy = {predicate_content}; }}"
            temp_ast = self.parser.parse(wrapped)
            # Extract the expression from the assignment
            for ext in temp_ast.ext:
                if isinstance(ext, c_ast.FuncDef):
                    for stmt in ext.body.block_items:
                        if isinstance(stmt, c_ast.Decl) and stmt.init:
                            return stmt.init
        except Exception:
            pass
        # Fallback: try to parse as a condition
        try:
            wrapped = f"void _temp() {{ if ({predicate_content}) {{ }} }}"
            temp_ast = self.parser.parse(wrapped)
            for ext in temp_ast.ext:
                if isinstance(ext, c_ast.FuncDef):
                    for stmt in ext.body.block_items:
                        if isinstance(stmt, c_ast.If):
                            return stmt.cond
        except Exception:
            pass
        # Final fallback: return as ID
        return c_ast.ID(predicate_content)

    def visit_Compound(self, node):
        """Visit compound statements to find and replace markers in block_items."""
        if node.block_items:
            new_block_items = []
            for item in node.block_items:
                if isinstance(item, c_ast.FuncCall) and isinstance(item.name, c_ast.ID):
                    marker_name = item.name.name
                    if (
                        marker_name.startswith("INVARIANT_MARKER_")
                        and marker_name in self.predicates
                    ):
                        # Replace marker with assume statement
                        predicate_content = self.predicates[marker_name]
                        predicate_expr = self._parse_predicate_expression(
                            predicate_content
                        )
                        assume_call = c_ast.FuncCall(
                            name=c_ast.ID("assume"),
                            args=c_ast.ExprList(exprs=[predicate_expr]),
                        )
                        # FuncCall itself is a valid statement in C
                        new_block_items.append(assume_call)
                    else:
                        new_block_items.append(item)
                else:
                    new_block_items.append(item)
            node.block_items = new_block_items
        self.generic_visit(node)


class Program:
    def __init__(self):
        self.input_path: Optional[Path] = None
        self.code: Optional[str] = None
        self.pp_code: Optional[str] = None
        self.llm_code: Optional[str] = None
        self.marked_code_from_ast: Optional[str] = None
        self.ast: Optional[c_ast.FileAST] = None
        self.llm_ast: Optional[c_ast.FileAST] = None
        self.marked_ast: Optional[c_ast.FileAST] = None
        # Compatibility with src.utils.program.Program API
        self.assertions: List[Predicate] = []
        self.lemmas: List[Predicate] = []
        # Marker name used to represent the (single) target assertion location in the AST
        self.target_assert_marker: str = "TARGET_ASSERT_MARKER"
        self.parser: Optional[c_parser.CParser] = c_parser.CParser()
        self.generator: Optional[c_generator.CGenerator] = c_generator.CGenerator()

    def from_file_path(self, input_path: Path) -> "Program":
        self.input_path = input_path
        self.code = input_path.read_text().strip()
        return self

    def from_code(self, code: str) -> "Program":
        self.code = code.strip()
        return self

    def preprocess(self) -> str:
        """Preprocess C file using GCC to handle directives, comments, and includes."""
        # Remove includes before GCC preprocessing to avoid macro expansion
        self.pp_code = re.sub(
            r'^\s*#include\s+[<"].*?[>"]\s*$', "", self.code, flags=re.MULTILINE
        )
        fd, path = tempfile.mkstemp(suffix=".c", prefix="gcc_preprocess_")
        os.close(fd)
        tmp_file = Path(path)
        try:
            # Write cleaned code to temp location
            tmp_file.write_text(self.pp_code)
            command = f"gcc -E -P -nostdinc {tmp_file}"
            p = subprocess.run(command.split(), capture_output=True)
            if p.returncode != 0:
                stderr = p.stderr.decode("utf-8", errors="replace")
                raise RuntimeError(f"GCC preprocessing failed: {stderr}")
            output = p.stdout.decode("utf-8", errors="replace")
            self.pp_code = output
        finally:
            if tmp_file.exists():
                tmp_file.unlink()

        # Remove __attribute__ declarations (GCC extension, keep regex removal before parsing)
        self.pp_code = re.sub(r"__attribute__\s*\(\([^)]*\)\)", "", self.pp_code)
        self.pp_code = re.sub(r"printf\s*\([^)]*\)\s*;", "", self.pp_code)

        # Remove __extension__ prefixes (GCC extension, keep regex removal before parsing)
        self.pp_code = re.sub(r"__extension__\s*", "", self.pp_code)
        self.pp_code = self.pp_code.replace("__VERIFIER_assert", "assert")
        self.pp_code = self.pp_code.replace("assume_abort_if_not", "assume")
        self.pp_code = re.sub(r"\n\s*\n\s*\n+", "\n\n", self.pp_code).strip()

    @staticmethod
    def _is_call_named(node: Any, name: str) -> bool:
        return (
            isinstance(node, c_ast.FuncCall)
            and isinstance(node.name, c_ast.ID)
            and node.name.name == name
        )

    def _extract_first_arg_as_text(self, call: c_ast.FuncCall) -> str:
        """
        Best-effort extraction of the call's first argument as C text.
        Falls back to "" if the structure is unexpected.
        """
        try:
            if call.args is None:
                return ""
            if not isinstance(call.args, c_ast.ExprList) or not call.args.exprs:
                return ""
            return self.generator.visit(call.args.exprs[0]).strip()
        except Exception:
            return ""

    def _replace_first_assert_with_target_marker(self, ast: c_ast.FileAST) -> None:
        """
        Find the first statement-level assert(...) call in the program and replace it
        with a marker statement TARGET_ASSERT_MARKER(); while recording its predicate.

        This lets us keep the *original location* of the target assertion in the AST,
        and later choose whether to materialize it (usefulness) or hide/replace it (correctness).
        """
        program_self = self

        class Markerizer(c_ast.NodeVisitor):
            def __init__(self):
                super().__init__()
                self.done = False

            def _maybe_replace_stmt(self, stmt: Any) -> Any:
                if self.done:
                    return stmt
                if Program._is_call_named(stmt, "assert"):
                    program_self.assertions.append(
                        Predicate(
                            marker_name=program_self.target_assert_marker,
                            content=program_self._extract_first_arg_as_text(stmt),
                        )
                    )
                    self.done = True
                    return c_ast.FuncCall(
                        name=c_ast.ID(program_self.target_assert_marker), args=None
                    )
                return stmt

            def visit_Compound(self, node: c_ast.Compound):
                if node.block_items:
                    new_items = []
                    for item in node.block_items:
                        if isinstance(item, c_ast.Label):
                            item.stmt = self._maybe_replace_stmt(item.stmt)
                            new_items.append(item)
                        else:
                            new_items.append(self._maybe_replace_stmt(item))
                    node.block_items = new_items
                self.generic_visit(node)

        Markerizer().visit(ast)

    def _remove_unresolved_markers(self, ast: c_ast.FileAST) -> None:
        """Remove any remaining marker statements to keep generated C valid."""

        class Cleaner(c_ast.NodeVisitor):
            def _is_marker_stmt(self, stmt: Any) -> bool:
                return (
                    isinstance(stmt, c_ast.FuncCall)
                    and isinstance(stmt.name, c_ast.ID)
                    and (
                        stmt.name.name.startswith("INVARIANT_MARKER_")
                        or stmt.name.name == self_marker
                    )
                )

            def visit_Compound(self, node: c_ast.Compound):
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

        self_marker = self.target_assert_marker
        Cleaner().visit(ast)

    def replace_nondet_with_rand_for_llm(self, llm_code_from_ast: str):
        """Replace __VERIFIER_nondet_*() calls with rand() in LLM code."""
        self.llm_code = llm_code_from_ast
        pattern = r"\b__VERIFIER_nondet_\w+\b"
        tokens = set(re.findall(pattern, llm_code_from_ast))
        for token in tokens:
            pattern = token + "()"
            replacement = (
                self.nondet_type(token.split("__VERIFIER_nondet_")[1]) + " rand()"
            )
            self.llm_code = self.llm_code.replace(pattern, replacement)

    def process(self, functions_to_remove=None, print_ast=False) -> str:
        """
        Complete processing pipeline: preprocess, parse, remove functions, add loop markers to LLM AST only.

        Args:
            functions_to_remove: List of function names to remove from AST.
                                Defaults to ["reach_error", "__VERIFIER_assert"]
            print_ast: If True, print the AST after parsing

        Returns:
            Generated C code with loop markers added
        """
        if not self.code:
            raise ValueError("Code is not set")

        if functions_to_remove is None:
            functions_to_remove = [
                "reach_error",
                "__VERIFIER_assert",
                "assert",
                "assume_abort_if_not",
                "assume",
            ]

        try:
            # Preprocess the file using GCC
            self.preprocess()

            # Debug: check if code is empty or malformed
            if not self.pp_code or not self.pp_code.strip():
                raise ValueError("Preprocessed code is empty")

            # 1. Parse the preprocessed code
            self.ast = self.parser.parse(self.pp_code)
            if print_ast:
                print("PP AST:", self.ast)

            # 2. Remove unwanted functions and extern declarations from AST
            declaration_remover = DeclarationRemover(
                functions_to_remove=functions_to_remove, remove_externs=True
            )
            declaration_remover.visit(self.ast)
            # 8. Add PATCH to both codes
            self.pp_code = PATCH + self.pp_code
            self.ast = self.parser.parse(self.pp_code)
            if print_ast:
                print("PP AST:", self.ast)

            # 2.5 Replace the (single) target assertion with a marker at the original location
            self.assertions = []
            self._replace_first_assert_with_target_marker(self.ast)

            # 3. Create a loop-marked AST for verification-time insertion (assert/assume by marker)
            self.marked_ast = copy.deepcopy(self.ast)
            # 4. Add loop markers to the marked AST
            labeler = LoopLabeler()
            labeler.visit(self.marked_ast)
            if print_ast:
                print("MARKED AST:", self.marked_ast)

            # 5. Create LLM-facing AST from the marked AST (includes markers)
            self.llm_ast = copy.deepcopy(self.marked_ast)

            # 5.5 In the LLM-facing AST, show the target assert (not the marker) at its original location
            if self.assertions:
                target = self.assertions[0]
                self._replace_marker_with_call(
                    self.llm_ast,
                    marker_name=self.target_assert_marker,
                    call_name="assert",
                    predicate_content=target.content,
                )

            # 6. Generate Code back from ASTs
            self.pp_code = self.generator.visit(self.ast).strip()
            self.llm_code = self.generator.visit(self.llm_ast).strip()
            # self.replace_nondet_with_rand_for_llm(
            #     self.generator.visit(self.llm_ast).strip()
            # )
            # self.llm_ast = self.parser.parse(self.llm_code)
            # self.llm_code = self.generator.visit(self.llm_ast).strip()
            self.marked_code_from_ast = self.generator.visit(self.marked_ast).strip()

        except Exception as e:
            import traceback

            raise RuntimeError(f"Error: {e}\n{traceback.format_exc()}") from e

    def nondet_type(self, type_str: str):
        """Convert nondet type to C cast expression."""
        if type_str == "uchar":
            return "(unsigned char)"
        elif type_str == "char":
            return "(signed char)"
        elif type_str == "uint":
            return "(unsigned int)"
        else:
            return f"({type_str})"

    def _parse_predicate_expression(self, predicate_content: str) -> c_ast.Node:
        """Parse a predicate string into an AST expression node."""
        # Wrap in a simple expression context to parse
        wrapped = f"void _temp() {{ int _dummy = {predicate_content}; }}"
        temp_ast = self.parser.parse(wrapped)
        # Extract the expression from the assignment
        for ext in temp_ast.ext:
            if isinstance(ext, c_ast.FuncDef):
                for stmt in ext.body.block_items:
                    if isinstance(stmt, c_ast.Decl) and stmt.init:
                        return stmt.init
        # Fallback: return as ID if parsing fails
        return c_ast.ID(predicate_content)

    def _add_assume_to_ast(self, ast: c_ast.FileAST, predicate_content: str):
        """(Legacy helper) Add an assume statement at the beginning of main function."""
        for ext in ast.ext:
            if isinstance(ext, c_ast.FuncDef) and isinstance(ext.decl, c_ast.Decl):
                if ext.decl.name == "main":
                    assume_expr = self._parse_predicate_expression(predicate_content)
                    assume_call = c_ast.FuncCall(
                        name=c_ast.ID("assume"),
                        args=c_ast.ExprList(exprs=[assume_expr]),
                    )
                    if isinstance(ext.body, c_ast.Compound):
                        if ext.body.block_items is None:
                            ext.body.block_items = [assume_call]
                        else:
                            ext.body.block_items.insert(0, assume_call)
                    else:
                        ext.body = c_ast.Compound(block_items=[assume_call, ext.body])
                    break

    def _add_assert_to_ast(self, ast: c_ast.FileAST, predicate_content: str):
        """(Legacy helper) Add an assert statement at the end of main function."""
        for ext in ast.ext:
            if isinstance(ext, c_ast.FuncDef) and isinstance(ext.decl, c_ast.Decl):
                if ext.decl.name == "main":
                    assert_expr = self._parse_predicate_expression(predicate_content)
                    assert_call = c_ast.FuncCall(
                        name=c_ast.ID("assert"),
                        args=c_ast.ExprList(exprs=[assert_expr]),
                    )
                    if isinstance(ext.body, c_ast.Compound):
                        if ext.body.block_items is None:
                            ext.body.block_items = [assert_call]
                        else:
                            ext.body.block_items.append(assert_call)
                    else:
                        ext.body = c_ast.Compound(block_items=[ext.body, assert_call])
                    break

    def _replace_marker_with_call(
        self,
        ast: c_ast.FileAST,
        marker_name: str,
        call_name: str,
        predicate_content: str,
    ) -> None:
        """Replace a specific marker statement with call_name(predicate_content)."""

        pred_expr = self._parse_predicate_expression(predicate_content)
        call_stmt = c_ast.FuncCall(
            name=c_ast.ID(call_name), args=c_ast.ExprList(exprs=[pred_expr])
        )

        class Replacer(c_ast.NodeVisitor):
            def visit_Compound(self, node: c_ast.Compound):
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
        """
        Replace INVARIANT_MARKER_k with assume/assert calls.

        replacements: { marker_name: { "kind": "assume"|"assert", "expr": "<c expr>" } }
        """

        class LoopMarkerReplacer(c_ast.NodeVisitor):
            def visit_Compound(self, node: c_ast.Compound):
                if node.block_items:
                    new_items = []
                    for item in node.block_items:
                        if (
                            isinstance(item, c_ast.FuncCall)
                            and isinstance(item.name, c_ast.ID)
                            and item.name.name in replacements
                        ):
                            spec = replacements[item.name.name]
                            expr = program_self._parse_predicate_expression(
                                spec["expr"]
                            )
                            new_items.append(
                                c_ast.FuncCall(
                                    name=c_ast.ID(spec["kind"]),
                                    args=c_ast.ExprList(exprs=[expr]),
                                )
                            )
                        else:
                            new_items.append(item)
                    node.block_items = new_items
                self.generic_visit(node)

        program_self = self
        LoopMarkerReplacer().visit(ast)

    def get_program_with_assertion(
        self,
        predicate: Optional[Predicate],
        assumptions: List[Predicate],
        for_llm: bool,
    ) -> str:
        """
        Generate program with caller-selected assertions/assumptions.

        Args:
            predicate: Optional predicate to assert (typically: candidate for correctness, or target for usefulness).
            assumptions: List of predicates to assume at the start of main (typically: candidate(s) for usefulness).
            for_llm: If True, use the loop-marked llm_ast; otherwise use ast.

        Returns:
            Generated C code with inserted assumes/assert (and optional marker replacement).
        """
        if not self.ast:
            raise ValueError("AST not initialized. Call process() first.")
        if not self.marked_ast:
            raise ValueError("Marked AST not initialized. Call process() first.")

        # Select which AST to use
        working_ast = copy.deepcopy(self.llm_ast if for_llm else self.marked_ast)

        # 1) Loop-marker driven assumes
        loop_repls: Dict[str, Dict[str, str]] = {}
        for a in assumptions:
            marker = getattr(a, "marker_name", None)
            if not marker or not marker.startswith("INVARIANT_MARKER_"):
                raise ValueError(
                    "AST pipeline expects assumptions to carry marker_name like INVARIANT_MARKER_k"
                )
            loop_repls[marker] = {"kind": "assume", "expr": a.content}

        # 2) Assertion placement:
        # - If predicate matches the target, materialize it at its original marker location.
        # - Else, if it targets a loop marker, assert at that loop marker.
        if predicate is not None:
            pred_marker = getattr(predicate, "marker_name", None)
            is_target_by_marker = pred_marker == self.target_assert_marker
            # is_target_by_content = self.assertions and predicate.content == self.assertions[0].content and pred_marker is None

            if is_target_by_marker:
                self._replace_marker_with_call(
                    working_ast,
                    marker_name=self.target_assert_marker,
                    call_name="assert",
                    predicate_content=predicate.content,
                )
            elif pred_marker and pred_marker.startswith("INVARIANT_MARKER_"):
                loop_repls[pred_marker] = {"kind": "assert", "expr": predicate.content}
            else:
                raise ValueError(
                    "AST pipeline expects predicate to be either the target or have marker_name=INVARIANT_MARKER_k"
                )

        # 3) Apply loop marker replacements (assume/assert)
        if loop_repls:
            self._replace_loop_markers(working_ast, loop_repls)

        # 5) Remove any unresolved markers so verifier code stays valid C
        if not for_llm:
            self._remove_unresolved_markers(working_ast)

        # Generate code
        program = self.generator.visit(working_ast).strip()
        return program

    def get_target_assert(self) -> Predicate:
        return self.assertions[0]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python labeler.py <preprocessed_file.c>")
    else:
        program = Program().from_file_path(Path(sys.argv[1]))
        program.process(print_ast=True)
        print("--------------------------------")
        print("PP Code:")
        print(program.pp_code)
        print("--------------------------------")
        print("LLM Code:")
        print(program.llm_code)
        print("--------------------------------")
        print("Marked Code from AST:")
        print(program.marked_code_from_ast)
        print("--------------------------------")
        for_llm = False
        predicate = Predicate(content="x > 0", marker_name="INVARIANT_MARKER_1")
        assumptions = []
        program_with_assertion = program.get_program_with_assertion(
            predicate, assumptions, for_llm
        )
        print("--------------------------------")
        print("Program with Assertion:")
        print(program_with_assertion)

        print("--------------------------------")
        predicate = program.get_target_assert()
        assumptions = [
            Predicate(content="x > 0", marker_name="INVARIANT_MARKER_1"),
            Predicate(content="y > 0", marker_name="INVARIANT_MARKER_2"),
        ]
        program_with_assertion = program.get_program_with_assertion(
            predicate, assumptions, for_llm
        )
        print("--------------------------------")
        print("Program with Assertion:")
        print(program_with_assertion)
