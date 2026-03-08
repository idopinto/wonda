#!/usr/bin/env python3
"""Self-contained demo of the AstProgram and Property pipeline.

Run with:
    uv run python -m wonda.demos.ast_program_demo

No external C files or tools (UAutomizer, runlim) are needed — this script
embeds a small SV-COMP-style C program and walks through every stage of the
AstProgram transformation, printing the results at each step.
"""
from __future__ import annotations

from wonda.core.ast_program import AstProgram
from wonda.core.property import Property

# ---------------------------------------------------------------------------
# 1. Sample SV-COMP C program (embedded as a string)
# ---------------------------------------------------------------------------
SAMPLE_PROGRAM = r"""
extern void abort(void);
extern void __assert_fail(const char *, const char *, unsigned int, const char *);
void reach_error() { __assert_fail("0", "example.c", 3, "reach_error"); }

extern int __VERIFIER_nondet_int(void);
void __VERIFIER_assert(int cond) { if (!(cond)) { ERROR: { reach_error(); abort(); } } }

int main() {
    int n = __VERIFIER_nondet_int();
    if (n <= 0) return 0;

    int sum = 0;
    int i = 0;
    while (i < n) {
        sum = sum + i;
        i = i + 1;
    }
    __VERIFIER_assert(sum >= 0);
    return 0;
}
"""

SEPARATOR = "=" * 70


def section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def main() -> None:
    # ------------------------------------------------------------------
    # 2. Create and process the AstProgram
    # ------------------------------------------------------------------
    section("1. RAW INPUT PROGRAM")
    print(SAMPLE_PROGRAM.strip())

    program = AstProgram().from_code(SAMPLE_PROGRAM)
    pp_code, llm_code = program.process(print_ast=False)

    # ------------------------------------------------------------------
    # 3. Show preprocessed code (verifier-facing, no patch)
    # ------------------------------------------------------------------
    section("2. PREPROCESSED CODE (verifier-facing, no PATCH header)")
    print(pp_code)

    # ------------------------------------------------------------------
    # 4. Show LLM-facing code (nondet -> rand, target assert visible)
    # ------------------------------------------------------------------
    section("3. LLM-FACING CODE (nondet -> rand(), target assert visible)")
    print(llm_code)

    # ------------------------------------------------------------------
    # 5. Show marked code (all markers visible)
    # ------------------------------------------------------------------
    section("4. MARKED CODE (all markers visible)")
    print(program.marked_code_from_ast)

    # ------------------------------------------------------------------
    # 6. Inspect extracted properties and markers
    # ------------------------------------------------------------------
    section("5. EXTRACTED TARGET ASSERTION")
    target = program.get_target_assert()
    print(f"  {target}")

    section("6. AVAILABLE MARKERS")
    for marker in program.get_available_markers():
        print(f"  - {marker}")
    print(f"\n  Number of loops found: {program.num_loops}")

    # ------------------------------------------------------------------
    # 7. Generate a CORRECTNESS verification program
    #    "Is  i >= 0 && sum >= 0  a valid loop invariant?"
    # ------------------------------------------------------------------
    candidate = Property(content="i >= 0 && sum >= 0", marker_name="INVARIANT_MARKER_1")

    section("7. CORRECTNESS QUERY — assert(candidate) at loop marker")
    print(f"  Candidate invariant: {candidate}\n")
    correctness_prog = program.get_program_with_assertion(
        property_to_assert=candidate,
        assumptions=[],
        for_llm=False,
    )
    print(correctness_prog)

    # ------------------------------------------------------------------
    # 8. Generate a USEFULNESS verification program
    #    "Assuming the candidate, does the target assertion hold?"
    # ------------------------------------------------------------------
    section("8. USEFULNESS QUERY — assume(candidate), assert(target)")
    print(f"  Candidate (assumed): {candidate}")
    print(f"  Target   (asserted): {target}\n")
    usefulness_prog = program.get_program_with_assertion(
        property_to_assert=target,
        assumptions=[candidate],
        for_llm=False,
    )
    print(usefulness_prog)

    # ------------------------------------------------------------------
    # 9. Generate the LLM-facing variant of the instrumented program
    # ------------------------------------------------------------------
    section("9. LLM-FACING INSTRUMENTED PROGRAM (for prompting)")
    llm_prog = program.get_program_with_assertion(
        property_to_assert=candidate,
        assumptions=[],
        for_llm=True,
    )
    print(llm_prog)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    section("SUMMARY")
    print(
        f"  This demo processed a C program with {program.num_loops} loop(s) and\n"
        f"  1 target assertion. The AstProgram pipeline:\n"
        f"\n"
        f"    1. Preprocessed the raw SV-COMP code (gcc -E, builtin rewrites)\n"
        f"    2. Parsed into a pycparser AST\n"
        f"    3. Replaced the target assert with TARGET_ASSERT_MARKER()\n"
        f"    4. Inserted INVARIANT_MARKER_1() at the loop head\n"
        f"    5. Built an LLM-facing AST (target assert visible, nondet -> rand)\n"
        f"    6. Generated correctness & usefulness verification programs\n"
        f"\n"
        f"  The correctness program checks: is the candidate a valid invariant?\n"
        f"  The usefulness program checks: does the invariant help prove the target?\n"
        f"  Both programs include the PATCH header (assert/assume definitions)\n"
        f"  and are ready to be fed to a verifier like UAutomizer.\n"
    )


if __name__ == "__main__":
    main()
