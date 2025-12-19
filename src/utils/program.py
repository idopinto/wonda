from enum import Enum
from typing import List, Set, Dict, Optional
from copy import copy
import re
from src.utils.predicate import Predicate

PATCH = ('void assert(int cond) { if (!(cond)) { ERROR : { reach_error(); abort(); } } }\n'
         'void assume(int cond) { if (!cond) { abort(); } }\n')

# PATCH_LINES = ['void assert(int cond) { if (!(cond)) { ERROR : { reach_error(); abort(); } } }',
#                'void assume(int cond) { if (!cond) { abort(); } }']
class AssertionPointAttributes(Enum):
    BeforeLoop = 1
    InLoop = 2
    BeforeAssertion = 3
    BeginningOfLoop = 4
    EndOfLoop = 5

class Program:
    def __init__(self, lines: List[str], replacement: Dict[str, str]):
        self.lines: List[str] = []
        
        self.assertions: List[Predicate] = []  # The assertion to add after the corresponding line number
        self.lemmas: List[Predicate] = []  # The lemmas to add after the corresponding line number
        self.assertion_points: Dict[
            int, Set[AssertionPointAttributes]] = {}  # Potentially adding assertions right after these lines

        self.replacement_for_GPT: Dict[str, str] = replacement
        self.in_loop: Dict[int, int] = dict()
        self.unclosed_brackets: Dict[int, int] = {-1: 0}

        self.number_of_loops = 0

        last_line_in_loop = False
        left_bracket = 0
        for line in lines:
            if last_line_in_loop:
                self.in_loop[len(self.lines) - 1] = True
            else:
                self.in_loop[len(self.lines) - 1] = False

            if last_line_in_loop and "{" in line:
                left_bracket += 1
            if last_line_in_loop and "}" in line:
                left_bracket -= 1
                if left_bracket == 0:
                    self.add_assertion_point(len(self.lines), AssertionPointAttributes.EndOfLoop)
                    #self.add_assertion_point(len(self.lines) - 1, AssertionPointAttributes.InLoop)
                    last_line_in_loop = False
            # print(line)
            stripped_line = line.strip()
            if stripped_line and stripped_line.split("(")[0] == "assert":
                # print(stripped_line)
                # print(stripped_line.split("(")[0])
                self.add_assertion_point(len(self.lines) - 1, AssertionPointAttributes.BeforeAssertion)
                if last_line_in_loop:
                    self.add_assertion_point(len(self.lines) - 1, AssertionPointAttributes.InLoop)

                result = re.search(r'assert\((.*?)\);', line)
                if result:
                    self.assertions.append(Predicate(result.group(1), len(self.lines) - 1))

            elif stripped_line and stripped_line.split("(")[0] == "assume":
                result = re.search(r'assume\((.*?)\);', line)
                if result:
                    self.lemmas.append(Predicate(result.group(1), len(self.lines) - 1))
            else:
                self.lines.append(line)
                if stripped_line and stripped_line.split()[0] in ["for", "do", "while"]:
                    self.number_of_loops += 1
                    self.add_assertion_point(len(self.lines) - 2, AssertionPointAttributes.BeforeLoop)
                    if last_line_in_loop:
                        self.add_assertion_point(len(self.lines) - 2, AssertionPointAttributes.InLoop)
                    self.add_assertion_point(len(self.lines) - 1, AssertionPointAttributes.InLoop)
                    self.add_assertion_point(len(self.lines) - 1, AssertionPointAttributes.BeginningOfLoop)
                    if not last_line_in_loop:
                        assert ("{" in line)
                        left_bracket += 1
                    last_line_in_loop = True
                if "{" in line:
                    if "}" not in line:
                        self.unclosed_brackets[len(self.lines) - 1] = self.unclosed_brackets[len(self.lines) - 2] + 1
                    else:
                        self.unclosed_brackets[len(self.lines) - 1] = self.unclosed_brackets[len(self.lines) - 2]
                elif "}" in line:
                    self.unclosed_brackets[len(self.lines) - 1] = self.unclosed_brackets[len(self.lines) - 2] - 1
                else:
                    self.unclosed_brackets[len(self.lines) - 1] = self.unclosed_brackets[len(self.lines) - 2]

        self.in_loop[len(self.lines) - 1] = False
        if self.number_of_loops > 0:
            del self.assertion_points[min(self.assertion_points)]


    def add_assertion_point(self, line_number: int, attribute: AssertionPointAttributes):
        if line_number not in self.assertion_points:
            self.assertion_points[line_number] = set()
        self.assertion_points[line_number].add(attribute)

    @staticmethod
    def assume_predicate(lines: List[str], predicate: Predicate):
        lines[predicate.line_number] += f"\nassume({predicate.content});"

    @staticmethod
    def assert_predicate(lines: List[str], predicate: Predicate):
        lines[predicate.line_number] += f"\nassert({predicate.content});"

    def get_program_with_assertion(self, predicate: Optional[Predicate], assumptions: List[Predicate],
                                   assertion_points: Dict[int,str],
                                   forGPT : bool,
                                   dump=False):
        program = "" if forGPT else PATCH

        lines = copy(self.lines)
        if forGPT:
            for i, line in enumerate(lines):
                if line in self.replacement_for_GPT:
                    lines[i] = self.replacement_for_GPT[line]

        for lemma in self.lemmas:
            self.assume_predicate(lines, lemma)
        # print(f"    Lines after lemmas: {lines}")
        for line_number, name in assertion_points.items():
            lines[line_number] += f"\n// Line {name}"

        for assumption in assumptions:
            self.assume_predicate(lines, assumption)

        if predicate is not None:
            self.assert_predicate(lines, predicate)

        if forGPT:
            if predicate is not None:
                for i, line in enumerate(lines):
                    if i < predicate.line_number:
                        program += line + "\n"
                    elif i >= predicate.line_number and (not self.in_loop[i]):
                        program += line + "\n"
                        for m in range(self.unclosed_brackets[i]):
                            if m == self.unclosed_brackets[i] - 1:
                                program += "return 1;\n"
                            program += "}\n"
                        break
                    else:
                        program += line + "\n"
            else:
                # If predicate is None, include all lines
                for line in lines:
                    program += line + "\n"
        else:
            for line in lines:
                program += line + "\n"
        program = program[:-1]

        # if dump:
        #     print("----------------------")
        #     print(program)
        #     print("----------------------")
        return program

    def decide_assertion_point(self, goal: Predicate):
        """
        - If the assertion is in the loop:
            - if the assertion is the first line of the loop
                - beginning of the loop
            - else:
                - before the assertion
        - else:
            - the beginning of the closest loop
        """
        print("Deciding assertion point...")
        closest_line = None
        if AssertionPointAttributes.InLoop in self.assertion_points[goal.line_number]:
            if AssertionPointAttributes.BeginningOfLoop in self.assertion_points[goal.line_number]:
                #print("assertion is the beginning of the loop, get the line right before the loop")
                assert(AssertionPointAttributes.BeforeLoop in self.assertion_points[goal.line_number - 1])
                closest_line = goal.line_number - 1
            else:
                #print("assertion is in the loop, find the beginning of the closest loop")
                tmp = goal.line_number
                while tmp >= 0:
                    tmp -= 1
                    if (tmp in self.assertion_points and
                            AssertionPointAttributes.BeginningOfLoop in self.assertion_points[tmp]):
                        closest_line = tmp
                        break
        else:
            #print("assertion is right after a loop, find the beginning of the closest loop")
            tmp = goal.line_number
            while tmp >= 0:
                tmp -= 1
                if (tmp in self.assertion_points and (not self.in_loop[tmp - 1]) and
                        AssertionPointAttributes.BeginningOfLoop in self.assertion_points[tmp]):
                    closest_line = tmp
                    break

        if closest_line is None:
            return None, None
        else:
            print(f"Deciding assertion point done: picked line {closest_line}.")
        return closest_line, self.assertion_points[closest_line]
        
    def __repr__(self):
        lines = []
        lines.append("--------------------------------- Program ---------------------------------------")
        
        # Program lines (clean, without metadata)
        lines.append(f"\n📝 Program lines without assertions ({len(self.lines)} total):")
        lines.append("-" * 40)
        for i, line in enumerate(self.lines):
            lines.append(f"  {i:2d}: {line}")
        
        # Assertions
        lines.append(f"\n🎯 Assertions ({len(self.assertions)} found):")
        lines.append("-" * 40)
        if self.assertions:
            for i, assertion in enumerate(self.assertions):
                lines.append(f"  {i+1}.After Line {assertion.line_number}: assert({assertion.content})")
        else:
            lines.append("  No assertions found")
        
        # Lemmas
        lines.append(f"\n📋 Lemmas ({len(self.lemmas)} found):")
        lines.append("-" * 40)
        if self.lemmas:
            for i, lemma in enumerate(self.lemmas):
                lines.append(f"  {i+1}. After Line {lemma.line_number}: assume({lemma.content})")
        else:
            lines.append("  No lemmas found")
        
        # Loop summary
        lines.append("\n🔄 Loop analysis:")
        lines.append("-" * 40)
        lines.append(f"  Total loops: {self.number_of_loops}")

        # Assertion points
        lines.append(f"\n📍 Assertion points ({len(self.assertion_points)} locations):")
        lines.append("-" * 40)
        if self.assertion_points:
            for line_num, attributes in sorted(self.assertion_points.items()):
                attr_names = [attr.name for attr in attributes]
                lines.append(f"  Line {line_num:2d}: {', '.join(attr_names)}")
        else:
            lines.append("  No assertion points found")
        
        # GPT replacements
        lines.append(f"\n🔄 GPT replacements ({len(self.replacement_for_GPT)} mappings):")
        lines.append("-" * 40)
        if self.replacement_for_GPT:
            for before, after in self.replacement_for_GPT.items():

                lines.append(f"  BEFORE: {before}")
                lines.append(f"  AFTER:  {after}")
                lines.append("  " + "-" * 30)
        else:
            lines.append("  No replacements defined")
        
        lines.append("\n" + "-" * 80)
        return "\n".join(lines)


if __name__ == "__main__":
    from src.utils.program_normalizer import ProgramNormalizer

    code = """
extern void abort(void);
#include <assert.h>
void reach_error() { assert(0); }

#define WHITE 0
#define BLUE 1

typedef struct TSLL {
    struct TSLL *next;
    struct TSLL *prev;
    int data;
} SLL;

int main() {
    // create the head
    SLL *head = malloc(sizeof(SLL));
    head->next = NULL;
    head->prev = NULL;
    head->data = WHITE;

    SLL *x = head;

    // create an arbitrary white list
    while (__VERIFIER_nondet_int()) {
        // create a node
        x->next = malloc(sizeof(SLL));
        x->next->prev = x;
        x = x->next;
        x->data = WHITE;
        x->next = NULL;
    }

    // insert a blue guy
    if (__VERIFIER_nondet_int()) { // the blue guy will be the head
        x = malloc(sizeof(SLL));
        x->data = BLUE;
        x->next = head;
        x->prev = NULL;
        head = x;
    } else {
        // choose a predecessor of the blue guy
        x = head;
        while (x->next != NULL) {
            if (__VERIFIER_nondet_int()) {
                break;
            }
            x = x->next;
        }

        // insert the blue guy
        SLL *y = x->next;
        x->next = malloc(sizeof(SLL));
        x->data = BLUE;
        x->next = y;
        if (y != NULL) {
            x->prev = y->prev;
            y->prev = x;
        }
    }

    // check the invariant
    x = head;

    // look for the first blue guy
    while (x->data != BLUE) {
        x = x->next; // fails if there is no blue guy
    }

    // look for another blue guy
    x = x->next;
    while (x) {
        if (x->data == BLUE) {
            __VERIFIER_assert(0);
        }
        x = x->next;
    }

    // destroy the list
    x = head;
    while (x != NULL) {
        head = x;
        x = x->next;
        free(head);
    }

    return 0;
}"""
    normalizer = ProgramNormalizer(code=code, rewrite=True)
    print(normalizer.new_code, )
    program = Program(normalizer.lines_to_verify, normalizer.replacement)
    print(program)
    # labeled, name_to_line = _label_assertion_points(program.assertion_points, only_loop_beginnings=True)
    # print(labeled)
    # print(name_to_line)