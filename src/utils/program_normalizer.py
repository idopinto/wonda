from typing import Dict
import re
from pathlib import Path
from configs import global_config as GC
import tempfile
import os
import subprocess


def run_subprocess_and_get_output(command):
    p = subprocess.run(command.split(), capture_output=True)
    return p.stdout.decode('utf-8', errors='replace'), p.stderr.decode('utf-8', errors='replace')

class ProgramNormalizer:
    """
    Processes C code by removing comments, formatting, and replacing verification functions.
    
    Transforms C code with verification annotations into clean, formatted code suitable
    for analysis by replacing __VERIFIER_* functions with standard equivalents.
    """
    def __init__(self, filename: Path = None, rewrite=True, handle_reach_error=False, code: str = None):
        """Initialize rewriter with C code from file or directly from code string."""
        if code is not None:
            self.code = code.strip()
        elif filename is not None:
            self.code = filename.read_text().strip()
        else:
            raise ValueError("Either 'filename' or 'code' must be provided")
        self.new_code = self.code
        if rewrite:
            self.new_code = re.sub(r'^\s*#include\s+[<"].*?[>"]\s*$', '', self.new_code, flags=re.MULTILINE)
            self.remove_comments()
            self.remove_re_pattern(r'__attribute__\s*\(\(.*?\)\)')
            self.remove_re_pattern(r'printf\s*\([^)]*\)\s*;')
            self.remove_function("void reach_error")
            self.remove_function("void __VERIFIER_assert")
            # Replace verification functions with standard equivalents
            self.new_code = self.new_code.replace("__VERIFIER_assert", "assert")
            self.remove_function("void assert")
            self.remove_function("void assume_abort_if_not")
            self.remove_function("void assume")
            self.remove_externs()
            self.new_code = self.new_code.replace("assume_abort_if_not", "assume")
            self.clang_format()
            self.remove_empty_lines()
            self.has_reach_error = False
            if handle_reach_error:
                self.replace_reach_error_with_assertion()
            # Join multi-line assertions into single lines (my addition)
            self.join_multiline_assertions()

        self.lines_to_verify = self.new_code.split("\n")
        # Replace nondeterministic functions with random values
        self.remove_verifier_nondet()
        self.lines_for_gpt = self.new_code.split("\n") 
        self.replacement: Dict[str, str] = {}
        assert(len(self.lines_for_gpt) == len(self.lines_to_verify))
        for i in range(len(self.lines_to_verify)):
            if self.lines_to_verify[i] != self.lines_for_gpt[i]:
                self.replacement[self.lines_to_verify[i]] = self.lines_for_gpt[i]

    def find_all_loops(self):
        """Count total number of loops in the code using clang AST."""
        tmp_file = Path("tmp.c")
        tmp_file.write_text(self.new_code)
        command = "clang -cc1 -ast-dump tmp.c"
        output, err = run_subprocess_and_get_output(command)
        tmp_file.unlink()
        num_loops = output.count("ForStmt") + output.count("WhileStmt") + \
            output.count("DoStmt")
        return num_loops

    def remove_re_pattern(self, pattern):
        """Remove text matching the given regex pattern."""
        self.new_code = re.sub(pattern, '', self.new_code)

    def remove_function(self, func_name: str):
        """Remove function definition from code by finding matching braces."""
        c_code = self.new_code
        # Optionally match 'extern' (possibly with whitespace) before func_name
        pattern = rf'(?:extern\s+)?{re.escape(func_name)}\s*\('
        match = re.search(pattern, c_code)
        function_index = match.start() if match else -1

        if function_index == -1:
            return None
        open_brackets = 0
        close_brackets = 0

        # Find the block that defines the function by matching braces
        for i in range(function_index, len(c_code)):
            if c_code[i] == '{':
                open_brackets += 1
            elif c_code[i] == '}':
                close_brackets += 1

                if open_brackets == close_brackets:
                    break
        self.new_code = c_code[: function_index] + c_code[i + 1: ]
        return


    def nondet_type(self, type_str : str):
        """Convert nondet type to C cast expression."""
        if type_str == "uchar":
            return "(unsigned char)"
        elif type_str == "char":
            return "(signed char)"
        elif type_str == "uint":
            return "(unsigned int)"
        else:
            return f"({type_str})"

    def get_tokens_with_verifier_nondet(self, input_string):
        """Extract all __VERIFIER_nondet_* tokens from input string."""
        pattern = r'\b__VERIFIER_nondet_\w+\b'
        tokens = set(re.findall(pattern, input_string))
        return tokens

    def remove_verifier_nondet(self):
        
        """Replace __VERIFIER_nondet_* functions with random value casts."""
        tokens = self.get_tokens_with_verifier_nondet(self.new_code)
        for token in tokens:
            pattern = token + "()"
            replacement = self.nondet_type(token.split("__VERIFIER_nondet_")[1]) + " rand()"
            self.new_code = self.new_code.replace(pattern, replacement)

    def remove_externs(self):
        """Remove extern declarations and __extension__ prefixes."""
        pattern = r'\bextern\s+.*?;'
        functions = re.findall(pattern, self.new_code, re.MULTILINE | re.DOTALL)
        for function in functions:
            self.new_code = self.new_code.replace(function, "")
        self.new_code = self.new_code.strip()
        lines = self.new_code.split("\n")
        new_lines = []
        for line in lines:
            if line.strip()[:13] == "__extension__":
                new_lines.append(line[13:])
            else:
                new_lines.append(line)
        self.new_code = "\n".join(new_lines)

    def clang_format(self):
        """Format code using clang-format with custom style configuration."""
        # Use unique temporary filename to avoid conflicts with parallel jobs
        fd, path = tempfile.mkstemp(suffix='.c', prefix='clang_format_')
        os.close(fd)
        tmp_file = Path(path)
        try:
            with tmp_file.open('w') as out_file:
                out_file.write(self.new_code)
            command = f"clang-format-15 --style=file:{GC.PATH_TO_CLANG_FORMAT} {tmp_file}"
            output, err = run_subprocess_and_get_output(command)
            self.new_code = output
        finally:
            if tmp_file.exists():
                tmp_file.unlink()

    def remove_empty_lines(self):
        """Remove empty lines from the code."""
        lines = self.new_code.split("\n")
        new_lines = []
        for line in lines:
            if line == "":
                continue
            else:
                new_lines.append(line)
        self.new_code = "\n".join(new_lines)

    def join_multiline_assertions(self):
        """Join multi-line assertions (assert/assume) into single lines.
        
        This method processes the code to find assertions that span multiple lines
        and joins them into a single line. Also splits any content after the semicolon
        (like closing braces) onto a separate line.
        
        Note: __VERIFIER_assert should already be replaced with assert by this point.
        """
        lines = self.new_code.split("\n")
        new_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Check if this line starts an assertion
            if not (stripped.startswith("assert(") or stripped.startswith("assume(")):
                new_lines.append(line)
                i += 1
                continue
            
            # Found an assertion - join it until we find the closing paren and semicolon
            paren_count = 0
            joined_line = line  # Preserve indentation
            found_semicolon = False
            
            # Count parentheses in the first line
            for char in stripped:
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
                elif char == ';' and paren_count == 0:
                    found_semicolon = True
                    break
            
            # Continue joining lines until we find the semicolon
            i += 1
            while i < len(lines) and not found_semicolon:
                current_line = lines[i]
                current_stripped = current_line.strip()
                
                # Skip empty lines
                if not current_stripped:
                    i += 1
                    continue
                
                # Stop if we hit a new statement
                if current_stripped.startswith(('if ', 'while ', 'for ', 'return ', 'break', 'continue', '}')):
                    break
                
                # Join the line
                joined_line += " " + current_stripped
                
                # Count parentheses and check for semicolon
                for char in current_stripped:
                    if char == '(':
                        paren_count += 1
                    elif char == ')':
                        paren_count -= 1
                    elif char == ';' and paren_count == 0:
                        found_semicolon = True
                        break
                
                i += 1
            
            # Split content after semicolon if present
            if found_semicolon:
                semicolon_idx = joined_line.find(';')
                if semicolon_idx != -1:
                    assertion_part = joined_line[:semicolon_idx + 1]
                    remaining_part = joined_line[semicolon_idx + 1:].strip()
                    
                    new_lines.append(assertion_part)
                    if remaining_part:
                        new_lines.append(remaining_part)
                else:
                    new_lines.append(joined_line)
            else:
                new_lines.append(joined_line)
        
        self.new_code = "\n".join(new_lines)

    def remove_comments(self):
        fd, path = tempfile.mkstemp(suffix='.c', prefix='gcc_preprocess_')
        os.close(fd)
        tmp_file = Path(path)
        try:
            tmp_file.write_text(self.new_code)
            command = f"gcc -E -P {tmp_file}"
            output, err = run_subprocess_and_get_output(command)
            self.new_code = output
        finally:
            if tmp_file.exists():
                tmp_file.unlink()

    def replace_reach_error_with_assertion(self):
        """Replace reach_error() calls with assert(!condition) statements."""
        c_code = self.new_code
        indices_object = re.finditer(pattern='reach_error', string=c_code)
        indices = [index.start() for index in indices_object]
        indices.reverse()

        for function_index in indices:
            self.has_reach_error = True
            assertion_start = None
            assertion_end = None
            block_end = None
            # Find the block that defines the function
            for i in range(function_index, 0, -1):
                if c_code[i] == ')' and assertion_end is None:
                    assertion_end = i
                if c_code[i:i + 3] == "if ":
                    assert(assertion_start is None and assertion_end is not None)
                    assertion_start = i + 3
                    break

            for i in range(function_index, len(c_code)):
                if c_code[i] == '}':
                    block_end = i
            condition = c_code[assertion_start: assertion_end + 1]
            condition = f"assert(!{condition});"
            c_code = c_code[:assertion_start - 3] + condition + c_code[block_end + 1:]

        self.new_code = c_code


if __name__ == "__main__":
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
    normalizer = ProgramNormalizer(code=code)
    print(normalizer.new_code)