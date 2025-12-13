def syntactic_validation(content: str) -> bool:
    """
    Validate that the candidate invariant is a syntactically correct boolean expression
    without side effects (no assignments or mutations).
    
    Based on paper requirements:
    - Must be a boolean condition over program state
    - Must not contain assignments (e.g., a += 1, a = 5)
    - Must be safely interpretable as a state predicate
    """
    # Early rejection of empty or whitespace-only strings
    if not content or not content.strip():
        return False
    
    try:
        import pycparser.c_parser
        from pycparser.c_ast import NodeVisitor
        
        # Parse the predicate as a C expression wrapped in an assert
        # This allows us to check its AST structure
        wrapped_expr = f"void test() {{ assert({content}); }}"
        ast = pycparser.c_parser.CParser().parse(wrapped_expr)
        
        # Visitor class to check for assignment operations
        class AssignmentVisitor(NodeVisitor):
            def __init__(self):
                self.has_assignment = False
            
            def visit_Assignment(self, node):
                self.has_assignment = True
                return False  # Don't traverse children once assignment found
        
        visitor = AssignmentVisitor()
        visitor.visit(ast)
        
        # If the predicate contains any assignment, it's invalid
        if visitor.has_assignment:
            return False
        
        # Reject increment/decrement operators anywhere (they mutate state)
        if '++' in content or '--' in content:
            return False
        
        return True
        
    except Exception as e:
        # If parsing fails, the predicate is likely malformed
        # Log the error but return False for safety
        print(f"Warning: Failed to parse predicate '{content}': {e}")
        return False