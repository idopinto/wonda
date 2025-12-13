import sys
from pathlib import Path
from src.utils.rewriter import Rewriter

def run_rewriter_on_c_file(c_file_path):
    """Run the Rewriter on a given C file and print the transformed code."""
    path = Path(c_file_path)
    print(f"Running Rewriter on {path}")
    print("="*80)
    print(path.read_text())
    print("="*80)
    if not path.exists():
        print(f"File {c_file_path} does not exist.")
        return
    rewriter = Rewriter(path)
    print(rewriter.new_code)

if __name__ == "__main__":
    root_path = Path(__file__).parent.parent.parent
    c_file_path = root_path / "test.c"
    run_rewriter_on_c_file(c_file_path)
