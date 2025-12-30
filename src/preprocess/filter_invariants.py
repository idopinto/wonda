import json


def is_trivial_invariant(invariant_text: str) -> bool:
    """
    Check if an invariant is trivial or not learnable from source code.

    Filters out invariants that:
    - Are exactly "1" (true in C)
    - Reference variables that don't exist in the source code (e.g., "!(cond == 0)"
      when cond is only a function parameter, not a program variable)
    """
    # Normalize whitespace for comparison
    normalized = invariant_text.strip()

    # Filter out trivial invariants
    trivial_patterns = [
        "1",  # Always true
        "!(cond == 0)",  # References cond which doesn't exist as program variable
    ]

    return normalized in trivial_patterns


def filter_invariants_in_file(input_path: str, output_path: str):
    """
    Load the dataset JSON, remove trivial invariants using is_trivial_invariant,
    and re-save the filtered JSON.
    """
    with open(input_path, "r") as f:
        data = json.load(f)

    filtered_data = []
    for entry in data:
        if "invariants" in entry and isinstance(entry["invariants"], list):
            filtered_invs = [
                inv
                for inv in entry["invariants"]
                if not is_trivial_invariant(inv.get("invariant", ""))
            ]
            entry = dict(entry)  # shallow copy
            entry["invariants"] = filtered_invs
        filtered_data.append(entry)

    with open(output_path, "w") as f:
        json.dump(filtered_data, f, indent=2)


if __name__ == "__main__":
    easy_path = "data/InvBench/Dataset/Evaluation/Easy/example_train.json"
    hard_path = "data/InvBench/Dataset/Evaluation/Hard/example_train.json"
    easy_filtered_path = "data/InvBench/Dataset/Evaluation/Easy/easy_eval.filtered.json"
    hard_filtered_path = "data/InvBench/Dataset/Evaluation/Hard/hard_eval.filtered.json"
    filter_invariants_in_file(easy_path, easy_filtered_path)
    filter_invariants_in_file(hard_path, hard_filtered_path)
