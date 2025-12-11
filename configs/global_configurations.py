from dataclasses import dataclass
from pathlib import Path

# Single source of truth for project root
# This file is in configs/, so go up 1 level to reach inv-gen/
ROOT_DIR = Path(__file__).parent.parent

# Common paths derived from root
DATASET_DIR = ROOT_DIR / "data"
EVALUATION_DATASET_DIR = DATASET_DIR / "eval"
TRAINING_DATASET_DIR = DATASET_DIR / "train"
PROPERTIES_DIR = DATASET_DIR / "properties"
TOOLS_DIR = ROOT_DIR / "tools"
# EXPERIMENTS_DIR = DATASET_DIR / "eval" / "experiments"
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
PATH_TO_CLANG_FORMAT = ROOT_DIR / "configs" / "clang_format"

# UAutomizer paths
UAUTOMIZER_PATHS = {
    "23": TOOLS_DIR / "UAutomizer23" / "Ultimate.py", # (0.2.2-dev-2329fc7): Java 11 (jdk-11.0.23)
    "24": TOOLS_DIR / "UAutomizer24" / "Ultimate.py", # (0.2.4-dev-0e0057c): Java 11 (jdk-11.0.23)
    "25": TOOLS_DIR / "UAutomizer25" / "Ultimate.py", # (0.3.0-dev-d790fec): Java 21 [default] (jdk-21.0.1)
    "26": TOOLS_DIR / "UAutomizer26" / "Ultimate.py", # (0.3.0-dev-9c83a1c48e): Java 21 (jdk-21.0.1)
}
#  The Ultimate.py scripts automatically detect and use the correct Java version based on the UAutomizer directory.
#  No manual switching is required.

@dataclass
class DefaultVerificationConfig:
    """Default verification configuration. This is shared across all verification calls by this instance. """
    version: str = '25'
    property_file_path: Path = PROPERTIES_DIR / "unreach-call.prp"
    arch: str = '32bit'
    timeout_seconds: float = 600.0

    def __post_init__(self):
        self.uautomizer_path = UAUTOMIZER_PATHS[self.version]