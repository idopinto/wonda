# Contributing to WONDA

Thank you for your interest in contributing. This document gives a short overview of how to get started.

## Development setup

1. Clone the repository and install dependencies (including dev extras):

   ```bash
   git clone https://github.com/idopinto/wonda.git
   cd wonda
   uv sync --extra dev
   ```

2. Run the test suite to confirm everything works:

   ```bash
   uv run pytest tests/ -v
   ```

   Tests do not require external tools (UAutomizer, runlim) or network access and are suitable for CI.

## Submitting changes

1. Open an issue or discuss in an existing one if the change is non-trivial.
2. Create a branch, make your changes, and add or update tests as appropriate.
3. Ensure the test suite passes: `uv run pytest tests/ -v`.
4. Open a pull request with a clear description of the change. The CI workflow will run the tests automatically.

## Code and config conventions

- The project uses [Hydra](https://hydra.cc/) for configuration; override options on the command line or in YAML under `configs/`.
- Default configs are set for public use (no telemetry or Hub push by default). When adding new features that touch W&B, Weave, or Hugging Face, keep defaults safe for first-time users.

## Questions

For questions or discussion, please open a GitHub issue.
