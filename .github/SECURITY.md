# Security

If you believe you have found a security vulnerability in this project, please report it responsibly.

## Reporting a vulnerability

**Do not** open a public GitHub issue for security-sensitive bugs.

Instead, please report the issue to the maintainers privately (e.g. via the repository owner’s contact information or a private disclosure process they provide). Include a clear description of the issue and steps to reproduce if possible.

We will acknowledge your report and work with you to understand and address it.

## Scope

This project is research code for loop invariant generation and program verification. Security-sensitive areas may include:

- Handling of credentials (API keys, tokens) via environment variables and `.env`
- Any code that executes untrusted input or runs external tools (e.g. verifiers) on user-provided programs

Thank you for helping keep this project safe.
