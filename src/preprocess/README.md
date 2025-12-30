┌─────────────────────────────────────────────────────────────┐
│  UAutomizer DISCOVERING invariants                          │
│  ─────────────────────────────────                         │
│  Uses: Abstract interpretation, automata, CEGAR            │
│  Can: Find complex relationships including non-linear      │
│  How: Computes fixpoints over abstract domains             │
└─────────────────────────────────────────────────────────────┘
                           ≠
┌─────────────────────────────────────────────────────────────┐
│  UAutomizer VERIFYING an assertion                          │
│  ─────────────────────────────────                          │
│  Uses: SMT solver (Z3) directly                             │
│  Problem: "Prove this formula is an inductive invariant"   │
│  Requires: Solve non-linear arithmetic → often TIMEOUT     │
└─────────────────────────────────────────────────────────────┘