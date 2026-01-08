# SV-COMP Benchmark Categories

Reference: [SV-COMP 2025 Benchmarks](https://sv-comp.sosy-lab.org/2025/benchmarks.php)  
Repository: [sv-benchmarks GitLab](https://gitlab.com/sosy-lab/benchmarking/sv-benchmarks/-/tree/main/c)

## ReachSafety Categories

### sv-benchmarks/c/Arrays.set
**Array Operations** - Programs involving array manipulations including array examples, industry patterns, tiling, multidimensional arrays, crafted benchmarks, and memory-safety checks on arrays. Tests reasoning about array indices, bounds, and element relationships.

### sv-benchmarks/c/BitVectors.set
**Bit-Vector Operations** - Programs with bit-level manipulations and bit-vector arithmetic. Includes regression tests and loop-based bit operations. Requires precise modeling of fixed-width integer semantics.

### sv-benchmarks/c/ControlFlow.set
**Complex Control Flow** - Programs with intricate control structures including NT drivers, OpenSSL code, lock implementations, memory model checks, and integer overflow detection. Tests handling of complex branching and state transitions.

### sv-benchmarks/c/ECA.set
**Event-Condition-Action Systems** - Programs from the RERS (Rigorous Examination of Reactive Systems) challenges and psyco benchmarks. Models reactive systems with event-driven behavior.

### sv-benchmarks/c/Floats.set
**Floating-Point Arithmetic** - Programs with floating-point computations including CDFPL, CBMC/ESBMC regression tests, newlib functions, scientific computing loops, neural networks, and nonlinear operations. Requires accurate floating-point reasoning.

### sv-benchmarks/c/Heap.set
**Heap/Dynamic Memory** - Programs with dynamic memory allocation, linked lists, and pointer manipulations. Includes LDV (Linux Driver Verification) benchmarks and memory safety checks (valid-free, valid-deref, valid-memtrack).

### sv-benchmarks/c/Loops.set
**Loop Constructs** - Programs with various loop patterns including crafted loops, loop acceleration, invariant generation benchmarks (loop-invgen), non-linear arithmetic (nla-digbench), and VerifyThis competition tasks. Core category for loop invariant reasoning.

### sv-benchmarks/c/ProductLines.set
**Software Product Lines** - Programs representing software product lines with feature variability. Tests verification across multiple product configurations.

### sv-benchmarks/c/Recursive.set
**Recursive Functions** - Programs with recursive function calls including simple recursion, recursion with pointers, and recursified versions of loop benchmarks. Tests handling of call stacks and recursive invariants.

### sv-benchmarks/c/Sequentialized.set
**Sequentialized Concurrency** - Sequential encodings of concurrent programs from SystemC models and pthread implementations. Allows testing concurrent behavior without a full concurrency model.

### sv-benchmarks/c/XCSP.set
**Constraint Satisfaction** - Constraint satisfaction problems (CSP) encoded as C verification tasks. Tests constraint propagation and search capabilities.

## Other Categories

### sv-benchmarks/c/Combinations.set
**Combined Challenges** - Programs that combine multiple verification challenges (arrays + loops, heap + recursion, etc.). Tests tool robustness across feature combinations.

### sv-benchmarks/c/Hardware.set
**Hardware Verification** - Programs derived from hardware designs and hardware-related software. Includes bit-precise operations common in hardware modeling.

### sv-benchmarks/c/Hardness.set
**Particularly Difficult Tasks** - A curated set of especially challenging verification tasks that stress-test verifier capabilities.

### sv-benchmarks/c/Fuzzle.set
**Fuzzing Puzzles** - Verification tasks designed to challenge fuzzing-based approaches. Often contains programs with hard-to-reach error states.

---

## Verification Property

Most ReachSafety tasks use the specification:
```
CHECK( init(main()), LTL(G ! call(reach_error())) )
```
This means: starting from `main()`, globally (always) the function `reach_error()` should never be called.
