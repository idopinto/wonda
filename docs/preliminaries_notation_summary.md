# Preliminaries Notation Summary

## Core Notation

| Symbol | Meaning |
|--------|---------|
| $P$ | Program (source code) |
| $\mathrm{Loc}(P)$ | Set of program locations in $P$ |
| $\mathcal{L}(P) \subseteq \mathrm{Loc}(P)$ | Loop entry locations in $P$ |
| $\langle p, q \rangle$ | Specification (precondition $p$, postcondition $q$) |
| $\langle p, P, q \rangle$ | Verification query |
| $(l, \varphi)$ | Candidate invariant (location $l$, predicate $\varphi$) |
| $A = \{(l_i, \varphi_i)\}$ | Set of candidate invariants |
| $\langle p, P, q, A \rangle$ | Annotated verification query |
| $\mathcal{V}$ | Verification oracle |
| $\varphi@l$ | Property $\varphi$ holds at location $l$ |
| $P_l^{+\texttt{assume}(\varphi)}$ | Program $P$ with `assume(φ)` inserted at $l$ |

## Verification Oracle

$$\mathcal{V}(p, P, q) \in \{\textsc{True}, \textsc{False}, \textsc{Unknown}\}$$

- **TRUE**: Query is valid (specification holds)
- **FALSE**: Counterexample exists
- **UNKNOWN**: Timeout or incompleteness

**Soundness requirement**: Conclusive results are guaranteed correct.

## Verification Primitives

- `assume(φ)`: Restricts traces to those satisfying φ. Semantics: `if (¬φ) halt`
- `assert(φ)`: Defines correctness property. Semantics: `if (¬φ) goto ERROR`

A program is **safe** iff ERROR is unreachable.

## Verification Procedure (following InvBench)

### Case $A = \emptyset$
Single query: $\mathcal{V}(p, P, q)$

### Case $|A| = 1$ with $A = \{(l, \varphi)\}$

Two queries **in parallel**:

1. **Invariant Check** ($\mathcal{V}_1$): $\mathcal{V}(p, P, \varphi@l)$
   - Checks if $\varphi$ holds at location $l$
   
2. **Sufficiency Check** ($\mathcal{V}_2$): $\mathcal{V}(p, P_l^{+\texttt{assume}(\varphi)}, q)$
   - Checks if $q$ holds when assuming $\varphi$ at $l$

### Decision Table

| $\mathcal{V}_1$ | $\mathcal{V}_2$ | Outcome |
|-----------------|-----------------|---------|
| TRUE | TRUE | TRUE |
| * | FALSE | FALSE |
| otherwise | | UNKNOWN |

## Tasks

### Program Verification Task
Determine: $\mathcal{V}(p, P, q) \in \{\textsc{True}, \textsc{False}\}$

### Invariant Generation Task
Given $\langle p, P, q \rangle$ and location $l \in \mathcal{L}(P)$, synthesize $\varphi$ such that $(l, \varphi)$ is:
1. **Correct**: $\varphi$ holds at $l$ on all executions
2. **Sufficient**: assuming $\varphi$ at $l$ enables $\mathcal{V}$ to prove $q$

## Loop Invariant Requirements (Hoare Logic)

For loop `while B do S` with precondition $P$, postcondition $Q$, invariant $I$:

1. **Initiation**: $P \Rightarrow I$
2. **Consecution**: $\{I \land B\} S \{I\}$
3. **Sufficiency**: $I \land \neg B \Rightarrow Q$

## Key References

- Decision procedure soundness: InvBench (Wei et al., 2025)
- LEMUR calculus: Wu et al., 2023
