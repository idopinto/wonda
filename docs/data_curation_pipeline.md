# Data Curation Pipeline

```mermaid
flowchart TD
    subgraph Input["📥 Input"]
        A[Load Training Data]
    end
    
    subgraph Process["🔄 Per Entry"]
        B[Parse Program]
        C[Prettify Invariant]
        D{Verbose?}
    end
    
    subgraph Verbose["📝 Verbose Path"]
        E[LLM: Generate N Candidates]
        F[Deduplicate]
        G[Verify ∥]
    end
    
    subgraph NonVerbose["📋 Non-Verbose"]
        H[Verify As-Is]
    end
    
    subgraph Filter["✅ Quality Filter"]
        I{Grade > 0?}
    end
    
    subgraph Output["💾 Output"]
        J[Save Entry]
        K[Push to Hub]
    end

    A --> B
    B --> C
    C --> D
    D -->|Yes| E
    D -->|No| H
    E --> F
    F --> G
    G --> I
    H --> I
    I -->|Yes| J
    I -->|No| L[Discard]
    J --> K
    J -.->|Next| B

    style A fill:#4285F4,color:#fff
    style E fill:#34A853,color:#fff
    style G fill:#EA4335,color:#fff
    style H fill:#EA4335,color:#fff
    style I fill:#FBBC04,color:#000
    style K fill:#8A2BE2,color:#fff
```

## Pipeline Summary

| Stage | Description |
|-------|-------------|
| **Load** | Load JSON training data with invariants |
| **Parse** | Parse C program AST once per entry |
| **Prettify** | Clean and format invariant text |
| **Verbose Check** | `min_disjuncts ≤ \|\| count ≤ max_disjuncts` AND `min_chars ≤ len ≤ max_chars` |
| **LLM Generation** | Generate N simplified candidates via Together API |
| **Deduplicate** | Remove exact duplicates by normalized content |
| **Verify** | Run UAutomizer: Correctness ∥ Usefulness (check speedup) |
| **Filter** | Keep only `quality_grade > 0` |
| **Save** | Write qualifying entries to JSON |
| **Hub** | Push dataset to HuggingFace Hub |

## Quality Grades

| Grade | Criteria |
|-------|----------|
| 0 | Not correct |
| 1 | Correct but target property doesn't hold |
| 2 | Correct + target holds, no speedup |
| 3 | Correct + target holds + speedup ✓ |
