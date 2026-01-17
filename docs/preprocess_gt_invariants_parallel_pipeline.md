# Preprocess GT Invariants Parallel - Data Pipeline

## Main Pipeline Flow

```mermaid
flowchart TD
    Start([Start]) --> Load[Load Data]
    Load --> Entry[For Each Entry: Parse to AST]
    Entry --> Inv[For Each Invariant: Prettify]
    
    Inv --> Branch{Verbose?}
    
    Branch -->|Yes| LLM[Generate N Candidates via LLM]
    LLM --> Dedup[Deduplicate]
    Dedup --> Verify[Verify All in Parallel]
    
    Branch -->|No| Verify
    
    Verify --> Grade[Compute Quality Grade]
    Grade --> Filter{Grade > 0?}
    
    Filter -->|Yes| Save[Save Entry]
    Filter -->|No| Skip[Skip]
    
    Save --> MoreInv{More Invariants<br/>in Entry?}
    Skip --> MoreInv
    MoreInv -->|Yes| Inv
    MoreInv -->|No| MoreEntry{More Entries?}
    MoreEntry -->|Yes| Entry
    MoreEntry -->|No| Hub[Push to Hub]
    Hub --> End([End])
    
    style Start fill:#90EE90
    style End fill:#FFB6C1
    style LLM fill:#87CEEB
    style Verify fill:#FFD700
    style Save fill:#98FB98
```

## Key Components

### Quality Grading System
- **Grade 0**: Not correct (syntactically invalid or verification failed)
- **Grade 1**: Correct but target property doesn't hold
- **Grade 2**: Correct, target holds, but no speedup
- **Grade 3**: Correct, target holds, has speedup

### Parallel Processing
- **LLM Generation**: N candidates generated in single API call
- **Verification**: All candidates verified in parallel (ThreadPoolExecutor)
- **Per-Candidate Checks**: Correctness + Usefulness run in parallel (2 workers)

### Verbose Detection
- Checks disjunct count (||) and character length
- Only verbose invariants get LLM simplification
- Non-verbose invariants verified as-is
