# Architecture and Workflow

```mermaid
flowchart TD
    A[Purchase request] --> B{Architecture}
    B -->|Single Agent| C[Agent loop]
    B -->|Staged Two-Agent| D[Analyst loop]
    D --> E[StructuredEvidencePack]
    E --> F[Reviewer]
    C --> G[Tool calls]
    D --> G
    G --> H[Pandas dataframes]
    G --> I[Mock vendor-risk API]
    H --> J[Deterministic policy rules]
    I --> J
    F --> J
    J --> K[Validated ProcurementOutput]
    K --> L[Evidence and risk display]
    L --> M[Human approve / reject / override]
```

## Assumptions and boundaries

- The CSV, JSON, and policy files under `data/` are the complete synthetic snapshot for the assessment.
- Date-based checks use the policy reference date `2026-09-30`, not the machine clock.
- Pandas tools are deterministic; the model proposes structured output but cannot purchase software, approve spend, change budgets, or accept legal terms.
- Vendor-risk API failures are material uncertainty. The vendor tool records unavailable evidence and falls back only to the local snapshot where possible.
- The Analyst gathers budget, catalog-overlap, and vendor-risk evidence. The Reviewer applies deterministic policy output and formats the final recommendation.
- Each pure-Python tool loop is capped at five iterations. Human review remains mandatory for approval decisions and exceptions.
