# Graph Report - comphy  (2026-10-08)

## Corpus Check
- Corpus is ~48,959 words - fits in a single context window. You may not need a graph.

## Summary
- 58 nodes · 81 edges · 13 communities (3 shown, 10 thin omitted)
- Extraction: 85% EXTRACTED · 15% INFERRED · 0% AMBIGUOUS · INFERRED: 12 edges (avg confidence: 0.92)
- Token cost: 2,800 input · 1,050 output

## Community Hubs (Navigation)
- ReportLab PDF Publishing Pipeline
- Convergence Acceleration & Stratified Sampling
- NumberedCanvas Pagination Engine
- Jittered Grid Verification Harness
- Standard & Sobol QMC Sampling
- Matplotlib Visualization Suite
- Execution Timing Benchmark Suite
- Pure Python Iterative Estimation
- Vectorized NumPy Pipeline
- Central Limit Theorem Error Scaling
- Geometric Unit-Square Sampling Principle
- Quasi-Monte Carlo & BVHK Degeneration
- Course Policy & Collaboration Scope

## God Nodes (most connected - your core abstractions)
1. `run_numerical_verification()` - 9 edges
2. `NumberedCanvas` - 6 edges
3. `main()` - 6 edges
4. `estimate_pi_stratified()` - 5 edges
5. `plot_errors()` - 5 edges
6. `estimate_pi_standard()` - 4 edges
7. `estimate_pi_sobol()` - 4 edges
8. `estimate_pi_loops()` - 4 edges
9. `estimate_pi_numpy()` - 4 edges
10. `benchmark_function()` - 4 edges

## Surprising Connections (you probably didn't know these)
- `Monte Carlo Principle Unit Square / Quarter Circle Diagram` --conceptually_related_to--> `Monte Carlo Pi Estimation Task`  [INFERRED]
  exercises/exercise_00/mc_principle.png → exercises/exercise_00/exercise_00.pdf
- `Monte Carlo Error Scaling Plot (O(1/?N))` --conceptually_related_to--> `Central Limit Theorem Scaling (O(1/?N))`  [INFERRED]
  exercises/exercise_00/errors.png → exercises/exercise_00/exercise_00_summary.pdf
- `Convergence Comparison Figure (Standard vs Stratified vs Sobol)` --conceptually_related_to--> `Stratified Jittered Grid Sampling (O(N^-0.75))`  [INFERRED]
  exercises/exercise_00/gods_monte_carlo_convergence.png → exercises/exercise_00/exercise_00_summary.pdf
- `Agentic AI Verification Rationale` --conceptually_related_to--> `Exercise 0: First Steps with Agentic AI`  [INFERRED]
  exercises/exercise_00/exercise_00_summary.pdf → exercises/exercise_00/exercise_00.pdf
- `Exercise 0 Scientific Summary Report` --references--> `Exercise 0: First Steps with Agentic AI`  [EXTRACTED]
  exercises/exercise_00/exercise_00_summary.pdf → exercises/exercise_00/exercise_00.pdf

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Monte Carlo Pi Estimation Methods** — exercises_exercise_00_exercise_00_summary_central_limit_theorem, exercises_exercise_00_exercise_00_summary_stratified_jittered_grid, exercises_exercise_00_exercise_00_summary_sobol_qmc [EXTRACTED 0.95]

## Communities (13 total, 10 thin omitted)

### Community 0 - "ReportLab PDF Publishing Pipeline"
Cohesion: 0.15
Nodes (3): Exercise 0: First Steps with Agentic AI, Exercise 0 Scientific Summary Report, NumPy Vectorization vs Python Loop Benchmark Chart

### Community 1 - "Convergence Acceleration & Stratified Sampling"
Cohesion: 0.40
Nodes (3): Convergence Acceleration Challenge (>1/?N), Stratified Jittered Grid Sampling (O(N^-0.75)), Convergence Comparison Figure (Standard vs Stratified vs Sobol)

### Community 3 - "Jittered Grid Verification Harness"
Cohesion: 0.40
Nodes (3): estimate_pi_stratified(), main(), run_numerical_verification()

## Knowledge Gaps
- **6 isolated node(s):** `Course Collaboration Policy`, `Monte Carlo Pi Estimation Task`, `Convergence Acceleration Challenge (>1/?N)`, `Central Limit Theorem Scaling (O(1/?N))`, `Sobol Quasi-Monte Carlo (QMC)` (+1 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 29 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `NumPy Vectorization vs Python Loop Benchmark Chart` connect `ReportLab PDF Publishing Pipeline` to `Execution Timing Benchmark Suite`?**
  _High betweenness centrality (0.406) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `main()` (e.g. with `estimate_pi_loops()` and `estimate_pi_numpy()`) actually correct?**
  _`main()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Course Collaboration Policy`, `Monte Carlo Pi Estimation Task`, `Convergence Acceleration Challenge (>1/?N)` to the rest of the system?**
  _6 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Why does `NumberedCanvas` connect `NumberedCanvas Pagination Engine` to `ReportLab PDF Publishing Pipeline`?**
  _High betweenness centrality (0.159) - this node is a cross-community bridge._