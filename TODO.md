# TODO: Roadmap for Advanced Upscaling & Coupled Solvers in XCALibre.jl

## 1. Monolithic Coupled Solvers
**Feasibility: High**
*   **Current State:** XCALibre solves equations sequentially (Segregated approach).
*   **Proposed Structure:** Leverage `BlockSparseMatrices.jl` or simply concatenate the CSR arrays into a single large system $[A_{uu} A_{up}; A_{pu} A_{pp}]$. 
*   **Switching Mechanism:** Use Julia's multiple dispatch to allow `solve_system!` to take a `Vector{ModelEquation}`.
*   **Benefits:** Faster convergence for highly coupled physics (e.g., Poroelasticity, high-Re flow).

## 2. Generic Newton Linearization
**Feasibility: High (Foundations implemented)**
*   **Current State:** I have implemented `NonLinearSi` and `NonLinearRobin` using `ForwardDiff.jl`.
*   **Proposed Improvement:** Extend the DSL to allow non-linear functions inside `Divergence` or `Laplacian`.
*   **Redefining Operators:** Redefine the `Operator` struct to optionally hold a `NonLinear` tag and a function. The `linearize_physics` tool should automatically handle the Jacobian of the entire residual $Res(\phi) = 0$.

## 3. Higher Order FV Operators
**Feasibility: Medium**
*   **Current State:** Mostly 2nd order (Linear/Upwind).
*   **Proposed:** Implement 3rd order QUICK or 4th order schemes.
*   **Matrix Sparsity:** Unlike OpenFOAM's standard `lduMatrix` (which is strictly 7-diagonal for hex), XCALibre builds actual `SparseMatrixCSR`. We can easily extend the connectivity to include second-neighbours (Stencil expansion) without breaking the solver architecture.

## 4. Gradient Transpose Operator ($\nabla u^T$)
**Feasibility: High**
*   **Problem in OpenFOAM:** Standard OpenFOAM stencils and matrix-free structures make direct manipulation of the full gradient tensor difficult in a single matrix assembly.
*   **XCALibre Advantage:** Since we have the full sparse matrix $A$, we can implement the transpose coupling directly.
*   **Application:** Essential for the full Stress Tensor in complex rheology and for accurate adjoint-based optimization.

## 5. Generic Periodic BCs with Rotations
**Feasibility: Completed (Initial implementation)**
*   I have already refactored `Periodic` to use `transform_point(transform, p)`. This should be further optimized for GPU kernels to avoid CPU-based mapping during every matrix assembly.
