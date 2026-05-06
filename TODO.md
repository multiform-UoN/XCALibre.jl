# TODO: Roadmap for Advanced Upscaling & Coupled Solvers in XCALibre.jl

## 1. Monolithic Coupled Solvers
**Feasibility: High**
*   **Current State:** XCALibre solves equations sequentially (Segregated approach).
*   **Proposed Structure:** Leverage `BlockSparseMatrices.jl` or simply concatenate the CSR arrays into a single large system $[A_{uu} A_{up}; A_{pu} A_{pp}]$. 
*   **Switching Mechanism:** Use Julia's multiple dispatch to allow `solve_system!` to take a `Vector{ModelEquation}`.
*   **Benefits:** Faster convergence for highly coupled physics (e.g., Poroelasticity, high-Re flow).

## 2. Generic Newton Linearization
**Feasibility: Completed (Enzyme & ForwardDiff supported)**
*   **Current State:** Implemented `NonLinearSi`, `NonLinearRobin`, and generic non-linear support for standard operators (Divergence/Laplacian).
*   **Architecture:** Layered AD strategy (ForwardDiff for local, Enzyme for global).
*   **Next Step:** Full GPU differentiation kernels using Enzyme.

## 3. Higher Order FV Operators
**Feasibility: Completed (Splitting & wide stencil supported)**
*   **Current State:** Implemented `Biharmonic` (4th order) operator. Supports both monolithic mixed-splitting (recommended) and extended-stencil single-equation.
*   **Sparsity:** `extended_sparse_matrix_connectivity` implemented for wide stencils.

## 7. Monolithic Block-Coupled Solvers
**Feasibility: Completed (Prototype implemented)**
*   **Current State:** `MonolithicSystem` and `solve_monolithic!` allow solving multiple fields in a single sparse matrix.
*   **Optimization:** Implement better block-preconditioners (Schur-complement based) for stiff problems like Cahn-Hilliard.

## 4. Gradient Transpose Operator ($\nabla u^T$)
**Feasibility: High**
*   **Problem in OpenFOAM:** Standard OpenFOAM stencils and matrix-free structures make direct manipulation of the full gradient tensor difficult in a single matrix assembly.
*   **XCALibre Advantage:** Since we have the full sparse matrix $A$, we can implement the transpose coupling directly.
*   **Application:** Essential for the full Stress Tensor in complex rheology and for accurate adjoint-based optimization.

## 5. Generic Periodic BCs with Rotations
**Feasibility: Completed (Initial implementation)**
*   I have already refactored `Periodic` to use `transform_point(transform, p)`. This should be further optimized for GPU kernels to avoid CPU-based mapping during every matrix assembly.

## 6. SciML and Neural Network Integration
*   **Lux.jl Integration:** Prefer `Lux.jl` over `Flux.jl` for scientific ML components (surrogates, learned preconditioners) due to its explicit state/parameter separation and Enzyme compatibility.
*   **Differentiable Simulation:** Targeted goal is a fully differentiable PDE framework allowing for sensitivity analysis, topology optimization, and inverse problems.
