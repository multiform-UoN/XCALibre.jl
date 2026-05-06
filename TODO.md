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
*   **Backends:** Support for both `:forwarddiff` and `:enzyme` backends in `linearize_physics`.
*   **Architecture (Layered AD Strategy):** 
    *   **Phase 1 (Done):** ForwardDiff for local constitutive laws.
    *   **Phase 2 (Next):** Enzyme reverse-mode for GPU nonlinear residual kernels.
    *   **Phase 3:** Transition to **Matrix-free Newton-Krylov (JFNK)** using Enzyme-generated Jacobian-Vector Products (JVPs). This bypasses expensive sparse matrix assembly on GPUs.
*   **Swappable Backends:** Transition to a unified `AbstractADBackend` interface (inspired by `DifferentiationInterface.jl`) to separate physics kernels from AD implementation.

## 3. Higher Order FV Operators
**Feasibility: Completed (Initial implementation)**
*   **Current State:** Implemented `Biharmonic` (4th order) operator in the DSL.
*   **Sparsity:** Implemented `extended_sparse_matrix_connectivity` which includes second-degree neighbours. This allows XCALibre to handle stencils wider than the standard OpenFOAM 7-point (hex) or immediate-neighbour pattern.
*   **Optimization:** The biharmonic discretization is currently a first-order stencil approximation. Need higher-fidelity interpolation for non-orthogonal meshes.

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
