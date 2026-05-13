# XCALibre.jl Code Audit & Potential Bugs

This document outlines architectural bottlenecks and inconsistencies identified during the codebase audit and the implementation of advanced non-linear/higher-order features.

## 1. Hardcoded ForwardDiff Usage (GPU Incompatibility)
**Severity: Medium**
*   **File:** `src/Solve/Solve_1_api.jl`
*   **Issue:** The generic Newton linearization (`linearize_physics`) relies on `ForwardDiff.derivative` inside a standard CPU `for` loop. This will throw an error or cause a performance hit if executed on a GPU `KernelAbstractions` backend.
*   **Recommendation:** For full GPU compatibility of non-linear sources, transition from `ForwardDiff` to a kernel-compatible autodiff backend like `Enzyme.jl`, or allow the user to provide analytic Jacobians via the DSL.

## 2. Non-Orthogonal Correction Missing in Higher-Order Stencils
**Severity: Low (Mathematical accuracy)**
*   **File:** `src/Discretise/Discretise_1_schemes.jl`
*   **Issue:** The `Biharmonic` operator implementation uses an expanded stencil (neighbors of neighbors) but currently applies a naive, orthogonal-assumed distance scaling. It does not account for the cross-diffusion terms required for highly skewed UNV meshes.
*   **Recommendation:** Implement the full non-orthogonal correction loop for the extended stencil similar to how the `Laplacian` handles `gradf`.
