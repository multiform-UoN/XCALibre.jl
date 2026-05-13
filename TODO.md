# TODO: Roadmap for Advanced Upscaling & Coupled Solvers in XCALibre.jl

## COMPLETED

### Monolithic Block-Coupled Solvers
- `MonolithicSystem` assembles multiple scalar equations into a single block-sparse system
- Term-to-column routing via `field_to_idx` (objectid-based, no special coupling operator needed)
- `solve_monolithic!` solves the assembled block system
- Examples: `monolithic_quad_laplacian.jl`, `linear_elastic_2d.jl`, `cahn_hilliard_monolithic.jl`

### Newton Linearisation (ForwardDiff + Enzyme)
- `NonLinearSi(func)` — nonlinear implicit source, linearised per cell
- `NonlinearOperator` — wraps any differential operator (Laplacian, Divergence) with a nonlinear map
- `NonlinearOperatorTemplate` — field-free nonlinear template for PDEOperator DSL
- `NonLinearRobin` — nonlinear boundary condition, linearised at each outer iteration
- `linearize_physics(BCs, eqn)` — full Newton pre-pass via ForwardDiff (Enzyme optional)
- `newton_solve!(L, phi, config)` — self-contained Newton loop with convergence history
- `homogeneous(L)` — zeros all Dirichlet values for correction-step BVPs
- Examples: `nonlinear_adr.jl`, `nonlinear_ops_adr.jl`, `nonlinear_source_adr.jl`

### Operator-First PDE Abstraction (Chebfun-style)
- `OperatorTemplate{F,S,T}` — operator without bound field; all existing operators return templates when called with flux only
- `PDEOperator` — container of templates + sources + BCs + SolverSetup
- DSL: `L = -Laplacian{Linear}(D) + Si(k) == Source(f)` → `L → BCs → solvers.C` → `eqn = L(phi)`
- `→ BoundaryCollection` attaches BCs; `→ SolverSetup` attaches solver config; fallback errors clearly
- `Source(x::Number)` auto-wraps to `ConstantScalar`
- `solve_equation!(eqn, config)` — self-contained solve from stored BCs and setup
- `ModelEquation` now stores BCs (`eqn.equation.BCs`) and SolverSetup (`eqn.setup`)

### Residual and Split-Assembly API (Phases 3–5)
- `residual(eqn, config)` / `residual!(r, eqn, config)` — mathematical residual vector `Au - b`
- `residual(L, phi, config)` — operator-first residual evaluation
- `residual_norm(r)` / `residual_norm(eqn, config)` — norm of mathematical residual
- `solve_residual(eqn, component, config)` — solver monitor norm (renamed from old `residual`)
- `assemble_matrix!(eqn, config)` — assemble stiffness matrix only (Phase 4)
- `assemble_rhs!(eqn, source, config)` — assemble RHS only, swap sources without rebuilding A (Phase 4)
- `explicit_residual!(r, eqn, phi, config)` — matrix-free residual kernel (Phase 5 foundation)

### Higher-Order Operators
- `Biharmonic{T}` — 4th-order operator (Δ²ϕ) with extended stencil support
- `extended_sparse_matrix_connectivity` — 2nd-degree neighbour stencil for wide-stencil operators
- Examples: `biharmonic_operator.jl`, `cahn_hilliard_scalar.jl`

### GradDiv Operator (replaces gradient transpose)
- `GradDiv{T,I,J}(flux)` — implicit two-point FVM operator for the (I,J) block of `(μ+λ)∇(∇·U)`
- Assembles full Cauchy-stress stiffness block-coupled with `Laplacian{Linear}(mu, U_i)`
- Replaces the need for a separate gradient-transpose operator: the off-diagonal blocks `GradDiv{T,i,j}` cover all coupling terms exactly
- Examples: `linear_elastic_1field.jl`, `linear_elastic_2d.jl`

### Robin and NonLinear Robin Boundary Conditions
- `Robin(:patch, a=..., b=..., value=...)` — generalised `a·ϕ + b·∂ₙϕ = c`
- `NonLinearRobin` — nonlinear Robin, linearised each outer iteration
- Tests: `unit_test_robin.jl`

### Periodic BC with Rotational Transform
- `RotationalTransform` + `transform_point` for rotationally periodic meshes

### Extended Post-Processing and Homogenisation
- Volume averaging, permeability tensor (2D/3D), dispersivity optimisation
- Examples: `homogenisationFoam.jl`, `permeability_tensor_2d/3d.jl`, `optimise_dispersivity.jl`

### Additional Examples
- Thin-film: viscous (`thin_film_shallow_water.jl`), Darcy (`thin_film_darcy.jl`), coupled (`thin_film_multiform.jl`)
- Phase field: Cahn-Hilliard scalar and monolithic, biharmonic demo
- Linear elastic: 1-field bar stretch, 2-field monolithic uniaxial

### Non-Newtonian & Viscoelastic Flow (Phase 2 & 3)
- Fully functional `(u, p, τ)` Practical branch using decoupled pressure and extra-stress.
- Monolithic field update infrastructure (`set_fields!`, `update_fields!`) and block-coupled variable scattering resolved.
- Explicit AST operator construction adopted for robust `PDEOperator` scaling and composition.
- Missing boundary conditions (`Dirichlet`, `Zerogradient`) for `ScalarGrad` and `VectorDiv` coupling operators implemented.
- Viscoelastic models implemented:
  - **Maxwell (Linear & Corotational/Jaumann)**
  - **Kelvin-Voigt**
  - **Oldroyd-B (Upper-Convected)**
- Advective transport (`Divergence{Upwind}`) for tensor fields integrated and stabilised.
- Validation hierarchy established (Stokes baseline $\rightarrow$ Maxwell $\rightarrow$ Oldroyd-B).
- **Finding**: Rhie-Chow pressure stabilisation does not conflict with viscoelastic extra-stresses; checkerboard modes are successfully suppressed.
- **Benchmark Suite**: A systematic `Stokes3x3` matrix created in `examples/nonNewtonian/benchmarks` exploring straight vs L-bend geometries and Neumann vs Pressure-driven BCs.
- **Compressible Foundations**: Weakly compressible Stokes and Maxwell formulations verified. Rhie-Chow stabilization gracefully accommodates the $\beta p + \nabla \cdot u = 0$ continuity modifications.

---

## PENDING

### Monolithic Periodic Boundary Conditions Redesign
- **Current Limitation**: The existing `Periodic` BC directly mutates sparse matrix indices using `Atomix.@atomic nzval[spindex(...)]`. This works for scalar PDEs but corrupts the block-coupled matrix in a `MonolithicSystem` because it is unaware of the `row_offset` and `col_offset` block shifts.
- **Required Design**:
  - `(bc::Periodic)` evaluations must return *nonlocal matrix insertions*, rather than just local `(AP, BP)` face scalars.
  - `monolithic_apply_bcs!` must intercept these insertions and apply the `row_off` and `col_off` shifts before injecting into the global `A_mono`.
  - Operator-specific logic must be defined (e.g., how the pressure gradient links across the periodic boundary for `ScalarGrad` and `VectorDiv`).
- **Implementation Path**: Do NOT rewrite the entire BC API globally yet. Introduce a specialized `apply_periodic_bcs_monolithic!` path as a transitional proof-of-concept before refactoring the core scalar API.

### GPU Newton / Enzyme Device Path
- Current `linearize_physics` runs a scalar CPU loop over cell values
- Enzyme device-side AD (kernel-level) needed for GPU Newton
- Requires Humberto's input on kernel AD API before implementing

### Non-Orthogonal Correction for Biharmonic
- Current Biharmonic scheme is orthogonal-mesh only
- Non-orthogonal correction requires the cross-diffusion term at ∇² level before applying Δ²

### Poroelasticity Example
- `examples/poroelastic/biot_consolidation_1d.jl`: Terzaghi 1-D consolidation with fixed-stress split ✓
- Elastic block: `MonolithicSystem([u_eqn, v_eqn])` + GradDiv (same as linear_elastic_2d)
- Flow block: `Time{Euler}(Sε) - Laplacian{Linear}(k) == Source(div_u_src)` via PDEOperator DSL
- Live-reference sources: `Source(p_grad_x)` / `Source(div_u_src)` update without equation rebuild
- Monolithic 3-field Biot complete: `biot_consolidation_monolithic.jl` — uses `ScalarGrad{T,I}` and `VectorDiv{T,J}` coupling operators

### SciML / Lux Integration
- Lux.jl preferred over Flux.jl for Enzyme compatibility
- Targeted use: learned preconditioners, neural constitutive models, surrogate BCs
- Fully differentiable simulation pipeline (sensitivity analysis, topology optimisation)

### JFNK (Jacobian-Free Newton-Krylov)
- `jvp!(y, L, u, v, config)` — Jacobian-vector product via `explicit_residual!` + finite difference
- Enables Newton with no matrix assembly; purely matrix-free inner Krylov iterations
- Foundation (`explicit_residual!`) is in place; only the JVP wrapper remains

### Upstream PR Candidates
- `Robin` + `NonLinearRobin` BCs (ready, no API dependency)
- Rotational periodic BC (ready, surgical change)
- `Biharmonic` operator (wait for non-orthogonal correction)
- `OperatorTemplate` / `PDEOperator` abstraction (wait for Humberto's API review)
