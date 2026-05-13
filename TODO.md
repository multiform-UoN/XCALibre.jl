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

### Topology-First Periodicity Prototype
- `XCALibre.Mesh.construct_periodic_topology` rewires translational periodic patch pairs into internal-face-like owner/neighbour connections before assembly.
- Straight-channel Stokes and Maxwell periodic examples assemble and solve using the topology path.
- Periodic examples now route through the shared mesh helper instead of maintaining separate sparse-mutation or copied connectivity logic.
- This is the preferred long-term direction for monolithic systems because scalar, pressure, stress, and auxiliary block offsets are handled by the ordinary sparsity builder.
- Still prototype-level for rotations, component transforms, non-orthogonal correction details, output metadata, and MPI/domain decomposition.

### GPU Kernel Prototypes and JVP Infrastructure
- `examples/gpu_kernels/prototype_A_laplacian_residual.jl` — matrix-free scalar Laplacian residual `@kernel`; no sparse matrix; allocation-free inner loop; GPU-portable
- `examples/gpu_kernels/prototype_B_bc_residual.jl` — static BC residual kernel; Dirichlet uses `Atomix.@atomic`; typed-tuple dispatch; `full_residual! = interior + BC` agrees with assembled `A·φ - b` to floating-point precision
- `examples/gpu_kernels/prototype_C_jvp.jl` — three JVP implementations: (1) assembled A·v, (2) linear-kernel JVP (exact for linear ops, zero FD error), (3) FD-JVP `(R(u+εv)-R(u))/ε` (universal, nonlinear-safe, GPU-compatible); all three agree; includes BC JVP kernel `dirichlet_bc_jvp_kernel!`
- `examples/gpu_kernels/prototype_D_bc_lowering.jl` — architectural lowering from `Discretise_7` action vocabulary to static `@inline bc_residual(BC::T, ...)` / `bc_jvp_coeff(BC::T, ...)` functions; verifies same numerical values as `LocalScalarResidualBC`; documents semantic distinction (row-replacement vs scatter-add) and `NonLinearRobin` extension path
- `examples/gpu_kernels/benchmark_mf_vs_assembled.jl` — timing benchmark on 3×3 / 5×5 / 2D mesh: assembled CSR SpMV vs `explicit_residual!` vs Prototype A+B; memory footprint comparison

---

## PENDING

### Consolidation Priorities
- Stabilise the operator-first, monolithic, residual, and topology-periodic APIs before adding more rheology models or external ML/AD integrations.
- Keep examples short and numerical: each benchmark should identify the model, mesh, boundary setup, stabilisation, solver/preconditioner, and reported residual.
- Avoid duplicating mesh rewiring, sparse conversion, field scattering, and pressure-pinning utilities in examples; move repeated mechanics into small explicit helpers.
- Preserve direct sparse solves as verification mode for benchmark-scale systems, especially when Krylov convergence is the object under study.
- Treat bend Maxwell/viscoelastic GMRES limits as conditioning/preconditioning work. The current evidence points to block scaling, near-zero solvent viscosity stiffness, Schur-complement quality, and simple preconditioners rather than corrupt PDE assembly.

### Monolithic Periodic Boundary Conditions Hardening
- **Current Limitation**: The existing `Periodic` BC is scalar-oriented. Its sparse connectivity extension assumes `global row == cell id`, which is not true for monolithic block systems where pressure/stress/auxiliary fields live at block offsets. Algebraic post-assembly correction can be made to work by applying row/column offsets, but it becomes fragile because every operator (`Laplacian`, `ScalarGrad`, `VectorDiv`, upwind transport, Rhie-Chow terms) needs its own periodic insertion semantics.
- **Preferred Long-Term Design**: Periodicity should remain mesh topology, not boundary-condition algebra. The translational prototype now exists in `src/Mesh/Mesh_2_periodic.jl`; the remaining task is to harden and generalise it.
- **Mesh Data Changes Needed**:
  - Add periodic face-pair records that store owner cell, periodic neighbour cell, master/slave face ids, orientation/sign, and the geometric transform from owner to periodic neighbour.
  - Extend `cell_faces`, `cell_neighbours`, `cell_nsign`, and each cell `faces_range` so periodic connections appear in the same loops as internal faces.
  - Preserve patch metadata for user-facing BC assignment and output, but exclude rewired periodic faces from ordinary boundary-condition application.
  - Compute periodic geometry (`delta`, interpolation weights, non-orthogonal correction vectors, transformed centre-to-centre vector) using the periodic transform rather than the physical unshifted coordinates.
- **Sparsity Requirement**: Sparse allocation must see periodic owner/neighbour pairs before matrix construction. Both scalar equation matrices and monolithic block matrices must allocate `(owner, periodic_neighbour)` and reciprocal entries; monolithic offsets then fall out naturally from the existing block sparsity builder.
- **Operators That Become Periodic-Safe Automatically**: Interior-face implementations of `Laplacian`, `Divergence`, `ScalarGrad`, `VectorDiv`, `GradDiv`, source-free cross-field couplings, and monolithic block assembly should work without periodic-specific sparse mutation once periodic faces are traversed as internal connections.
- **Remaining Delicate Areas**:
  - Rhie-Chow / pressure-Laplacian stabilisation must use periodic face distances and transformed pressure gradients consistently.
  - Upwind transport must choose owner/neighbour states using the periodic face flux orientation and transform vector/tensor components where needed.
  - Non-orthogonal corrections must use periodic centre-to-centre vectors, not physical patch separation.
  - Vector/tensor fields may need rotational transforms, not only translational pairing.
  - MPI/domain decomposition must either create periodic ghost cells or include periodic neighbour ownership in halo exchange and sparsity ownership.
  - VTK/OpenFOAM output should retain patch identity even if solver topology treats the faces as internal.
- **Transitional Option**: A post-assembly periodic correction can be used only as a narrow proof-of-concept for scalar operators. It should not become the main monolithic design because it duplicates operator logic and is easy to break with block offsets.

### Residual/Jacobian BC Action Prototype
- `examples/operatorBC/fv_residual_bc_laplacian.jl` connects the BC residual/action idea to a real assembled FV Laplacian rather than a synthetic matrix.
- The reusable action types and `LocalScalarResidualBC` helper now live in `src/Discretise/Discretise_7_boundary_actions.jl`; the examples no longer define their own action backend.
- Useful parts:
  - BC semantics can be expressed as local residuals `B(u)=0`.
  - Nonlinear BC Jacobian rows fit naturally into Newton correction systems.
  - Matrix layout, monolithic offsets, CSR/CSC storage, and future matrix-free backends can be treated as backend concerns.
- Keep experimental:
  - Current action objects and vectors are allocation-heavy and CPU-oriented.
  - Sparse row replacement is a demonstration backend, not a production assembly path.
  - Real integration needs explicit hooks in FV operator/boundary assembly, not a dispatch-heavy layer wrapped around completed matrices.
- GPU/HPC requirement before promotion: lower BC actions into static, allocation-free residual/Jacobian kernels with predictable memory access and no runtime dispatch in inner loops.

### Direct Solver Diagnostics (Completed)
- Diagnostics run using Julia's sparse `\` (UMFPACK) on the unstructured 9440-cell L-bend mesh confirmed the PDE assembly is correct, and isolated the slow Krylov convergence in Maxwell to an iterative conditioning problem.
  - *Stokes-like ($\mu_s=1, \mu_p=0$)*: Direct solve 1.18s, max|u| 0.124.
  - *Oldroyd-B-like ($\mu_s=1, \mu_p=1$)*: Direct solve 0.20s, max|u| 0.064.
  - *Maxwell ($\mu_s=10^{-6}, \mu_p=1$)*: Direct solve 5.88s, max|u| 0.148.
- **Conclusion**: The monolithic block-coupled equations are physically and algebraically valid. The stiffness introduced by the near-zero solvent viscosity in Maxwell cases makes simple GMRES+Jacobi struggle. Direct solvers provide an excellent verification mode for benchmark-scale domains without requiring advanced preconditioning research.

### GPU Readiness Audit

#### Currently GPU-ready (runs as `@kernel` today)
- `apply_boundary_conditions!` — `Discretise_5_apply_bcs.jl`; typed BC tuple, no dynamic dispatch
- `correct_boundaries!` — `Discretise_3_boundary_conditions.jl`
- All FV assembly kernels — `Discretise_2_generated_distretisation.jl` (`@generated` + `@kernel`)
- `explicit_residual!` — matrix-free, allocation-free, type-stable
- Prototype A/B kernels — `examples/gpu_kernels/`; swap `backend = CUDABackend()` to run on GPU

#### GPU-compatible architecture (CPU today; GPU port is straightforward)
- `construct_periodic_topology` — CPU preprocessing; once topology is rewired, periodic faces appear as interior connections and all cell-loop `@kernel`s are automatically periodic-safe, no kernel changes needed
- `MonolithicSystem` sparsity construction — CPU preprocessing; the resulting CSR matrix can be adapted to GPU with `adapt(backend, A)`
- Assembled CSR path — `SparseMatricesCSR` on GPU is supported by `CUDA.jl` via `CUSPARSE`

#### CPU-only (explicit design choice or technical blocker)
- `linearize_physics` — ForwardDiff on CPU arrays; `_assert_cpu_linearization` throws if backend ≠ CPU
- `Discretise_7_boundary_actions.jl` — `SparseMatrixCSC` row surgery, `Vector{AbstractBoundaryAction}` dynamic dispatch; documented CPU-only
- `monolithic_discretise!` — sequential CPU loop (equations × cells × faces); no `@kernel`
- `monolithic_apply_bcs!` — sequential CPU loop over boundary faces
- `newton_solve!` (all variants) — depends on `linearize_physics`

#### GPU-dangerous abstractions (need redesign before GPU port)
- `Vector{<:AbstractBoundaryAction}` in `apply_boundary_actions!` — runtime polymorphism kills GPU dispatch
- `SparseMatrixCSC` mutation anywhere — column-oriented, GPU-hostile; GPU path must use CSR
- `objectid(phi.values)` in `field_to_idx` — pointer-based identity; brittle if CPU/GPU arrays are different objects (currently only used at CPU assembly time, not in kernels, so not an immediate hazard)

#### Staged GPU roadmap
1. **(Ready now)** Matrix-free residual `R(u)`: `explicit_residual!` / Prototype A+B.
2. **(Ready now)** FD-JVP `J(u)·v`: Prototype C `fd_jvp!` — two `full_residual!` calls; no matrix; nonlinear-safe; GPU-compatible by swapping backend.
3. **(Near)** JFNK solver: wrap `fd_jvp!` in a matrix-free Krylov loop (GMRES/BiCGSTAB via LinearAlgebra or IterativeSolvers); outer Newton uses `full_residual!` for convergence check.
4. **(Near)** Diagonal preconditioner: kernel that accumulates `Σ D_f` per cell — avoids `linearize_physics` entirely for Jacobi preconditioning.
5. **(Medium)** BC JVP for `NonLinearRobin`: scalar FD or `Enzyme.autodiff` per face; allocation-free, device-safe (Prototype D documents the extension path).
6. **(Long)** Assembled GPU Jacobian: Enzyme device-side AD through `@kernel` functions, or CPU-compute J then `adapt(backend, J)`. Required only for ILU/block preconditioners. `linearize_physics` stays CPU-only until this step.

### Non-Orthogonal Correction for Biharmonic
- Current Biharmonic scheme is orthogonal-mesh only
- Non-orthogonal correction requires the cross-diffusion term at ∇² level before applying Δ²

### Poroelasticity Example
- `examples/linearElastic/biot_consolidation_1d.jl`: Terzaghi 1-D consolidation with fixed-stress split ✓
- Elastic block: `MonolithicSystem([u_eqn, v_eqn])` + GradDiv (same as linear_elastic_2d)
- Flow block: `Time{Euler}(Sε) - Laplacian{Linear}(k) == Source(div_u_src)` via PDEOperator DSL
- Live-reference sources: `Source(p_grad_x)` / `Source(div_u_src)` update without equation rebuild
- Monolithic 3-field Biot complete: `examples/linearElastic/biot_consolidation_monolithic.jl` — uses `ScalarGrad{T,I}` and `VectorDiv{T,J}` coupling operators

### SciML / Lux Integration
- Lux.jl preferred over Flux.jl for Enzyme compatibility
- Targeted use: learned preconditioners, neural constitutive models, surrogate BCs
- Fully differentiable simulation pipeline (sensitivity analysis, topology optimisation)
- Secondary until core FV residual/Jacobian and monolithic solver APIs are stable.

### JFNK (Jacobian-Free Newton-Krylov)
- `jvp!(y, L, u, v, config)` — Jacobian-vector product via `explicit_residual!` + finite difference
- Enables Newton with no matrix assembly; purely matrix-free inner Krylov iterations
- Foundation (`explicit_residual!`) is in place; only the JVP wrapper remains
- Keep this behind explicit residual verification; matrix-free correctness should be tested against assembled residuals before adding more nonlinear physics.

### Upstream PR Candidates
- `Robin` + `NonLinearRobin` BCs (ready, no API dependency)
- Rotational periodic BC (algebraic path exists, but topology-first implications should be reviewed before upstreaming)
- `Biharmonic` operator (wait for non-orthogonal correction)
- `OperatorTemplate` / `PDEOperator` abstraction (wait for Humberto's API review)
