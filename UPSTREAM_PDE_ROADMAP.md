# Upstream Alignment and General-PDE Roadmap

Status date: 29 July 2026

This document records how the fork can remain close to upstream XCALibre while
developing its general finite-volume PDE capabilities. It also identifies
candidate upstream pull requests, architectural and numerical risks, a staged
roadmap, and questions to discuss with Humberto.

The current `main` includes upstream v0.6.0 at `859d3a92`. The integration merge
is `810c2eee`. The post-merge CPU test suite passed 926/926 tests. GPU execution
was not available in the integration environment, so GPU correctness still
depends on hardware-backed CI.

## Direction

The most promising identity for this software is:

> A typed, backend-aware finite-volume operator platform whose primary
> production application is CFD.

Upstream should continue to lead the mesh, field-storage, sparse-algebra,
backend, GPU-kernel, and production CFD layers. The fork's main contribution
should be a mathematical operator layer that lowers into those same mechanisms.

The goal is not to replace specialised SIMPLE, PISO, CPISO, Godunov, multiphase,
or thin-film algorithms with one universal nonlinear solver. The goal is to let
the same low-level finite-volume machinery also support ADR, elasticity, Biot,
phase-field, homogenisation, non-Newtonian, and other PDE systems.

The central long-term contract should be:

> Define an operator once and execute it as an assembled equation, explicit
> residual, matrix-free action, Jacobian-vector product, or monolithic block
> contribution.

### Suggested layers

| Layer | Responsibility | Likely ownership |
|---|---|---|
| Mesh, fields, storage and backends | Geometry, connectivity, field types, CPU/GPU storage | Predominantly upstream |
| Local finite-volume kernels | Face fluxes, cell sources, interpolation, gradients | Predominantly upstream |
| Mathematical operator representation | Unbound templates, composition, scaling, nonlinear maps, cross-field binding | Fork, designed for possible upstream adoption |
| Normalisation and lowering | Canonical term order, BC lowering, scalar/vector traits, execution selection | Shared architectural boundary |
| Algorithms | SIMPLE/PISO family, Newton, Picard, JFNK, block solves | Specialised algorithms over common kernels |
| Applications and validation | CFD and general-PDE examples, manufactured solutions, benchmarks | Both |

## Principal risks

### 1. Two overlapping boundary-condition paths

The production boundary path modifies the ordinary finite-volume coefficients.
The experimental boundary-action path expresses residual and Newton actions.
They currently duplicate parts of the BC semantics, and the action path is
CPU-only and allocation-heavy.

Mitigation:

- Represent mathematical BC intent once.
- Lower that representation separately to assembled, explicit-residual,
  Newton-row, and device-kernel implementations.
- Keep the existing production GPU path authoritative until a replacement
  passes parity and performance tests.

### 2. Hidden operator-ordering rules

The legacy discretisation expects a time term to occur first. Operator
composition now preserves that invariant, but the invariant should not remain
implicit in tuple ordering.

Mitigation:

- Add an explicit normalisation pass before binding/discretisation.
- Classify terms by traits such as temporal, face, cell, source, explicit and
  implicit.
- Test equivalent expressions written in different algebraic orders.

### 3. Wrapper types leaking into generated discretisation

`ScaledFlux` required special unwrapping so generated code could determine
whether a source was scalar or vector. Similar wrapper combinations could expose
more hidden assumptions.

Mitigation:

- Define central traits such as `base_flux_type`, `value_rank`,
  `operator_kind`, `bound_field` and `coefficient_type`.
- Normalise nested wrappers before generated kernel selection.
- Test scaling, negation, affine wrapping and nonlinear wrapping together.

### 4. Residual terminology is ambiguous

The code has several useful but distinct residuals:

- the algebraic residual `A*phi - b`;
- an explicit discrete PDE residual;
- a nonlinear residual;
- a normalised linear-solver monitoring residual.

Mitigation:

- Give these separate public names and definitions.
- State whether BC contributions are included.
- State whether assembly or linearisation occurs during evaluation.
- Use the nonlinear/PDE residual, not the Krylov monitor, for Newton
  convergence.

### 5. Mandatory scientific dependencies enlarge the core

Gmsh, ForwardDiff and Enzyme are direct dependencies even though many CFD users
do not need them. This increases installation and compilation burden and may
conflict with upstream's lean GPU philosophy.

Mitigation:

- Move Gmsh conversion into a Julia package extension/weak dependency.
- Make ForwardDiff and Enzyme optional AD backends.
- Retain analytic derivatives as the dependency-free nonlinear contract.

### 6. Matrix-free evaluation lacks a mature preconditioning strategy

Residuals and finite-difference JVPs are useful, but difficult PDEs will not be
competitive without preconditioning.

Mitigation:

- Use matrix-free JVPs for Krylov actions.
- Assemble a cheaper approximate Jacobian for AMG, ILU or block
  preconditioning.
- Treat the matrix-free and assembled forms as complementary execution modes.

### 7. Monolithic sparsity and solvers do not yet scale well

The current block sparsity builder allocates mesh-neighbour connectivity for
every possible variable block, including structurally zero blocks. The default
linear solve has no field-aware block preconditioner.

Mitigation:

- Declare the block dependency graph before allocating sparsity.
- Allocate only active blocks.
- Develop field scaling, null-space treatment, field splits and Schur
  approximations.

### 8. CPU/GPU feature asymmetry

Nonlinear linearisation currently performs CPU loops and scalar AD calls.
Boundary actions also mutate CPU sparse matrices. A general API can therefore
appear backend-generic while failing only when executed on a GPU.

Mitigation:

- State backend support in each API contract.
- Add a minimal hardware-backed GPU smoke suite.
- Require inference, allocation and CPU/GPU parity checks for proposed
  upstream abstractions.

### 9. Examples have mixed maturity

Validated tutorials, diagnostics, research experiments and pending designs are
currently stored near one another. A script starting successfully does not
necessarily establish numerical validity.

Mitigation:

- Classify examples as `validated`, `smoke`, `experimental`, `benchmark` or
  `design-sketch`.
- Record expected norms, convergence rates or reference values for validated
  cases.
- Keep external-data and long-running cases out of the fast test tier.

### 10. The fork delta may become expensive to maintain

The fork adds substantial operator, solver and example functionality. Repeated
edits inside upstream solver files will make every future merge harder.

Mitigation:

- Prefer new modules, extension methods and narrow hooks.
- Maintain a short list of deliberate upstream deviations.
- Merge upstream frequently.
- Review the upstream-relative diff after each release and remove obsolete
  compatibility code.

## Roadmap

### Phase 1: Synchronisation and package hygiene

- Continue regular upstream merges into `main`.
- Avoid broad rewrites of production CFD solvers.
- Classify examples by maturity and expected runtime.
- Move Gmsh and AD packages toward optional extensions.
- Add a small CPU smoke tier and at least one GPU smoke job.
- Document residual definitions and backend limitations.

Exit criteria:

- The working tree remains close enough to upstream that a release merge is a
  reviewable semantic change rather than a rewrite.
- Optional PDE facilities do not impose unnecessary dependencies on core CFD
  users.

### Phase 2: Stable scalar operator contract

- Formalise construction, binding, normalisation and lowering as separate
  stages.
- Stabilise scalar time, divergence, Laplacian, implicit source and explicit
  source operators.
- Make operator scaling, signs and time ordering canonical.
- Define a single mathematical BC interface with multiple lowering targets.
- Verify assembled and explicit residual equivalence.

Exit criteria:

- A scalar ADR operator can be written once and used for assembly, residual
  evaluation and JVP tests.
- Equivalent operator expressions lower to equivalent discrete equations.

### Phase 3: Nonlinear and matrix-free execution

- Standardise nonlinear residual evaluation.
- Add reusable JVP workspaces for allocation-sensitive solver loops.
- Add analytic, ForwardDiff and Enzyme derivative backends behind one contract.
- Add a high-level matrix-free Krylov/JFNK driver.
- Pair matrix-free actions with assembled approximate preconditioners.

Exit criteria:

- JVPs pass directional-derivative comparisons.
- Scalar Newton and JFNK converge to the same manufactured solution.
- CPU and supported GPU backends agree numerically.

### Phase 4: Scalable CPU monolithic nonlinear systems

- Add backtracking or another nonlinear globalization strategy.
- Support multivariate nonlinear maps and their off-diagonal Jacobian blocks.
- Allocate only structurally active block sparsity.
- Add field scaling and block/Schur preconditioning.
- Compare assembled Jacobian actions with finite-difference JVPs.
- Validate nonlinear Robin conditions inside block systems.

Exit criteria:

- A nonlinear two-field manufactured problem shows quadratic Newton
  convergence.
- Vector decomposition produces the same solution as the equivalent manually
  decomposed scalar system.
- Representative saddle-point systems converge with mesh-independent or
  acceptably growing Krylov iteration counts.

### Phase 5: Production multiphysics and GPU strategy

- Add block preconditioners and Schur/field-split approximations.
- Harden elasticity, Biot, phase-field and viscoelastic systems.
- Decide between assembled GPU Jacobians and GPU JFNK with approximate
  preconditioners.
- Add performance and memory regression thresholds.

Exit criteria:

- Representative coupled systems scale beyond demonstration meshes.
- GPU execution is not merely API-compatible but performance-competitive.

## Candidate upstream pull requests

No pull request should combine all of the items below. Small, independently
reviewable contributions are much more likely to match upstream's style.

### Relatively small and independent

1. **Linear Robin boundary condition**

   Contribute only the standard linear finite-volume Robin BC, following the
   current upstream BC dispatch and GPU style. Include a manufactured scalar
   test. Leave nonlinear Robin and boundary actions for later.

2. **Gmsh mesh conversion as an optional extension**

   Provide direct 2D/3D Gmsh conversion without making Gmsh a core dependency.
   Include a very small conversion and boundary-name test.

3. **Patch and domain integral utilities**

   Extract generally useful integration functions with backend-safe reductions
   and conservation tests.

4. **Topological periodic mesh support**

   Propose periodic owner/neighbour connectivity at the mesh level, with a
   small Laplace or Stokes test. Agree on topology representation before
   preparing the PR.

5. **Residual terminology and documentation**

   Separate algebraic, explicit PDE, nonlinear and solver-monitor residuals.
   This can improve the public API without importing the full operator DSL.

6. **Example smoke-test metadata**

   Add a minimal convention that identifies fast validated tutorials, longer
   smoke cases and experimental scripts.

### Foundational PRs requiring prior design agreement

7. **Minimal scalar `PDEOperator` template and field binding**

   Start with a small scalar Laplace/ADR example. Demonstrate lowering into the
   existing upstream discretisation without altering SIMPLE or PISO.

8. **Operator normalisation and scaling**

   Add canonical time ordering, sign handling and coefficient unwrapping only
   after the base operator abstraction is accepted.

9. **Assembled versus explicit residual API**

   Use one scalar equation and standard BCs. Test numerical equality between
   the matrix and direct finite-volume evaluations.

10. **Matrix-free action and finite-difference JVP**

    Present these as execution policies over existing kernels. Include
    assembled-Jacobian-vector equivalence tests and a documented
    preconditioning strategy.

11. **Local nonlinear maps**

    Begin with user-provided analytic derivatives. Add ForwardDiff and Enzyme as
    optional extensions after the mathematical contract is stable.

12. **Monolithic block assembly**

    Discuss this before preparing code. A credible first PR needs an explicit
    block dependency graph, a very small coupled scalar test, correct BC
    lowering and a clear CPU/GPU scope.

13. **Monolithic Newton**

    This should follow, not accompany, basic block assembly. It needs exact
    multivariate Jacobian tests, nonlinear globalization and a field-aware
    inner-solver/preconditioner contract.

## Questions to discuss with Humberto

1. Does he see XCALibre primarily as a CFD solver or as a finite-volume platform
   whose main application is CFD?
2. Should equations own their BCs and solver setup, or should configuration
   remain the single owner?
3. What is the smallest acceptable operator protocol: face contribution, cell
   contribution, source contribution and BC contribution?
4. Should term ordering be an explicit lowering trait rather than tuple
   position?
5. Which residual definitions should be public and stable?
6. Is matrix-free execution desirable when an approximate matrix remains
   available for preconditioning?
7. Which features must be GPU-ready before merge, and which may initially be
   explicitly CPU-only?
8. Would he accept Gmsh, ForwardDiff and Enzyme as optional package extensions?
9. Should experimental block and nonlinear solvers live in the main package, an
   experimental namespace, or a companion package?
10. What inference, allocation, numerical-equivalence and performance gates
    should new abstractions satisfy?
11. How should field scaling, pressure gauges and block preconditioning be
    represented?
12. Would he support a staged upstream roadmap beginning with a linear Robin BC
    and a minimal scalar operator layer?

## Current monolithic Newton status

The CPU assembled Newton path is operational rather than pending. Its public
entry point is `newton_solve!(::MonolithicSystem, config; kwargs...)`; callers
may still provide an explicit scalar-equation BC list when needed.

It currently supports:

- automatic `VectorModel` decomposition into scalar blocks;
- linear and scalar nonlinear self-field operators;
- ordinary cross-field operators routed to off-diagonal blocks;
- correct homogeneous BCs for Newton correction equations;
- fixed damping of the Newton correction;
- pressure/reference constraints in both the nonlinear residual and correction
  system;
- configurable inner tolerance, iteration limit, preconditioner use and
  failure policy;
- analytic, ForwardDiff or Enzyme scalar-map derivatives on CPU;
- reuse of the global residual workspace across Newton iterations.

The validated example is
`examples/operatorBC/monolithic_newton_vector.jl`. It solves the manufactured
nonlinear vector problem

```text
-gamma*laplacian(U) + U.^3 = f
```

with nonzero Dirichlet data. Automatic vector decomposition produces two scalar
blocks, and Newton reaches the constant solution `[2, -1]` in six iterations.

The fast regression test also verifies:

- fixed damping on `u^2 = 4`;
- nonlinear convergence with nonzero Dirichlet data;
- pressure/null-space removal through a Newton reference row.

### Remaining difficulty

The remaining work is not basic Newton plumbing. It is the solver research
needed for large, strongly coupled multiphysics:

- general multivariate maps such as `f(u, p, T)` and their off-diagonal
  Jacobian blocks;
- backtracking, trust-region or another globalization strategy;
- active-block sparsity rather than allocating every possible block;
- field/equation scaling;
- block, Schur or field-split preconditioners;
- assembled-Jacobian versus JVP verification for coupled systems;
- nonlinear Robin conditions in monolithic systems;
- reduced per-iteration matrix and linearisation allocation;
- GPU residual/JVP execution and hardware-backed validation.

For one experienced developer, a stronger CPU research solver with
multivariate local Jacobians, globalization and better validation is a
several-week project. A robust solver for incompressible, Biot, phase-field and
viscoelastic systems is likely several weeks to a few months because
preconditioning and scaling dominate the difficulty. A performant GPU
monolithic Newton implementation remains a multi-stage project; GPU JFNK with
an assembled approximate preconditioner is probably the closest match to
upstream philosophy.
