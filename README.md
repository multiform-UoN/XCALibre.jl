<meta name="google-site-verification" content="UZSnZbbvZqRUM_1_d5d9ox1IeO5z9iE8Oynt7mBjJaM" />

[![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] [![][CI-img]][CI-url] [![][JOSS-img]][JOSS-url]

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://mberto79.github.io/XCALibre.jl/stable/

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://mberto79.github.io/XCALibre.jl/dev/

[CI-img]: https://github.com/mberto79/XCALibre.jl/actions/workflows/CI.yml/badge.svg
[CI-url]: https://github.com/mberto79/XCALibre.jl/actions/workflows/CI.yml

[JOSS-img]: https://joss.theoj.org/papers/10.21105/joss.07441/status.svg
[JOSS-url]: https://doi.org/10.21105/joss.07441


# XCALibre.jl

*XPU CFD Algorithms and libraries*

## What is XCALibre.jl?


XCALibre.jl (pronounced as the mythical sword *Excalibur*) is a general purpose Computational Fluid Dynamics (CFD) library for 2D and 3D simulations on structured/unstructured grids using the finite volume method. XCALibre.jl has been designed to act as a platform for developing, testing and using *XPU CFD Algorithms and Libraries* to give researchers in both academia and industry alike a tool that can be used to test out ideas easily within a framework that offers acceptable performance. To this end, XCALibre.jl has been implemented to offer both CPU multi-threaded capabilities or GPU acceleration using the same codebase (thanks to the unified programming framework provided by [KernelAbstractions.jl](https://juliagpu.github.io/KernelAbstractions.jl/stable/)). XCALibre.jl also offers a friendly API for those users who are interested in running CFD simulations with the existing solvers and models built into XCALibre.jl.

#### Large Eddy Simulation
![](docs/src/figures/animated_cylinder_re1000-2x.gif)

#### Reynolds-Averaged Navier-Stokes Simulation
![](docs/src/figures/F1-RANS.png)
(mesh file downloaded from [FetchCFD](https://fetchcfd.com/view-project/136-f1-mesh-for-simulation#))

## Installation


First, you need to [download and install Julia on your system](https://julialang.org/downloads/). Once you have a working installation of Julia, XCALibre.jl can be installed using the built-in package manager.

XCALibre.jl is available directly from the the General Julia Registry. Thus, to install XCALibre.jl open a Julia REPL, press `]` to enter the package manager. The REPL prompt icon will change from **julia>** (green) to **pkg>** (and change colour to blue) or **(myenvironment) pkg>** where `myenvironment` is the name of the currently active Julia environment. Once you have activated the package manager mode enter

```julia
pkg> add XCALibre
```

To install XCALibre.jl directly from Github enter the following command (for the latest release)

```julia
pkg> add XCALibre https://github.com/mberto79/XCALibre.jl.git
```

A specific branch can be installed by providing the branch name precided by a `#`, for example, to install the `dev-0.3-main` branch enter

```julia
pkg> add XCALibre https://github.com/mberto79/XCALibre.jl.git#dev-0.3-main
```

## Main features


* Multithreaded or GPU execution with support for multiple GPU backends  (NVidia, AMD and Intel) - as supported by [KernelAbstractions.jl](https://juliagpu.github.io/KernelAbstractions.jl/stable/) (except Apple hardware)
* Ability to import *.unv* and OpenFOAM grids. Simulation results written in `VTK` or `OpenFOAM` file formats, allowing postprocessing in [ParaView](https://www.paraview.org/)
* Incompressible and (weakly) compressible flow solvers
* Viscoelastic and Non-Newtonian flows (Maxwell, Kelvin-Voigt, Oldroyd-B prototypes) via monolithic block-coupled $(u, p, \tau)$ formulation
* RANS and LES turbulence modelling (`KOmega` and `KOmegaLKE` for RANS and `Smagorinsky` for LES, for now!)
* Energy modelling using Sensible Energy model
* Classic boundary conditions, including Dirichlet, Neumann, Wall, Symmetry, etc.
* User-defined boundary conditions as neural networks or user-defined functions (source/sink terms soon)
* Easy to link with Julia ecosystem - making it easy to embed custom machine learning models, perform optimisation runs, etc. (see examples in the [documentation](https://mberto79.github.io/XCALibre.jl/stable/))
* A good selection of discretisation schemes available e.g. Euler, Upwind, LUST, etc.
* Simple API for defining new transport equations or solvers

Code example

```julia
L_U = ((
          Time{schemes.U.time}()
        + Divergence{schemes.U.divergence}(mdotf)
        - Laplacian{schemes.U.laplacian}(nueff)
        ==
        Source(-∇p.result)
      ) → BCs.U) → solvers.U

U_eqn = L_U(U)
```

## Main dependencies


XCALibre.jl relies on the functionality provided by other packages from the Julia ecosystem. For a full list of direct dependencies please refer to the Project.toml file included with this repository. We are thankful to the teams that have helped develop and maintain every single of our dependencies. Major functionally is provided by the following:

* KernelAbstractions.jl - provides a unified parallel programming framework for CPUs and GPUs
* Krylov.jl - provides solvers for linear systems at the heart of XCALibre.jl
* LinearOperators.jl - wrappers for matrices and linear operators
* Atomix.jl - enables atomix operations to ensure race conditions are avoided in parallel kernels
* CUDA.jl, AMD.jl, Metal.jl and OneAPI.jl - not direct dependencies but packages enabling GPU usage in Julia
* StaticArrays.jl - provides definitions and performant primitives for working with vectors and matrices

## Related projects


There are other wonderful fluid simulation packages available in the Julia ecosystem (please let us know if we missed any):

* [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl)
* [Waterlilly.jl](https://github.com/WaterLily-jl/WaterLily.jl)
* [Trixi.jl](https://github.com/trixi-framework/Trixi.jl)

## How to Cite

If you have used XCALibre.jl in your work, please cite it using the reference below:

```
@article{Medina2025,
  author = {Humberto Medina and Christopher D. Ellis and Tom Mazin and Oscar Osborn and Timothy Ward and Stephen Ambrose and Svetlana Aleksandrova and Benjamin Rothwell and Carol Eastwick},
  title = {XCALibre.jl: A Julia XPU unstructured finite volume Computational Fluid Dynamics library},
  journal = {Journal of Open Source Software},
  publisher = {The Open Journal},
  volume = {10},
  number = {107},
  pages = {7441},
  year = {2025},
  doi = {10.21105/joss.07441},
  url = {https://doi.org/10.21105/joss.07441}
}
```


## multiform-UoN Fork New Features

This fork extends XCALibre.jl with capabilities for nonlinear physics, coupled multi-field solvers, and a higher-level PDE abstraction layer. The emphasis is on keeping finite-volume operators explicit and inspectable while reducing bookkeeping in multi-physics examples.

### Operator-First PDE DSL

Equations are defined independently of fields, then bound at solve time — similar to [Chebfun](https://www.chebfun.org/):

```julia
# Define the PDE once
L = (
      Divergence{Upwind}(mdotf)
    - Laplacian{Linear}(D)
    + NonLinearSi(NonlinearMap(c -> k*c^2))
    == Source(0.0)
) → BCs.C → solvers.C

# Bind to any field and solve
C_eqn = L(C)
solve_equation!(C_eqn, config)        # linear solve
newton_solve!(L, C, config; tol=1e-8) # Newton solve — same operator
```

Boundary conditions and solver settings travel with the operator. No manual `@reset` bookkeeping.

### Newton Linearisation with Automatic Differentiation

Nonlinear operators and sources are linearised automatically each Newton iteration using ForwardDiff (or a user-supplied analytic derivative):

```julia
# Nonlinear reaction k·C² — no manual Jacobian needed
L = -Laplacian{Linear}(D) + NonLinearSi(NonlinearMap(c -> k*c^2)) == Source(f)
result = newton_solve!(L → BCs → solvers, C, config; tol=1e-8, maxiter=20, verbose=true)
```

`homogeneous(L)` automatically zeros Dirichlet values for the correction step, giving a correct Newton BVP for δC without any user intervention.

### Monolithic Block-Coupled Solvers

Multiple coupled scalar equations are assembled into a single block-sparse system and solved simultaneously:

```julia
sys = MonolithicSystem([u_eqn, v_eqn], [u, v])
solve_monolithic!(sys, (BCs.u, BCs.v), config)
```

Used for linear elasticity, Cahn-Hilliard phase field, multi-species transport, and the current incompressible/weakly-compressible viscoelastic prototypes. The practical viscoelastic path uses the conventional $(u,p,\tau)$ split, with pressure retained and $\tau$ treated as extra stress. Rhie-Chow-style pressure stabilisation is available for collocated Stokes/viscoelastic systems; difficult bend Maxwell cases are currently understood as conditioning/preconditioning problems rather than assembly failures.

### Topology-First Periodicity

Periodic patch pairs can be rewired into internal-face-like owner/neighbour connections before assembly:

```julia
mesh_periodic = XCALibre.Mesh.construct_periodic_topology(
    mesh, :inlet, :outlet, [Lx, 0.0, 0.0]
)
```

This avoids post-assembly sparse-matrix surgery and naturally respects monolithic block offsets. Once periodic faces appear in the same connectivity loops as ordinary internal faces, operators such as `Laplacian`, `Divergence`, `ScalarGrad`, `VectorDiv`, and monolithic sparsity allocation do not need separate scalar-specific periodic code. The current implementation is a translational prototype; rotational transforms, vector/tensor component transforms, non-orthogonal corrections, output metadata, and MPI/domain-decomposition support still need careful review.

### GradDiv Operator for Elasticity

The `GradDiv{T,I,J}` operator assembles the full Cauchy-stress stiffness `(μ+λ)∂(∇·U)/∂xᵢ` block-coupled with the standard Laplacian — no gradient-transpose postprocessing required:

```julia
u_eqn = (
    - Laplacian{Linear}(mu_flux, u)
    - GradDiv{Linear,1,1}(alpha_flux, u)
    - GradDiv{Linear,1,2}(alpha_flux, v)
    == Source(0.0)
) → ScalarEquation(u, BCs.u)
```

### Higher-Order Operators

`Biharmonic{T}` implements the 4th-order operator Δ²ϕ natively in the FVM DSL, enabling Cahn-Hilliard and thin-film phase-field models without operator splitting.

### Robin and Nonlinear Robin Boundary Conditions

Generalised `a·ϕ + b·∂ₙϕ = c` conditions, including a nonlinear variant linearised automatically each outer iteration.

### Split Assembly and Residual API

```julia
assemble_matrix!(eqn, config)          # build A once for parametric/linear problems
assemble_rhs!(eqn, new_source, config) # swap RHS without rebuilding A
r = residual(L, phi, config)           # mathematical residual vector Au - b
explicit_residual!(r, eqn, phi, config) # matrix-free residual kernel (JFNK foundation)
```

### Residual/Jacobian Boundary Actions (Experimental)

Boundary conditions are being explored as residual contributions,

```math
\mathcal{B}(u) = 0,
```

that provide explicit Jacobian/RHS actions during Newton assembly. This is useful for nonlinear Robin-type conditions and for separating BC semantics from matrix storage layout. The current examples remain prototypes under `examples/operatorBC/`: `fv_residual_bc_laplacian.jl` applies a residual/Jacobian boundary row to a real FV Laplacian, while `boundary_action_sketch.jl` and `residual_bc_sketch.jl` show the smaller action vocabulary. This direction should not be treated as a production GPU implementation yet; dynamic action lists and sparse row mutation would need to be lowered to static, allocation-free kernels for performance-sensitive paths.

### HPC and GPU Position

The fork aims to preserve GPU-compatible architecture: explicit operators, topology-level connectivity, backend-neutral residual concepts, and matrix-free residual hooks. That is different from claiming that all new abstractions are already GPU efficient. Current nonlinear linearisation and exploratory BC-action examples are CPU-oriented; device kernels, allocation behaviour, and dispatch boundaries must be audited before promoting these paths to production GPU workflows.

### Extended Post-Processing and Homogenisation

Volume averaging, permeability tensor computation (2D/3D), and dispersivity optimisation for porous media upscaling. See `examples/homogenisation/`.

### New Examples

| Example | Feature demonstrated |
|---|---|
| `ADR/adr_scalar.jl` | PDEOperator DSL, Robin BC, full SIMPLE + scalar transport |
| `ADR/nonlinear_adr.jl` | Operator-first ADR with Newton-linearised nonlinear Robin BC |
| `ADR/nonlinear_source_adr.jl` | Operator-first ADR with Newton-linearised implicit nonlinear source |
| `ADR/monolithic_quad_laplacian.jl` | Block-coupled monolithic solve |
| `linearElastic/linear_elastic_2d.jl` | 2-field monolithic elasticity via GradDiv |
| `phaseField/cahn_hilliard_monolithic.jl` | Biharmonic + monolithic Cahn-Hilliard |
| `thinFilm/thin_film_multiform.jl` | Coupled thin-film (viscous + Darcy) |
| `homogenisation/permeability_tensor_2d.jl` | Upscaled permeability from pore-scale DNS |
| `linearElastic/biot_consolidation_1d.jl` | Biot consolidation: elastic MonolithicSystem + transient pressure PDEOperator |
| `nonNewtonian/benchmarks/straight/stokes_periodic.jl` | Topology-first periodic Stokes assembly |
| `operatorBC/fv_residual_bc_laplacian.jl` | Experimental residual/Jacobian BC action on a real FV operator |
| `operatorBC/topological_periodic_stokes.jl` | Periodicity as mesh topology, not sparse matrix surgery |

See `examples/` for full usage and `TODO.md` for the development roadmap.
