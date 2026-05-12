# XCALibre.jl Viscoelastic and Non-Newtonian Framework

This directory contains the authoritative set of scripts demonstrating the monolithic block-coupled viscoelastic solvers in XCALibre.jl. 

## Architectural Philosophy

The implementation uses a **Practical Branch $(u, p, \tau)$** formulation. Unlike traditional mixed FEM $(u, \sigma)$ total-stress formulations, separating the pressure from the extra-stress tensor allows us to leverage Rhie-Chow stabilisation. This prevents the checkerboard nullspaces and volumetric locking often seen in collocated finite volume methods. 

## Numerical Benchmarks and Examples

The scripts are divided into three categories:

### 1. Baselines & Core Architecture
* **`stokes_incomp_channel.jl`**: Standard 2D Channel Stokes benchmark. Use this as a reference for validating pure Newtonian implementations against OpenFOAM.
* **`stokes_incomp_bend.jl`**: Validates the mixed-form Stokes equations on a more complex imported OpenFOAM mesh (L-Bend).
* **`stokes_core_diagnostics.jl`**: Low-level script used to diagnose matrix nullspaces and penalty pinning.
* **`branch_practical_up.jl`**: The standard $(u, p, \tau)$ template showing how to decouple pressure and extra-stress.
* **`branch_experimental_usigma.jl`**: A research-level $(u, \sigma)$ template showing the total-stress formulation (provided for comparison).

### 2. Viscoelastic Constitutive Models
* **`maxwell_incomp_channel.jl`**: Standard $(u, p, \tau)$ Maxwell solver. Demonstrates fully implicit constitutive coupling with stability independent of relaxation time $\lambda$.
* **`oldroyd_incomp_channel.jl`**: Upper-Convected Oldroyd-B solver. Demonstrates the upwinded advective transport of the tensor field ($u \cdot \nabla \tau$), critical for High Weissenberg Number problem studies.
* **`kv_incomp_channel.jl`**: Kelvin-Voigt solver demonstrating history-dependent discrete strain updates over multiple time steps.

### 3. Objective Rates
* **`maxwell_objective_rates_channel.jl`**: Direct comparison of Corotational (Jaumann) vs. Upper-Convected (Stretching-Only) objective stress rates in a rotationally-dominant channel. 

## Numerical Study Findings
Based on the `run_study.jl` parameter sweep:
1. **Mesh & Timestep Dependence**: The $(u,p,\tau)$ branch is stable on standard quad grids up to very high Weissenberg numbers. 
2. **Rhie-Chow Interactions**: There are no negative interactions between the pressure regularisation (Rhie-Chow) and the viscoelastic extra stresses.
3. **Advection**: The primary source of instability is the $u \cdot \nabla \tau$ term. Upwind differencing successfully controls convective instabilities at $\lambda = 10.0$ and above.
