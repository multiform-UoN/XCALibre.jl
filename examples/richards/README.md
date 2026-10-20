# Richards' equation and solute transport with Newton's method

Variably saturated flow on the PDE layer, with analytic derivatives for every
nonlinear term. The flow is written so that each term is one the DSL already
linearises:

    θ(h)/Δt - ∇·(Ks ∇Φ(h)) + ∇·(F_g kr(h)) = θⁿ/Δt,    Φ' = kr,  F_g = Ks ĝ·S_f

- storage `NonLinearSi(θ/Δt, C/Δt)`, old content as a `Source` (mixed form,
  mass conservative);
- diffusion as a `Laplacian` of the Kirchhoff potential `Φ` (the face flux is
  `Ks (Φ_N - Φ_P)/δ`, i.e. `Ks` times the mean of `kr` over `[h_P, h_N]`);
- gravity as an upwind `Divergence` of `kr(h)`.

`linearize_physics` then assembles the exact Jacobian; `newton_step!`
(`richards_common.jl`) adds a backtracking line search and reports failure so
that the caller halves the step.

| file | what |
|---|---|
| `vanGenuchten.jl` | van Genuchten–Mualem θ, C, kr, dkr/dh (analytic), Φ by composite Gauss–Legendre in ln(-h) |
| `richards_common.jl` | operator, Newton step with line search, discrete Darcy flux, cell speed |
| `celia_infiltration_1D.jl` | Celia et al. (1990) infiltration + tracer; benchmark shared with GMRTFoam |
| `langmuir_transport_1D.jl` | solute with Langmuir sorption (Newton on transport), checked against the Rankine–Hugoniot shock speed |
| `column1D/polyMesh` | the 100-cell column of the GMRTFoam case (`blockMesh`) |

## Shared benchmark with GMRTFoam

`celia_infiltration_1D.jl` solves the case of
`GMRTFoam/tutorials/richardsHeadFoam/celiaInfiltration1D` (same mesh, same
face conductivity, same upwind gravity flux, fixed Δt = 100 s) and writes
`celia_xcalibre_<t>.csv`; that case's `compare.py` reads them. Agreement at
equal Δt: max |Δθ| ≈ 1e-6 at Δt = 100 s, 2e-11 at Δt = 10 s; both codes close
the water balance to round-off.

Fixed-Δt sweep, 1 day (`RICHARDS_FIXED_DT=<s>`; error in θ against Δt = 10 s):

| Δt [s] | Newton steps | Newton its | rejected | err θ | Picard substeps | Picard its | rejected | err θ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 8640 | 26543 | 0 | ref. | 8640 | 36269 | 0 | 1.8e-11 |
| 100 | 865 | 3550 | 1 | 2.4e-4 | 870 | 7059 | 3 | 2.4e-4 |
| 300 | 291 | 1386 | 2 | 7.8e-4 | 425 | 5020 | 22 | 7.9e-4 |
| 900 | 111 | 647 | 4 | 2.5e-3 | 370 | 5407 | 64 | 2.5e-3 |
| 3600 | 80 | 483 | 7 | 8.3e-3 | 819 | 7850 | 65 | 1.2e-3 |

Newton converges quadratically (max|Δh|: 0.16, 4.7e-3, 3.9e-6, 2.9e-12 m) and takes
large steps in dry soil with a few rejections; modified Picard needs 2–16× more
iterations and splits large steps into many substeps. Per iteration
richardsHeadFoam is 15–30× faster on this 100-cell column (compiled C++,
tridiagonal solve, no Julia overhead), so in wall time the 1D case favours
GMRTFoam.

## Langmuir sorption

`langmuir_transport_1D.jl`: steady unit-gradient column (h = -0.2 m,
q = 1.65e-5 m/s), `ρS(c) = 0.2·10c/(1 + 10c)`. Front speed between 3 h and 6 h
against the shock speed `q/(θ + ρS(1))`:

| Δt [s] | Courant | rel. error | Newton its/step |
|---:|---:|---:|---:|
| 30 | 0.15 | 2.0e-3 | 4.0 |
| 600 | 3.1 | 5.1e-3 | 5.2 |
| 3600 | 18 | 2.2e-2 | 6.2 |

## Limits

- One soil per run: `NonlinearMap` sees the value only, not the cell, so the
  van Genuchten parameters cannot vary in space yet (`Ks` can, through the
  flux of the Laplacian; linear cell-dependent terms such as θc stay `Si`).
- Dirichlet, zero-gradient and empty boundaries; no flux or seepage switching
  conditions yet.
- Implicit Euler only.
