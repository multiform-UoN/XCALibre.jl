# XCALibre.jl Non-Newtonian Examples

This directory contains compact examples for the XCALibre non-Newtonian and
viscoelastic finite-volume machinery.  Keep these examples small and useful for
XCALibre users:

- `stokes_incomp_channel.jl`: Newtonian incompressible channel Stokes example.
- `stokes_incomp_bend.jl`: imported OpenFOAM bend-mesh Stokes example.
- `maxwell_incomp_channel.jl`: pressure-plus-stress Maxwell prototype.
- `oldroyd_incomp_channel.jl`: pressure-plus-stress Oldroyd-B prototype.
- `kv_incomp_channel.jl`: Kelvin-Voigt history prototype.
- `maxwell_objective_rates_channel.jl`: objective-rate comparison prototype.
- `branch_practical_up.jl`: practical `(u,p,tau)` formulation sketch.
- `branch_experimental_usigma.jl`: total-stress formulation sketch.

The detailed paper-specific benchmark and audit scripts have been moved to:

```text
/Volumes/OpenFOAM/mixed_viscoelasticity/xcalibre/benchmarks
```

Use that folder for GMRTFoam/MFEM/OpenFOAM comparison work.  The current strict
cross-code gate there is Newtonian Stokes on a common OpenFOAM mesh; the
stress-coupled Maxwell/Oldroyd-B scripts remain diagnostic until the
Newtonian-equivalent stress-field audit is green.
