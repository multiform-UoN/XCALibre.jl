# Periodic-circle-2D Stokes comparison

Run `julia --project=. examples/homogenisation/permeability_tensor_2d_circle_gmsh.jl`
from the XCALibre repository root. This is the XCALibre finite-volume member
of a three-way comparison with:

- `PoreScaleHomogenisation.jl/homogenization_2d.jl` (Ferrite finite elements),
- `multiformFoam/tutorials/upscaling/homogenisationFoam/circleBenchmark`
  (OpenFOAM finite volumes).

All solve `-Δw^(j)+∇π^(j)=e_j`, `div(w^(j))=0` on the periodic square
`[-1,1]^2` minus a no-slip disk of radius `0.5`; all report
`K_ij=|Y|⁻¹ ∫_{Y_f} w_i^(j)`. In the XCALibre script,
`include_convection=false` selects this *linear Stokes* problem. Setting it
to `true` restores the steady Navier–Stokes variant, for which the velocity
response depends on forcing strength and should not be called linear
permeability.

| Implementation | Mesh | `Kxx` | `Kyy` |
| --- | --- | ---: | ---: |
| XCALibre | Gmsh triangles, `h=0.03` | 0.0792399 | 0.0790074 |
| PoreScaleHomogenisation.jl | curved P2/P1 triangles, `h=0.03` | 0.0790596 | 0.0790619 |
| multiformFoam | 100×100 Cartesian, stair-step disk | 0.0785069 | 0.0785069 |

The in-plane diagonal entries agree within about 1%. This is a numerical
cross-check, not proof of mesh convergence for all three methods. In
particular, the OpenFOAM disk removes whole Cartesian cells. The older
`fair_comparison.jl` uses a different, non-periodic streamtube and is not
part of this matched cell-problem comparison.
