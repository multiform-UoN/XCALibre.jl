# =============================================================================
# Celia, Bouloutas & Zarba (1990) infiltration into a dry loam, with a tracer
# =============================================================================
# Shared benchmark with GMRTFoam's richardsHeadFoam (tutorial
# tutorials/richardsHeadFoam/celiaInfiltration1D, same polyMesh).
#
#   1 m column, 100 cells, z up; van Genuchten θr = 0.102, θs = 0.368,
#   α = 3.35 1/m, n = 2, Mualem l = 0.5, Ks = 9.22e-5 m/s
#   h(z, 0) = -10 m;  h = -0.75 m at the top, h = -10 m at the bottom; 1 day
#   tracer: c = 0 initially, c = 1 in the infiltrating water, Dm = 1e-9 m²/s,
#   αL = 0.01 m
#
# Newton with the exact (analytic) Jacobian and a backtracking line search, at a
# fixed Δt = 100 s like the GMRTFoam case; a failed step is halved and retried, and
# the substeps grow back ×1.25 after ≤ 4 iterations, as in richardsHeadFoam.
# Profiles are written to celia_xcalibre_<t>.csv at 6, 12, 18 and 24 h; compare
# them with tutorials/richardsHeadFoam/celiaInfiltration1D/compare.py in GMRTFoam.
#
# Run:  julia --project=. examples/richards/celia_infiltration_1D.jl
# =============================================================================

include("richards_common.jl")

const here = @__DIR__
outdir = get(ENV, "RICHARDS_OUT", here)

mesh = FOAM3D_mesh(joinpath(here, "column1D", "polyMesh"), scale=1)
backend = CPU(); hardware = Hardware(backend=backend, workgroup=1024)
mesh = adapt(backend, mesh)

soil = VanGenuchten(θr=0.102, θs=0.368, α=3.35, n=2.0, l=0.5)
Ks = 9.22e-5
ĝ = SVector(0.0, 0.0, -1.0)
h_init, h_top, h_bottom = -10.0, -0.75, -10.0
Dm, αL = 1e-9, 0.01
t_end, write_times = 86400.0, [21600.0, 43200.0, 64800.0, 86400.0]
dt0, dt_max, htol = 10.0, 1800.0, 1e-7
# Fixed Δt = 100 s by default, as in the GMRTFoam case (a step is split only when Newton
# fails); RICHARDS_FIXED_DT=<s> changes it, RICHARDS_FIXED_DT=adaptive uses the
# iteration-count control instead
fixed_dt = get(ENV, "RICHARDS_FIXED_DT", "100") == "adaptive" ? nothing :
    parse(Float64, get(ENV, "RICHARDS_FIXED_DT", "100"))
fixed_dt === nothing || (dt0 = dt_max = fixed_dt)

BCs = assign(region=mesh, (
    h = [Dirichlet(:top, h_top), Dirichlet(:bottom, h_bottom), Empty(:sides)],
    c = [Dirichlet(:top, 1.0), Zerogradient(:bottom), Empty(:sides)],
))
solvers = (
    h = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-12,
        relax=1.0, rtol=1e-14, atol=1e-16, itmax=1000),
    c = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-12,
        relax=1.0, rtol=1e-14, atol=1e-16, itmax=1000),
)
schemes = (h=Schemes(laplacian=Linear, divergence=Upwind), c=Schemes(laplacian=Linear, divergence=Upwind))
config = Configuration(solvers=solvers, schemes=schemes, hardware=hardware, boundaries=BCs,
    runtime=Runtime(iterations=1, write_interval=-1, time_step=dt0))

h = ScalarField(mesh); initialise!(h, h_init)
c = ScalarField(mesh); initialise!(c, 0.0)
ncell = length(mesh.cells)
vol = [cell.volume for cell in mesh.cells]
z = [cell.centre[3] for cell in mesh.cells]

Fg = gravity_flux(mesh, Ks, ĝ)
dt = Ref(dt0)
rhs = ScalarField(mesh)                                # θⁿ/Δt
L = richards_operator(soil, Ks, Fg, dt, rhs, BCs.h, solvers.h)
eqn = L(h)
@reset eqn.preconditioner = set_preconditioner(solvers.h.preconditioner, eqn)
@reset eqn.solver = _workspace(solvers.h.solver, XCALibre._b(eqn))
bvals = dirichlet_faces(mesh, Dict(:top => h_top, :bottom => h_bottom))

# tracer operator: (θ/Δt) c + ∇·(F c) - ∇·(D ∇c) = θⁿ cⁿ/Δt
F = FaceScalarField(mesh); Dface = FaceScalarField(mesh)
θdt = ScalarField(mesh); crhs = ScalarField(mesh)
Lc = ((Si(θdt) + Divergence{Upwind}(F) - Laplacian{Linear}(Dface) == Source(crhs)) → BCs.c) → solvers.c
ceqn = Lc(c)
@reset ceqn.preconditioner = set_preconditioner(solvers.c.preconditioner, ceqn)
@reset ceqn.solver = _workspace(solvers.c.solver, XCALibre._b(ceqn))

function tracer_step!(θold, θnew, Δt)
    darcy_flux!(F, mesh, soil, Ks, Fg, h, bvals)
    speed = cell_speed(mesh, F)
    Dc = θnew .* Dm .+ αL .* speed
    for (i, f) in enumerate(mesh.faces)
        P, N = f.ownerCells
        Dface.values[i] = (Dc[P] + Dc[N])/2
    end
    @. θdt.values = θnew/Δt
    @. crhs.values = θold*c.values/Δt
    solve_equation!(ceqn, config)
end

write_profile(t) = open(joinpath(outdir, @sprintf("celia_xcalibre_%d.csv", round(Int, t))), "w") do io
    println(io, "z,h,theta,c")
    for k in sortperm(z)
        @printf(io, "%.6f,%.12e,%.12e,%.12e\n", z[k], h.values[k], θ(soil, h.values[k]), c.values[k])
    end
end

water(hv) = sum(θ(soil, hv[k])*vol[k] for k in 1:ncell)

t = 0.0; nsteps = 0; nnewton = 0; nrejected = 0; balance = 0.0
iters_hist = Int[]
wi = 1
elapsed = @elapsed while t < t_end - 1e-9
    global t, nsteps, nnewton, nrejected, balance, wi
    # next stop: the next write time, and in fixed-Δt mode the next multiple of Δt
    t_stop = write_times[wi]
    fixed_dt === nothing || (t_stop = min(t_stop, (floor(t/fixed_dt + 1e-9) + 1)*fixed_dt))
    Δt = min(dt[], t_stop - t)
    hold = copy(h.values)
    θold = [θ(soil, v) for v in hold]
    dt[] = Δt
    @. rhs.values = θold/Δt
    ok, it, res = newton_step!(eqn, h, BCs.h, config; htol=htol)
    nnewton += it
    if !ok
        h.values .= hold; dt[] = Δt/2; nrejected += 1
        dt[] < 1e-3 && error("Newton failed with Δt = $(dt[])")
        continue
    end
    θnew = [θ(soil, v) for v in h.values]
    darcy_flux!(F, mesh, soil, Ks, Fg, h, bvals)
    inflow = -Δt*sum(F.values[i] for i in keys(bvals))
    balance += sum((θnew .- θold) .* vol) - inflow
    tracer_step!(θold, θnew, Δt)
    t += Δt; nsteps += 1; push!(iters_hist, it)
    dt[] = it <= 4 ? min(1.25Δt, dt_max) : it >= 10 ? 0.7Δt : Δt
    # fixed-Δt mode: each step starts again from the full Δt, as in richardsHeadFoam
    fixed_dt !== nothing && abs(t - t_stop) < 1e-9 && (dt[] = fixed_dt)
    # keep the step reached before a write time was clipped
    if abs(t - write_times[wi]) < 1e-9
        write_profile(t); @printf("t = %6.0f s  steps %4d  Newton its %5d  water balance error %.2e m3\n",
            t, nsteps, nnewton, balance)
        wi = min(wi + 1, length(write_times))
        dt[] = max(dt[], Δt)
    end
end

@printf("\nXCALibre Newton: %d steps, %d Newton iterations (%.2f/step), %d rejected, %.1f s wall\n",
    nsteps, nnewton, nnewton/nsteps, nrejected, elapsed)
@printf("water balance error %.3e m3, tracer mass %.6e\n", balance,
    sum(θ(soil, h.values[k])*c.values[k]*vol[k] for k in 1:ncell))
