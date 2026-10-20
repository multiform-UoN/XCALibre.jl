# =============================================================================
# Solute with Langmuir sorption in unsaturated flow: Newton on both equations
# =============================================================================
#   flow:      Richards (richards_common.jl), Celia loam, steady unit-gradient
#              column: h = -0.2 m at the top, free drainage at the bottom
#   transport: ∂(θc + ρS(c))/∂t + ∇·(q c) - ∇·(D∇c) = 0,
#              ρS(c) = ρSmax b c/(1 + b c),  D = θ Dm + αL |q|
#              c = 1 enters at the top of a solute-free column
#
# The isotherm is concave, so the sorption front self-sharpens into a travelling
# wave that moves at the Rankine–Hugoniot speed
#     v = q / (θ + [ρS(1) - ρS(0)]/(1 - 0)),
# whatever the dispersion. The storage ρS(c)/Δt is a NonLinearSi with its analytic
# derivative, so each time step is a Newton solve; the linear θc part, which
# varies from cell to cell, stays a linear Si. The front position (c = 1/2) is
# compared with the shock for a sweep of time steps, up to Courant numbers ≫ 1.
#
# Run:  julia --project=. examples/richards/langmuir_transport_1D.jl
# =============================================================================

include("richards_common.jl")

const here = @__DIR__
mesh = FOAM3D_mesh(joinpath(here, "column1D", "polyMesh"), scale=1)
backend = CPU(); hardware = Hardware(backend=backend, workgroup=1024)
mesh = adapt(backend, mesh)

soil = VanGenuchten(θr=0.102, θs=0.368, α=3.35, n=2.0, l=0.5)
Ks = 9.22e-5
ĝ = SVector(0.0, 0.0, -1.0)
h_top = -0.2
Dm, αL = 1e-9, 0.01
ρSmax, b = 0.2, 10.0
S(c) = ρSmax*b*c/(1 + b*c)
dS(c) = ρSmax*b/(1 + b*c)^2

BCs = assign(region=mesh, (
    h = [Dirichlet(:top, h_top), Zerogradient(:bottom), Empty(:sides)],
    c = [Dirichlet(:top, 1.0), Zerogradient(:bottom), Empty(:sides)],
))
setup = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-12,
    relax=1.0, rtol=1e-14, atol=1e-16, itmax=1000)
solvers = (h=setup, c=setup)
schemes = (h=Schemes(laplacian=Linear, divergence=Upwind), c=Schemes(laplacian=Linear, divergence=Upwind))
config = Configuration(solvers=solvers, schemes=schemes, hardware=hardware, boundaries=BCs,
    runtime=Runtime(iterations=1, write_interval=-1, time_step=1.0))

z = [cell.centre[3] for cell in mesh.cells]
order = sortperm(z)

prepare(eqn, s) = begin
    @reset eqn.preconditioner = set_preconditioner(s.preconditioner, eqn)
    @reset eqn.solver = _workspace(s.solver, XCALibre._b(eqn))
end

# ── flow: one Newton solve of the steady problem (large Δt) from a perturbed guess
h = ScalarField(mesh); initialise!(h, -0.5)
Fg = gravity_flux(mesh, Ks, ĝ)
dt = Ref(1e8)
rhs = ScalarField(mesh); @. rhs.values = θ(soil, h.values)/dt[]
heqn = prepare(richards_operator(soil, Ks, Fg, dt, rhs, BCs.h, solvers.h)(h), setup)
for k in 1:5                     # pseudo-time steps towards the steady state
    @. rhs.values = θ(soil, h.values)/dt[]
    ok, it, _ = newton_step!(heqn, h, BCs.h, config; htol=1e-12)
    @printf("flow pseudo-step %d: Newton %s in %d iterations, h ∈ [%.6f, %.6f]\n",
        k, ok ? "converged" : "FAILED", it, extrema(h.values)...)
end
F = FaceScalarField(mesh)
darcy_flux!(F, mesh, soil, Ks, Fg, h, dirichlet_faces(mesh, Dict(:top => h_top)))
θc = [θ(soil, v) for v in h.values]
q = Ks*kr(soil, h_top)
bottom = mesh.boundaries[findfirst(b -> b.name == :bottom, mesh.boundaries)].IDs_range[1]
@printf("unit-gradient flux: q = %.6e m/s at the bottom, Ks kr(h_top) = %.6e, θ = %.4f\n",
    F.values[bottom]/mesh.faces[bottom].area, q, θc[1])

v_shock = q/(θ(soil, h_top) + S(1.0) - S(0.0))
speed = cell_speed(mesh, F)
Dface = FaceScalarField(mesh)
for (i, f) in enumerate(mesh.faces)
    P, N = f.ownerCells
    Dface.values[i] = (θc[P]*Dm + αL*speed[P] + θc[N]*Dm + αL*speed[N])/2
end

# ── transport: Newton with NonLinearSi(ρS(c)/Δt, ρS'(c)/Δt)
function run_transport(Δt, t_end)
    c = ScalarField(mesh); initialise!(c, 0.0)
    rΔt = Ref(Δt)
    θdt = ScalarField(mesh); @. θdt.values = θc/Δt
    crhs = ScalarField(mesh)
    L = ((Si(θdt) + NonLinearSi(x -> S(x)/rΔt[], x -> dS(x)/rΔt[])
          + Divergence{Upwind}(F) - Laplacian{Linear}(Dface) == Source(crhs)) → BCs.c) → solvers.c
    ceqn = prepare(L(c), setup)
    t = 0.0; its = Int[]; fronts = Dict{Float64,Float64}()
    while t < t_end - 1e-9
        @. crhs.values = (θc*c.values + S(c.values))/Δt
        ok, it, _ = newton_step!(ceqn, c, BCs.c, config; htol=1e-10)
        ok || error("transport Newton failed at t = $t")
        push!(its, it); t += Δt
        for tm in (t_end/2, t_end)
            abs(t - tm) < 1e-6 && (fronts[tm] = front(c))
        end
    end
    return (fronts=fronts, its=its)
end

# depth of the c = 1/2 point below the top, by linear interpolation
function front(c)
    cs, zs = c.values[order], z[order]
    k = findlast(i -> cs[i] < 0.5 <= cs[i+1], 1:length(cs)-1)
    zf = zs[k] + (0.5 - cs[k])*(zs[k+1] - zs[k])/(cs[k+1] - cs[k])
    return 1 - zf
end

t_end = 6*3600.0
dz = 0.01
@printf("\nRankine–Hugoniot shock speed v = %.5e m/s\n", v_shock)
@printf("%8s %8s %14s %10s %10s %8s\n", "Δt [s]", "Courant", "front speed", "rel. err", "Newton/st", "max its")
for Δt in (30.0, 120.0, 600.0, 1800.0, 3600.0)
    r = run_transport(Δt, t_end)
    vf = (r.fronts[t_end] - r.fronts[t_end/2])/(t_end/2)
    @printf("%8.0f %8.2f %14.5e %10.2e %10.2f %8d\n", Δt, q/θc[1]*Δt/dz, vf,
        vf/v_shock - 1, sum(r.its)/length(r.its), maximum(r.its))
end
