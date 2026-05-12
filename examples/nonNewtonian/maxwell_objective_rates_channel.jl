# =============================================================================
# Practical incompressible Maxwell objective-rate variants: (u, v, p, tau)
# =============================================================================
#
# This stays in the pressure plus extra-stress formulation,
#   sigma = -p I + tau,
# and intentionally omits the convective transport term u · grad(tau).  The
# objective-rate coefficients are lagged from the previous velocity iterate so
# the block solve remains linear.
#
# Run:
#   julia --project=. examples/nonNewtonian/maxwell_objective_rates_channel.jl Jaumann
#   julia --project=. examples/nonNewtonian/maxwell_objective_rates_channel.jl StretchingOnlyUCD
# =============================================================================

using XCALibre
using LinearAlgebra
using Printf

RATE = length(ARGS) >= 1 ? Symbol(ARGS[1]) : :Jaumann
@assert RATE in (:Jaumann, :StretchingOnlyUCD)

function small_channel_mesh(nx=8, ny=8)
    p1 = Point(0.0, 0.0, 0.0)
    p2 = Point(1.0, 0.0, 0.0)
    p3 = Point(1.0, 1.0, 0.0)
    p4 = Point(0.0, 1.0, 0.0)
    points = [p1, p2, p3, p4]
    edges = [
        line!(points, 1, 2, nx),
        line!(points, 2, 3, ny),
        line!(points, 4, 3, nx),
        line!(points, 1, 4, ny),
    ]
    blocks = [quad(edges, [1, 3, 4, 2])]
    patches = [
        Patch(:bottom, [1]),
        Patch(:outlet, [2]),
        Patch(:top, [3]),
        Patch(:inlet, [4]),
    ]
    mesh = generate!(MeshBuilder2D(points, edges, patches, blocks))
    return XCALibre.UNV2.update_mesh_format(mesh, Int64, Float64)
end

function fill_field!(f, value)
    f.values .= value
    return f
end

function update_rate_coefficients!(coeffs, u, v, BCs, config)
    mesh = u.mesh
    grad_u = Grad{Gauss}(u); uf = FaceScalarField(mesh)
    grad_v = Grad{Gauss}(v); vf = FaceScalarField(mesh)
    grad!(grad_u, uf, u, BCs.u, nothing, config)
    grad!(grad_v, vf, v, BCs.v, nothing, config)

    ux = grad_u.result.x.values
    uy = grad_u.result.y.values
    vx = grad_v.result.x.values
    vy = grad_v.result.y.values

    @. coeffs.w.values = 0.5 * (uy - vx)
    @. coeffs.neg_2w.values = -2.0 * coeffs.w.values
    @. coeffs.pos_2w.values = 2.0 * coeffs.w.values
    @. coeffs.neg_2ux.values = -2.0 * ux
    @. coeffs.neg_2uy.values = -2.0 * uy
    @. coeffs.neg_2vx.values = -2.0 * vx
    @. coeffs.neg_2vy.values = -2.0 * vy
    @. coeffs.neg_uy.values = -uy
    @. coeffs.neg_vx.values = -vx
    @. coeffs.neg_trace.values = -(ux + vy)
    return nothing
end

mesh = adapt(CPU(), small_channel_mesh())
hardware = Hardware(backend=CPU(), workgroup=1024)

u = ScalarField(mesh); initialise!(u, 0.0)
v = ScalarField(mesh); initialise!(v, 0.0)
p = ScalarField(mesh); initialise!(p, 0.0)
txx = ScalarField(mesh); initialise!(txx, 0.0)
tyy = ScalarField(mesh); initialise!(tyy, 0.0)
txy = ScalarField(mesh); initialise!(txy, 0.0)

BCs = assign(
    region=mesh,
    (
        u=[Dirichlet(:top, 0.0), Dirichlet(:bottom, 0.0), Zerogradient(:inlet), Zerogradient(:outlet)],
        v=[Dirichlet(:top, 0.0), Dirichlet(:bottom, 0.0), Zerogradient(:inlet), Zerogradient(:outlet)],
        p=[Zerogradient(:top), Zerogradient(:bottom), Zerogradient(:inlet), Zerogradient(:outlet)],
        txx=[Zerogradient(:top), Zerogradient(:bottom), Zerogradient(:inlet), Zerogradient(:outlet)],
        tyy=[Zerogradient(:top), Zerogradient(:bottom), Zerogradient(:inlet), Zerogradient(:outlet)],
        txy=[Zerogradient(:top), Zerogradient(:bottom), Zerogradient(:inlet), Zerogradient(:outlet)],
    ),
)

solvers = (
    u=SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
    v=SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
    p=SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
    txx=SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
    tyy=SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
    txy=SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
)
config = Configuration(
    solvers=solvers,
    schemes=(u=Schemes(), v=Schemes(), p=Schemes(), txx=Schemes(), tyy=Schemes(), txy=Schemes()),
    runtime=Runtime(iterations=1, write_interval=-1, time_step=1.0),
    hardware=hardware,
    boundaries=BCs,
)

mu_s = ConstantScalar(1.0)
mu_p = 1.0
lambda_p = 1.0
inv_lam = ConstantScalar(1.0 / lambda_p)
mu_lam = ConstantScalar(mu_p / lambda_p)
two_mu_lam = ConstantScalar(2.0 * mu_p / lambda_p)
one = ConstantScalar(1.0)
tau_rc = ConstantScalar(1.0e-3)

coeff_names = (
    :w, :neg_2w, :pos_2w, :neg_2ux, :neg_2uy, :neg_2vx,
    :neg_2vy, :neg_uy, :neg_vx, :neg_trace,
)
coeff_values = map(coeff_names) do _
    f = ScalarField(mesh)
    initialise!(f, 0.0)
    f
end
coeffs = NamedTuple{coeff_names}(coeff_values)

@info "Starting practical Maxwell objective-rate channel" RATE
mesh_writer = initialise_writer(VTK(), mesh)

for step in 1:4
    update_rate_coefficients!(coeffs, u, v, BCs, config)

    L_u = (((-Laplacian{Linear}(mu_s) - ScalarGrad{Linear,1}(one, txx) - ScalarGrad{Linear,2}(one, txy) + ScalarGrad{Linear,1}(one, p)) == Source(1.0)) → BCs.u) → solvers.u
    L_v = (((-Laplacian{Linear}(mu_s) - ScalarGrad{Linear,1}(one, txy) - ScalarGrad{Linear,2}(one, tyy) + ScalarGrad{Linear,2}(one, p)) == Source(0.0)) → BCs.v) → solvers.v
    L_p = (((-Laplacian{Linear}(tau_rc) + VectorDiv{Linear,1}(one, u) + VectorDiv{Linear,2}(one, v)) == Source(0.0)) → BCs.p) → solvers.p

    if RATE == :Jaumann
        L_txx = (((Si(inv_lam) + Si(coeffs.neg_2w, txy) - ScalarGrad{Linear,1}(two_mu_lam, u)) == Source(0.0)) → BCs.txx) → solvers.txx
        L_tyy = (((Si(inv_lam) + Si(coeffs.pos_2w, txy) - ScalarGrad{Linear,2}(two_mu_lam, v)) == Source(0.0)) → BCs.tyy) → solvers.tyy
        L_txy = (((Si(inv_lam) + Si(coeffs.w, txx) - Si(coeffs.w, tyy) - ScalarGrad{Linear,2}(mu_lam, u) - ScalarGrad{Linear,1}(mu_lam, v)) == Source(0.0)) → BCs.txy) → solvers.txy
    else
        L_txx = (((Si(inv_lam) + Si(coeffs.neg_2ux, txx) + Si(coeffs.neg_2uy, txy) - ScalarGrad{Linear,1}(two_mu_lam, u)) == Source(0.0)) → BCs.txx) → solvers.txx
        L_tyy = (((Si(inv_lam) + Si(coeffs.neg_2vx, txy) + Si(coeffs.neg_2vy, tyy) - ScalarGrad{Linear,2}(two_mu_lam, v)) == Source(0.0)) → BCs.tyy) → solvers.tyy
        L_txy = (((Si(inv_lam) + Si(coeffs.neg_vx, txx) + Si(coeffs.neg_uy, tyy) + Si(coeffs.neg_trace, txy) - ScalarGrad{Linear,2}(mu_lam, u) - ScalarGrad{Linear,1}(mu_lam, v)) == Source(0.0)) → BCs.txy) → solvers.txy
    end

    sys = MonolithicSystem([L_u(u), L_v(v), L_p(p), L_txx(txx), L_tyy(tyy), L_txy(txy)], [u, v, p, txx, tyy, txy])
    res = solve_monolithic!(sys, (BCs.u, BCs.v, BCs.p, BCs.txx, BCs.tyy, BCs.txy), config; reference=(3, 0.0, 1))
    max_tau = maximum(abs, vcat(txx.values, tyy.values, txy.values))
    @printf("Step %d, residual %.3e, max|u| %.6e, max|tau| %.6e\n", step, res, maximum(abs.(u.values)), max_tau)
end

write_results(0, 0.0, mesh, mesh_writer, BCs, ("u", u), ("v", v), ("p", p), ("txx", txx), ("tyy", tyy), ("txy", txy); suffix="_maxwell_rate")
@info "Benchmark maxwell_objective_rates finished."
