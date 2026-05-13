# =============================================================================
# Bend viscoelastic monolithic conditioning diagnostics
# =============================================================================
#
# Usage examples:
#   julia --project=. examples/nonNewtonian/benchmarks/bend/viscoelastic_conditioning_diagnostics.jl Stokes Jacobi 1 5000
#   julia --project=. examples/nonNewtonian/benchmarks/bend/viscoelastic_conditioning_diagnostics.jl Maxwell None 1 1000
#   julia --project=. examples/nonNewtonian/benchmarks/bend/viscoelastic_conditioning_diagnostics.jl All Jacobi 3 5000 true
#
# Models: Stokes, Maxwell, Jaumann, StretchingOnlyUCD, OldroydB, All
# Preconditioners: None, Jacobi, NormDiagonal, DILU
# =============================================================================

include("../benchmark_utils.jl")

const MODEL = length(ARGS) >= 1 ? Symbol(ARGS[1]) : :All
const PRECON = length(ARGS) >= 2 ? Symbol(ARGS[2]) : :Jacobi
const NSTEPS = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 3
const ITMAX = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 5000
const EQUILIBRATE = length(ARGS) >= 5 ? parse(Bool, ARGS[5]) : false

const ALL_MODELS = (:Stokes, :Maxwell, :Jaumann, :StretchingOnlyUCD, :OldroydB)
@assert MODEL == :All || MODEL in ALL_MODELS
@assert PRECON in (:None, :Jacobi, :NormDiagonal, :DILU)

preconditioner(::Val{:Jacobi}) = Jacobi()
preconditioner(::Val{:NormDiagonal}) = NormDiagonal()
preconditioner(::Val{:DILU}) = DILU()

function scalar_setup(precon)
    SolverSetup(solver=Gmres(), preconditioner=precon, convergence=1e-10, relax=1.0)
end

function make_solvers(precon_symbol, names)
    precon_symbol == :None && return nothing
    precon = preconditioner(Val(precon_symbol))
    values = map(_ -> scalar_setup(precon), names)
    return NamedTuple{names}(values)
end

function make_config(mesh_dev, BCs, names; precon_symbol=PRECON)
    solvers = make_solvers(precon_symbol, names)
    scheme_values = map(_ -> Schemes(), names)
    return Configuration(
        solvers=solvers,
        schemes=NamedTuple{names}(scheme_values),
        runtime=Runtime(iterations=1, write_interval=-1, time_step=0.01),
        hardware=Hardware(backend=CPU(), workgroup=1024),
        boundaries=BCs,
    )
end

function bend_bcs(mesh_dev, fields)
    base = (
        u=[Dirichlet(:walls, 0.0), Zerogradient(:inlet), Zerogradient(:outlet), Symmetry(:frontAndBack)],
        v=[Dirichlet(:walls, 0.0), Zerogradient(:inlet), Zerogradient(:outlet), Symmetry(:frontAndBack)],
        p=[Zerogradient(:walls), Zerogradient(:inlet), Zerogradient(:outlet), Symmetry(:frontAndBack)],
    )
    if fields == (:u, :v, :p)
        return assign(region=mesh_dev, base)
    end
    stress = (
        txx=[Zerogradient(:walls), Zerogradient(:inlet), Zerogradient(:outlet), Symmetry(:frontAndBack)],
        tyy=[Zerogradient(:walls), Zerogradient(:inlet), Zerogradient(:outlet), Symmetry(:frontAndBack)],
        txy=[Zerogradient(:walls), Zerogradient(:inlet), Zerogradient(:outlet), Symmetry(:frontAndBack)],
    )
    return assign(region=mesh_dev, merge(base, stress))
end

function new_scalar(mesh_dev, value=0.0)
    f = ScalarField(mesh_dev)
    initialise!(f, value)
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

function update_mass_flux!(mass_flux, u, v)
    mesh = u.mesh
    nbf = length(mesh.boundary_cellsID)
    for fID in eachindex(mesh.faces)
        face = mesh.faces[fID]
        if fID > nbf
            c1 = face.ownerCells[1]
            c2 = face.ownerCells[2]
            uf = 0.5 * (u.values[c1] + u.values[c2])
            vf = 0.5 * (v.values[c1] + v.values[c2])
            mass_flux.values[fID] = (uf * face.normal[1] + vf * face.normal[2]) * face.area
        else
            c1 = face.ownerCells[1]
            mass_flux.values[fID] = (u.values[c1] * face.normal[1] + v.values[c1] * face.normal[2]) * face.area
        end
    end
    return nothing
end

function matrix_spread(A_csr)
    A = get_sparse_matrix(A_csr)
    diag_abs = abs.(diag(A))
    row_norms = vec(sum(abs, A; dims=2))
    nz_diag = diag_abs[diag_abs .> 0]
    nz_rows = row_norms[row_norms .> 0]
    return (
        n=size(A, 1),
        nnz=nnz(A),
        diag_spread=maximum(nz_diag) / minimum(nz_diag),
        row1_spread=maximum(nz_rows) / minimum(nz_rows),
    )
end

function print_step(model, precon, step, solve_result, u, v, p, stress_fields, spread)
    stat = solve_result.linear[end]
    max_u = max(maximum(abs.(u.values)), maximum(abs.(v.values)))
    max_p = maximum(abs.(p.values))
    max_tau = stress_fields === nothing ? 0.0 :
        maximum(map(f -> maximum(abs.(f.values)), stress_fields))
    @printf(
        "%-17s %-12s eq=%5s step=%02d solved=%5s gmres=%5d krylov=%.3e true=%.3e relTrue=%.3e max|U|=%.6e max|p|=%.6e max|tau|=%.6e diagSpread=%.3e rowSpread=%.3e\n",
        String(model), String(precon), string(EQUILIBRATE), step, string(stat.solved), stat.gmres_iterations,
        stat.final_residual, stat.true_residual, stat.relative_true_residual,
        max_u, max_p, max_tau,
        spread.diag_spread, spread.row1_spread,
    )
end

function run_stokes(mesh_dev, precon_symbol, nsteps, itmax)
    fields = (:u, :v, :p)
    u = new_scalar(mesh_dev); v = new_scalar(mesh_dev); p = new_scalar(mesh_dev)
    BCs = bend_bcs(mesh_dev, fields)
    config = make_config(mesh_dev, BCs, fields; precon_symbol=precon_symbol)
    solvers = make_solvers(:Jacobi, fields)

    mu = ConstantScalar(1.0)
    one = ConstantScalar(1.0)
    tau_rc = ConstantScalar(0.1)

    for step in 1:nsteps
        L_u = ((-Laplacian{Linear}(mu) + ScalarGrad{Linear,1}(one, p) == Source(1.0)) → BCs.u) → solvers.u
        L_v = ((-Laplacian{Linear}(mu) + ScalarGrad{Linear,2}(one, p) == Source(1.0)) → BCs.v) → solvers.v
        L_p = ((-Laplacian{Linear}(tau_rc) + VectorDiv{Linear,1}(one, u) + VectorDiv{Linear,2}(one, v) == Source(0.0)) → BCs.p) → solvers.p
        sys = MonolithicSystem([L_u(u), L_v(v), L_p(p)], [u, v, p])
        A_csr, _ = assemble_monolithic_system(sys, (BCs.u, BCs.v, BCs.p), config)
        XCALibre.Solve.apply_monolithic_reference!(A_csr, zeros(eltype(u.values), 3 * length(u.values)), sys, 3, 0.0, 1)
        spread = matrix_spread(A_csr)
        result = solve_monolithic!(
            sys, (BCs.u, BCs.v, BCs.p), config;
            reference=(3, 0.0, 1), diagnostics=true,
            use_preconditioner=(precon_symbol != :None), equilibrate=EQUILIBRATE,
            itmax=itmax,
        )
        print_step(:Stokes, precon_symbol, step, result, u, v, p, nothing, spread)
    end
end

function run_viscoelastic(model, mesh_dev, precon_symbol, nsteps, itmax)
    fields = (:u, :v, :p, :txx, :tyy, :txy)
    u = new_scalar(mesh_dev); v = new_scalar(mesh_dev); p = new_scalar(mesh_dev)
    txx = new_scalar(mesh_dev); tyy = new_scalar(mesh_dev); txy = new_scalar(mesh_dev)
    BCs = bend_bcs(mesh_dev, fields)
    config = make_config(mesh_dev, BCs, fields; precon_symbol=precon_symbol)
    solvers = make_solvers(:Jacobi, fields)

    one = ConstantScalar(1.0)
    mu_s = ConstantScalar(model == :Maxwell ? 1e-6 : 1.0)
    tau_rc = ConstantScalar(0.1)
    inv_lam = ConstantScalar(1.0)
    two_mu_lam = ConstantScalar(2.0)
    mu_lam = ConstantScalar(1.0)

    coeff_names = (
        :w, :neg_2w, :pos_2w, :neg_2ux, :neg_2uy, :neg_2vx,
        :neg_2vy, :neg_uy, :neg_vx, :neg_trace,
    )
    coeff_values = map(_ -> new_scalar(mesh_dev), coeff_names)
    coeffs = NamedTuple{coeff_names}(coeff_values)
    mass_flux = FaceScalarField(mesh_dev)

    for step in 1:nsteps
        if model in (:Jaumann, :StretchingOnlyUCD)
            update_rate_coefficients!(coeffs, u, v, BCs, config)
        elseif model == :OldroydB
            update_mass_flux!(mass_flux, u, v)
        end

        L_u = ((-Laplacian{Linear}(mu_s) - ScalarGrad{Linear,1}(one, txx) - ScalarGrad{Linear,2}(one, txy) + ScalarGrad{Linear,1}(one, p) == Source(1.0)) → BCs.u) → solvers.u
        L_v = ((-Laplacian{Linear}(mu_s) - ScalarGrad{Linear,1}(one, txy) - ScalarGrad{Linear,2}(one, tyy) + ScalarGrad{Linear,2}(one, p) == Source(1.0)) → BCs.v) → solvers.v
        L_p = ((-Laplacian{Linear}(tau_rc) + VectorDiv{Linear,1}(one, u) + VectorDiv{Linear,2}(one, v) == Source(0.0)) → BCs.p) → solvers.p

        if model == :Jaumann
            L_txx = ((Si(inv_lam) + Si(coeffs.neg_2w, txy) - ScalarGrad{Linear,1}(two_mu_lam, u) == Source(0.0)) → BCs.txx) → solvers.txx
            L_tyy = ((Si(inv_lam) + Si(coeffs.pos_2w, txy) - ScalarGrad{Linear,2}(two_mu_lam, v) == Source(0.0)) → BCs.tyy) → solvers.tyy
            L_txy = ((Si(inv_lam) + Si(coeffs.w, txx) - Si(coeffs.w, tyy) - ScalarGrad{Linear,2}(mu_lam, u) - ScalarGrad{Linear,1}(mu_lam, v) == Source(0.0)) → BCs.txy) → solvers.txy
        elseif model == :StretchingOnlyUCD
            L_txx = ((Si(inv_lam) + Si(coeffs.neg_2ux, txx) + Si(coeffs.neg_2uy, txy) - ScalarGrad{Linear,1}(two_mu_lam, u) == Source(0.0)) → BCs.txx) → solvers.txx
            L_tyy = ((Si(inv_lam) + Si(coeffs.neg_2vx, txy) + Si(coeffs.neg_2vy, tyy) - ScalarGrad{Linear,2}(two_mu_lam, v) == Source(0.0)) → BCs.tyy) → solvers.tyy
            L_txy = ((Si(inv_lam) + Si(coeffs.neg_vx, txx) + Si(coeffs.neg_uy, tyy) + Si(coeffs.neg_trace, txy) - ScalarGrad{Linear,2}(mu_lam, u) - ScalarGrad{Linear,1}(mu_lam, v) == Source(0.0)) → BCs.txy) → solvers.txy
        elseif model == :OldroydB
            L_txx = ((Si(inv_lam) + Divergence{Upwind}(mass_flux) - ScalarGrad{Linear,1}(two_mu_lam, u) == Source(0.0)) → BCs.txx) → solvers.txx
            L_tyy = ((Si(inv_lam) + Divergence{Upwind}(mass_flux) - ScalarGrad{Linear,2}(two_mu_lam, v) == Source(0.0)) → BCs.tyy) → solvers.tyy
            L_txy = ((Si(inv_lam) + Divergence{Upwind}(mass_flux) - ScalarGrad{Linear,2}(mu_lam, u) - ScalarGrad{Linear,1}(mu_lam, v) == Source(0.0)) → BCs.txy) → solvers.txy
        else
            L_txx = ((Si(inv_lam) - ScalarGrad{Linear,1}(two_mu_lam, u) == Source(0.0)) → BCs.txx) → solvers.txx
            L_tyy = ((Si(inv_lam) - ScalarGrad{Linear,2}(two_mu_lam, v) == Source(0.0)) → BCs.tyy) → solvers.tyy
            L_txy = ((Si(inv_lam) - ScalarGrad{Linear,2}(mu_lam, u) - ScalarGrad{Linear,1}(mu_lam, v) == Source(0.0)) → BCs.txy) → solvers.txy
        end

        sys = MonolithicSystem([L_u(u), L_v(v), L_p(p), L_txx(txx), L_tyy(tyy), L_txy(txy)], [u, v, p, txx, tyy, txy])
        A_csr, _ = assemble_monolithic_system(sys, (BCs.u, BCs.v, BCs.p, BCs.txx, BCs.tyy, BCs.txy), config)
        XCALibre.Solve.apply_monolithic_reference!(A_csr, zeros(eltype(u.values), 6 * length(u.values)), sys, 3, 0.0, 1)
        spread = matrix_spread(A_csr)
        result = solve_monolithic!(
            sys, (BCs.u, BCs.v, BCs.p, BCs.txx, BCs.tyy, BCs.txy), config;
            reference=(3, 0.0, 1), diagnostics=true,
            use_preconditioner=(precon_symbol != :None), equilibrate=EQUILIBRATE,
            itmax=itmax,
        )
        print_step(model, precon_symbol, step, result, u, v, p, (txx, tyy, txy), spread)
    end
end

function main()
    _, mesh_dev = get_bend_mesh()
    models = MODEL == :All ? ALL_MODELS : (MODEL,)
    @printf("model preconditioner nsteps=%d itmax=%d equilibrate=%s\n", NSTEPS, ITMAX, string(EQUILIBRATE))
    for model in models
        if model == :Stokes
            run_stokes(mesh_dev, PRECON, NSTEPS, ITMAX)
        else
            run_viscoelastic(model, mesh_dev, PRECON, NSTEPS, ITMAX)
        end
    end
end

main()
