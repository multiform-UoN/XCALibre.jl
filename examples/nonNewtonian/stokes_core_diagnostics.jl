using XCALibre
using LinearAlgebra
using Printf
using SparseArrays

function channel_mesh(nx=4, ny=4; length=1.0, height=1.0)
    p1 = Point(0.0, 0.0, 0.0)
    p2 = Point(length, 0.0, 0.0)
    p3 = Point(length, height, 0.0)
    p4 = Point(0.0, height, 0.0)
    points = [p1, p2, p3, p4]

    e1 = line!(points, 1, 2, nx)
    e2 = line!(points, 2, 3, ny)
    e3 = line!(points, 4, 3, nx)
    e4 = line!(points, 1, 4, ny)
    edges = [e1, e2, e3, e4]

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

function csr_to_sparse(A)
    rows = Vector{Int}(undef, length(A.nzval))
    for r in 1:(length(A.rowptr) - 1)
        for nzi in A.rowptr[r]:(A.rowptr[r + 1] - 1)
            rows[nzi] = r
        end
    end
    return sparse(rows, A.colval, A.nzval, size(A)...)
end

function matrix_stats(A; tol=1e-9)
    dense = Matrix(A)
    s = svdvals(dense)
    smax = maximum(s)
    threshold = tol * max(smax, 1.0)
    nullity = count(<(threshold), s)
    cond_est = minimum(s) <= threshold ? Inf : smax / minimum(s)
    evals = eigvals(dense)
    smallest = sort(evals; by=abs)[1:min(8, length(evals))]
    return (; cond_est, nullity, min_sv=minimum(s), smallest)
end

function build_stokes_system(mesh; mu=1.0, force_x=1.0, tau_p=0.0, eps_p=0.0, gamma_gd=0.0)
    u = ScalarField(mesh); initialise!(u, 0.0)
    v = ScalarField(mesh); initialise!(v, 0.0)
    p = ScalarField(mesh); initialise!(p, 0.0)

    bcs = assign(
        region=mesh,
        (
            u=[
                Dirichlet(:top, 0.0),
                Dirichlet(:bottom, 0.0),
                Zerogradient(:inlet),
                Zerogradient(:outlet),
            ],
            v=[
                Dirichlet(:top, 0.0),
                Dirichlet(:bottom, 0.0),
                Zerogradient(:inlet),
                Zerogradient(:outlet),
            ],
            p=[
                Zerogradient(:top),
                Zerogradient(:bottom),
                Zerogradient(:inlet),
                Zerogradient(:outlet),
            ],
        ),
    )

    solvers = (
        u=SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-12, relax=1.0),
        v=SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-12, relax=1.0),
        p=SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-12, relax=1.0),
    )
    config = Configuration(
        solvers=solvers,
        schemes=(u=Schemes(), v=Schemes(), p=Schemes()),
        runtime=Runtime(iterations=1, write_interval=-1, time_step=1.0),
        hardware=Hardware(backend=CPU(), workgroup=1024),
        boundaries=bcs,
    )

    mu_c = ConstantScalar(mu)
    one_c = ConstantScalar(1.0)
    tau_c = ConstantScalar(tau_p)
    eps_c = ConstantScalar(eps_p)
    gd_c = ConstantScalar(gamma_gd)

    L_u = ((
        (
            -Laplacian{Linear}(mu_c)
            -GradDiv{Linear,1,1}(gd_c)
            -GradDiv{Linear,1,2}(gd_c, v)
            + ScalarGrad{Linear,1}(one_c, p)
        ) == Source(force_x)
    ) → bcs.u) → solvers.u

    L_v = ((
        (
            -Laplacian{Linear}(mu_c)
            -GradDiv{Linear,2,1}(gd_c, u)
            -GradDiv{Linear,2,2}(gd_c)
            + ScalarGrad{Linear,2}(one_c, p)
        ) == Source(0.0)
    ) → bcs.v) → solvers.v

    L_p = ((
        (
            -Laplacian{Linear}(tau_c)
            + Si(eps_c)
            + VectorDiv{Linear,1}(one_c, u)
            + VectorDiv{Linear,2}(one_c, v)
        ) == Source(0.0)
    ) → bcs.p) → solvers.p

    sys = MonolithicSystem([L_u(u), L_v(v), L_p(p)], [u, v, p])
    return (; sys, fields=(u=u, v=v, p=p), bcs=(bcs.u, bcs.v, bcs.p), config)
end

function assemble_sparse(case; pin_pressure=false)
    A_csr, b = XCALibre.Solve.assemble_monolithic_system(case.sys, case.bcs, case.config)
    if pin_pressure
        apply_monolithic_reference!(A_csr, b, case.sys, 3, 0.0, 1)
    end
    return csr_to_sparse(A_csr), b
end

function operator_blocks(case)
    n = case.sys.n_cells
    A_csr, b = XCALibre.Solve.assemble_monolithic_system(case.sys, case.bcs, case.config)
    A = csr_to_sparse(A_csr)
    Gx = A[1:n, 2n+1:3n]
    Gy = A[n+1:2n, 2n+1:3n]
    Bx = A[2n+1:3n, 1:n]
    By = A[2n+1:3n, n+1:2n]
    return (; Gx, Gy, Bx, By)
end

function solve_case!(case; pin_pressure=true)
    A, b = assemble_sparse(case; pin_pressure)
    x = A \ b
    set_fields!(case.sys, x)
    u = case.fields.u.values
    v = case.fields.v.values
    p = case.fields.p.values
    speed = sqrt.(u.^2 .+ v.^2)
    return (; residual=norm(A * x - b), max_speed=maximum(abs.(speed)), max_p=maximum(abs.(p)))
end

function run()
    mesh = channel_mesh(4, 4; length=1.0, height=1.0)
    variants = [
        (:free_saddle, false, 0.0, 0.0, 0.0),
        (:pin_only, true, 0.0, 0.0, 0.0),
        (:pressure_laplacian, true, 1.0e-3, 0.0, 0.0),
        (:grad_div, true, 0.0, 0.0, 1.0),
        (:artificial_compressibility, true, 0.0, 1.0e-6, 0.0),
    ]

    println("variant,pinned,nullity,cond_est,min_sv,max_speed,max_p,residual")
    for (name, pin, tau_p, eps_p, gamma_gd) in variants
        case = build_stokes_system(mesh; tau_p, eps_p, gamma_gd)
        A, b = assemble_sparse(case; pin_pressure=pin)
        stats = matrix_stats(A)
        result = pin ? solve_case!(case; pin_pressure=true) : (; residual=NaN, max_speed=NaN, max_p=NaN)
        @printf(
            "%s,%s,%d,%.6e,%.6e,%.6e,%.6e,%.6e\n",
            String(name), string(pin), stats.nullity, stats.cond_est, stats.min_sv,
            result.max_speed, result.max_p, result.residual,
        )
        println("  smallest eigs = ", join(string.(stats.smallest), ", "))
    end

    adj_case = build_stokes_system(mesh; tau_p=0.0, eps_p=0.0, gamma_gd=0.0)
    blocks = operator_blocks(adj_case)
    println("operator_checks")
    @printf("  ||Bx + Gx'||/||Bx|| = %.6e\n", norm(blocks.Bx + blocks.Gx') / max(norm(blocks.Bx), eps()))
    @printf("  ||By + Gy'||/||By|| = %.6e\n", norm(blocks.By + blocks.Gy') / max(norm(blocks.By), eps()))
end

if abspath(PROGRAM_FILE) == @__FILE__
    run()
end
