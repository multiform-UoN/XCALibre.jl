# =============================================================================
# Incompressible Stokes: Straight Channel (Periodic/Body Force Driven)
# =============================================================================

include("../benchmark_utils.jl")

mesh_cpu, _ = get_straight_mesh()
mesh_per = XCALibre.Mesh.construct_periodic_topology(mesh_cpu, :inlet, :outlet, [25.0, 0.0, 0.0])
n_boundary = XCALibre.Mesh.total_boundary_faces(mesh_per)
n_total = length(mesh_per.faces)
n_internal = n_total - n_boundary
@info "Mesh Info" n_cells=length(mesh_per.cells) n_internal=n_internal n_boundary=n_boundary
mesh_dev = adapt(CPU(), mesh_per)

u = ScalarField(mesh_dev); initialise!(u, 0.0)
v = ScalarField(mesh_dev); initialise!(v, 0.0)
p = ScalarField(mesh_dev); initialise!(p, 0.0)

# Rewired patches are now internal. Standard BC loop should ignore them.
BCs = assign(
    region = mesh_dev,
    (
        u = [Dirichlet(:top, 0.0), Dirichlet(:bottom, 0.0), Empty(:inlet), Empty(:outlet)],
        v = [Dirichlet(:top, 0.0), Dirichlet(:bottom, 0.0), Empty(:inlet), Empty(:outlet)],
        p = [Zerogradient(:top), Zerogradient(:bottom), Empty(:inlet), Empty(:outlet)],
    )
)

solvers = (
    u = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
    v = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
    p = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
)
config = Configuration(solvers=solvers, schemes=(u=Schemes(), v=Schemes(), p=Schemes()),
                       runtime=Runtime(iterations=1, write_interval=-1, time_step=1.0), hardware=Hardware(backend=CPU(), workgroup=1024), boundaries=BCs)

mu_cst = ConstantScalar(1.0)
one_cst = ConstantScalar(1.0)
tau_rc_cst = ConstantScalar(0.1)

L_u = ((- Laplacian{XCALibre.Linear}(mu_cst) + ScalarGrad{XCALibre.Linear,1}(one_cst, p) == Source(1.0)) → BCs.u) → solvers.u
L_v = ((- Laplacian{XCALibre.Linear}(mu_cst) + ScalarGrad{XCALibre.Linear,2}(one_cst, p) == Source(0.0)) → BCs.v) → solvers.v
L_p = ((- Laplacian{XCALibre.Linear}(tau_rc_cst) + VectorDiv{XCALibre.Linear,1}(one_cst, u) + VectorDiv{XCALibre.Linear,2}(one_cst, v) == Source(0.0)) → BCs.p) → solvers.p

sys = MonolithicSystem([L_u(u), L_v(v), L_p(p)], [u, v, p])

@info "Assembling..."
A_csr, b_mono = assemble_monolithic_system(sys, (BCs.u, BCs.v, BCs.p), config)
A_julia = get_sparse_matrix(A_csr)
@info "Matrix Stats" size=size(A_julia) nnz=nnz(A_julia)

# Pin pressure
p_row = 2 * length(mesh_dev.cells) + 1
A_julia[p_row, :] .= 0.0
A_julia[p_row, p_row] = 1.0
b_mono[p_row] = 0.0

@info "Solving..."
x = A_julia \ b_mono
XCALibre.Solve.update_fields!(sys, x)
res = norm(A_julia*x - b_mono)

report_results("Stokes Straight Periodic (Core Topology)", res, u, p)
