using XCALibre
using Test

# NeumannFunction with a constant gradient must reproduce Neumann with that value:
# same assembled Laplace system, and face value = cell value + delta * gradient.
mesh = UNV2D_mesh(joinpath(pkgdir(XCALibre, "examples/0_GRIDS"), "laplace_unit_3by3.unv"))
hardware = Hardware(backend=CPU(), workgroup=1024)
solvers = (T = SolverSetup(solver=Cg(), preconditioner=Jacobi(), convergence=1e-8, relax=1.0),)
schemes = (T = Schemes(laplacian=Linear),)
gamma = ConstantScalar(1.0)

function solve_laplace(right_bc)
    BCs = assign(region = mesh, (T = [
        Dirichlet(:left_wall, 50.0), right_bc,
        Dirichlet(:bottom_wall, 10.0), Zerogradient(:upper_wall)],))
    config = Configuration(solvers=solvers, schemes=schemes,
        runtime=Runtime(iterations=1, write_interval=1, time_step=1), hardware=hardware, boundaries=BCs)
    T = ScalarField(mesh)
    model = (-Laplacian{Linear}(gamma, T) == Source(ConstantScalar(0.0))) → ScalarEquation(T, config.boundaries.T)
    discretise!(model, T, config)
    apply_boundary_conditions!(model, config.boundaries.T, nothing, 0.0, config)
    A = Matrix(model.equation.A.parent); b = copy(model.equation.b)
    T.values .= A \ b
    Tf = FaceScalarField(mesh)
    interpolate!(Tf, T, config)
    correct_boundaries!(Tf, T, config.boundaries.T, 0.0, config)
    A, b, T, Tf
end

A1, b1, T1, Tf1 = solve_laplace(Neumann(:right_wall, 2.0))
A2, b2, T2, Tf2 = solve_laplace(NeumannFunction(:right_wall, (coords, time, i) -> 2.0))
@test A1 ≈ A2
@test b1 ≈ b2
@test T1.values ≈ T2.values

right = only(b for b in mesh.boundaries if b.name == :right_wall)
for fID in right.IDs_range
    cID = mesh.boundary_cellsID[fID]
    @test Tf2.values[fID] ≈ T2.values[cID] + 2.0 * mesh.faces[fID].delta
    @test Tf2.values[fID] ≈ Tf1.values[fID]
end
