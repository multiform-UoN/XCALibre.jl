using XCALibre
using Test

# The PDE layer wraps operators (NonlinearOperator, AffineOperator). Code that rebuilds a term,
# e.g. into its kernel form or onto another field, must keep the wrappers: an upstream rewrite of
# the form `Operator(t.flux, t.phi, t.sign, ...)` drops them without an error and turns a nonlinear
# term into a linear one. These tests pin the rebuild sites to `map_operator`.

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "laplace_unit_3by3.unv"))
mesh_dev = adapt(CPU(), mesh)

nl_map = NonlinearMap(x -> x^2, x -> 2x)

@testset "map_operator keeps PDE-layer wrappers" begin
    C = ScalarField(mesh_dev); initialise!(C, 0.5)
    D = ScalarField(mesh_dev); initialise!(D, 0.1)
    nl = Laplacian{Linear}(ConstantScalar(1.0), nl_map, C)
    @test nl isa NonlinearOperator

    rebound = map_operator(op -> Operator(op.flux, D, op.sign, op.type), nl)
    @test rebound isa NonlinearOperator
    @test rebound.map === nl.map
    @test rebound.op.phi === D

    jac = ScalarField(mesh_dev); off = ScalarField(mesh_dev)
    aff = AffineOperator(nl.op, jac, off, C, nl_map)
    seen = Any[]
    rebound = map_operator(op -> Operator(op.flux, D, op.sign, op.type), aff;
        fields=f -> (push!(seen, f); f))
    @test rebound isa AffineOperator
    @test rebound.op.phi === D
    @test rebound.reference === C
    @test length(seen) == 3   # jacobian, offset, reference

    # kernel and typed forms (the upstream kernel-argument path)
    m = (; face_gDiff=mesh_dev.face_gDiff)
    @test XCALibre.Discretise._kernel_term(nl, m) isa NonlinearOperator
    @test XCALibre.Discretise._kernel_term(aff, m) isa AffineOperator
    @test XCALibre.Discretise._typed_term(nl) isa NonlinearOperator
    @test XCALibre.Discretise._typed_term(aff) isa AffineOperator
end

@testset "decompose keeps nonlinear vector terms" begin
    U = VectorField(mesh_dev); initialise!(U, [0.5, 0.25, 0.0])
    BCs = assign(
        region=mesh_dev,
        (U = [Dirichlet(:left_wall, [0.0, 0.0, 0.0]), Dirichlet(:right_wall, [1.0, 0.0, 0.0]),
              Zerogradient(:upper_wall), Zerogradient(:bottom_wall)],)
    )
    L = (-Laplacian{Linear}(ConstantScalar(1.0), nl_map) == Source(0.0)) → BCs.U
    eqns = decompose(L(U))
    @test length(eqns) >= 2
    for (i, eqn) in enumerate(eqns)
        term = eqn.model.terms[1]
        @test term isa NonlinearOperator
        @test term.map === nl_map
        @test term.op.sign == -1
        comp = i == 1 ? U.x : U.y
        @test term.op.phi.values === comp.values
    end
end

# Tripwire for upstream merges: rebuilding an Operator from another term's fields is allowed only
# where the argument is a bare Operator (scheme fallbacks, the affine BC inner operator). A new
# site, or more of them, has to use `map_operator` or be added here after checking it.
@testset "no new wrapper-stripping Operator rebuilds" begin
    allowed = Dict(
        "Discretise/Discretise_1_schemes.jl" => 2,            # _upwind, _lust (term::Operator)
        "Discretise/Discretise_3_boundary_conditions.jl" => 1, # LUSTDC boundary (term::Operator)
        "Discretise/boundary_conditions/affine.jl" => 1,       # inner operator of an AffineOperator
    )
    pattern = r"\bOperator\(\s*\w+(\.op)?\.flux\s*,"
    src = pkgdir(XCALibre, "src")
    found = Dict{String,Int}()
    for (root, _, files) in walkdir(src), file in files
        endswith(file, ".jl") || continue
        path = joinpath(root, file)
        n = count(eachline(path)) do line
            occursin(pattern, line) && !occursin(r"^\s*#", line) && !occursin("map_operator", line)
        end
        n > 0 && (found[relpath(path, src)] = n)
    end
    for (file, n) in found
        n <= get(allowed, file, 0) ||
            @warn "Operator rebuilt from term fields in $file ($n sites): use map_operator"
        @test n <= get(allowed, file, 0)
    end
end
