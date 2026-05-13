using XCALibre
using LinearAlgebra
using Printf
using SparseArrays
using StaticArrays

include("benchmark_utils.jl")

mesh_cpu, _ = get_straight_mesh()

function make_periodic_topology(mesh::Mesh2, patch1::Symbol, patch2::Symbol, translation::AbstractVector)
    b_idx1 = findfirst(x -> x.name == patch1, mesh.boundaries)
    b_idx2 = findfirst(x -> x.name == patch2, mesh.boundaries)
    
    faces1_range = mesh.boundaries[b_idx1].IDs_range
    faces2_range = mesh.boundaries[b_idx2].IDs_range

    centers1 = [mesh.faces[id].centre for id in faces1_range]
    centers2 = [mesh.faces[id].centre for id in faces2_range]

    tol = 1e-5
    face_map = Dict{Int, Int}()
    for (i, c1) in enumerate(centers1)
        expected_c2 = c1 + translation
        for (j, c2) in enumerate(centers2)
            if norm(expected_c2 - c2) < tol
                face_map[faces1_range[i]] = faces2_range[j]
                break
            end
        end
    end
    
    @info "Found $(length(face_map)) periodic face pairs between $patch1 and $patch2"

    new_faces = copy(mesh.faces)
    new_cell_faces = Int[]
    new_cell_neighbours = Int[]
    new_cell_nsign = Int[]
    new_cells = empty(mesh.cells)
    
    cell_new_faces = Dict{Int, Vector{Tuple{Int, Int, Int}}}()
    for i in 1:length(mesh.cells)
        cell_new_faces[i] = Tuple{Int, Int, Int}[]
    end
    
    current_fID = length(mesh.faces) + 1
    
    for (f1, f2) in face_map
        face1 = mesh.faces[f1]
        face2 = mesh.faces[f2]
        
        owner1 = face1.ownerCells[1]
        owner2 = face2.ownerCells[1]
        
        C1 = mesh.cells[owner1].centre
        C2_eff = mesh.cells[owner2].centre - translation
        
        d = C2_eff - C1
        delta = norm(d)
        e = d / delta
        
        new_face = Face2D(face1.nodes_range, SVector(owner1, owner2), face1.centre, face1.normal, e, face1.area, delta, 0.5)
        push!(new_faces, new_face)
        
        push!(cell_new_faces[owner1], (current_fID, owner2, 1))
        push!(cell_new_faces[owner2], (current_fID, owner1, -1))
        current_fID += 1
    end
    
    for cID in 1:length(mesh.cells)
        cell = mesh.cells[cID]
        start_idx = length(new_cell_faces) + 1
        
        for fi in cell.faces_range
            fID = mesh.cell_faces[fi]
            if !haskey(face_map, fID) && !(fID in values(face_map))
                push!(new_cell_faces, mesh.cell_faces[fi])
                push!(new_cell_neighbours, mesh.cell_neighbours[fi])
                push!(new_cell_nsign, mesh.cell_nsign[fi])
            end
        end
        
        for (fID, nID, nsign) in cell_new_faces[cID]
            push!(new_cell_faces, fID)
            push!(new_cell_neighbours, nID)
            push!(new_cell_nsign, nsign)
        end
        
        stop_idx = length(new_cell_faces)
        new_faces_range = start_idx:stop_idx
        push!(new_cells, Cell(cell.centre, cell.volume, cell.nodes_range, new_faces_range))
    end
    
    new_mesh = Mesh2(new_cells, mesh.cell_nodes, new_cell_faces, new_cell_neighbours, new_cell_nsign, new_faces, mesh.face_nodes, mesh.boundaries, mesh.nodes, mesh.node_cells, mesh.get_float, mesh.get_int, mesh.boundary_cellsID)
    return new_mesh
end

mesh_per = make_periodic_topology(mesh_cpu, :inlet, :outlet, [25.0, 0.0, 0.0])
mesh_dev = adapt(CPU(), mesh_per)

u = ScalarField(mesh_dev); initialise!(u, 0.0)
v = ScalarField(mesh_dev); initialise!(v, 0.0)
p = ScalarField(mesh_dev); initialise!(p, 0.0)

# Provide Empty BC for inlet and outlet, so standard BC loop ignores them.
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

# Pin pressure
p_row = 2 * length(mesh_dev.cells) + 1
A_julia[p_row, :] .= 0.0
A_julia[p_row, p_row] = 1.0
b_mono[p_row] = 0.0

@info "Solving..."
x = A_julia \ b_mono
XCALibre.Solve.update_fields!(sys, x)

report_results("Topological Periodic Stokes", norm(A_julia*x - b_mono), u, p)