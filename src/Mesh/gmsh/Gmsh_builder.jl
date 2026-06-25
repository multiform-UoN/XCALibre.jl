module Gmsh

export Gmsh2D_mesh, Gmsh3D_mesh

import Gmsh: gmsh
using StaticArrays
using XCALibre.Mesh: Mesh2, Mesh3
using XCALibre.UNV2
using XCALibre.UNV3

# This module provides direct conversion from the Gmsh API to XCALibre mesh types.
# It leverages the connectivity building logic from the UNV modules to avoid duplication.

"""
    Gmsh2D_mesh(; scale=1, integer_type=Int64, float_type=Float64) -> Mesh2

Directly convert the active Gmsh model mesh into XCALibre.jl Mesh2 format.
Bypasses intermediate file formats (UNV/MSH).
"""
function Gmsh2D_mesh(; scale=1, integer_type=Int64, float_type=Float64)
    TI = integer_type
    TF = float_type

    println("Extracting Gmsh mesh data...")

    # 1. Get Nodes
    nodeTags, coords, _ = gmsh.model.mesh.getNodes()
    n_nodes = length(nodeTags)

    # Map Gmsh nodeTags to 1-based indices for XCALibre
    tag_to_idx = Dict{Int, TI}()
    points = Vector{UNV2.Point{TF}}(undef, n_nodes)
    for i in 1:n_nodes
        tag_to_idx[Int(nodeTags[i])] = TI(i)
        points[i] = UNV2.Point(SVector{3, TF}(coords[3i-2], coords[3i-1], coords[3i]))
    end

    if scale != one(typeof(scale))
        UNV2.scalePoints!(points, scale)
    end

    # 2. Get Elements
    elementTypes, elementTags, nodeTagsPerElement = gmsh.model.mesh.getElements(2) # 2D elements
    bc_elementTypes, bc_elementTags, bc_nodeTagsPerElement = gmsh.model.mesh.getElements(1)

    max_el_id = 0
    if !isempty(elementTags) max_el_id = max(max_el_id, maximum(vcat(elementTags...))) end
    if !isempty(bc_elementTags) max_el_id = max(max_el_id, maximum(vcat(bc_elementTags...))) end

    elements = Vector{UNV2.Element{TI}}(undef, max_el_id)
    for i in 1:max_el_id
        elements[i] = UNV2.Element(TI(i), TI(0), TI[])
    end

    # Process 1D Elements (Lines for boundaries)
    for (t, tags, nodes) in zip(bc_elementTypes, bc_elementTags, bc_nodeTagsPerElement)
        nodes_per_elem = Int(gmsh.model.mesh.getElementProperties(t)[4])
        for i in 1:length(tags)
            el_id = Int(tags[i])
            el_nodes = [tag_to_idx[Int(n)] for n in nodes[(i-1)*nodes_per_elem+1 : i*nodes_per_elem]]
            elements[el_id] = UNV2.Element(TI(el_id), TI(nodes_per_elem), el_nodes)
        end
    end

    # Process 2D Elements (Triangles/Quads for cells)
    for (t, tags, nodes) in zip(elementTypes, elementTags, nodeTagsPerElement)
        nodes_per_elem = Int(gmsh.model.mesh.getElementProperties(t)[4])
        for i in 1:length(tags)
            el_id = Int(tags[i])
            el_nodes = [tag_to_idx[Int(n)] for n in nodes[(i-1)*nodes_per_elem+1 : i*nodes_per_elem]]
            elements[el_id] = UNV2.Element(TI(el_id), TI(nodes_per_elem), el_nodes)
        end
    end

    # 3. Process Physical Groups (Boundaries)
    physGroups = gmsh.model.getPhysicalGroups(1) # 1D boundaries
    boundaryElements = UNV2.BoundaryLoader{TI}[]
    for (dim, tag) in physGroups
        name = gmsh.model.getPhysicalName(dim, tag)
        if isempty(name) name = "group_$tag" end

        entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
        group_elements = TI[]
        for ent in entities
            _, tags, _ = gmsh.model.mesh.getElements(dim, ent)
            append!(group_elements, TI.(vcat(tags...)))
        end

        loader = UNV2.BoundaryLoader(name, TI(tag), group_elements)
        push!(boundaryElements, loader)
    end

    println("Generating mesh...")
    bfaces = UNV2.total_boundary_faces(boundaryElements)
    cells, faces, nodes, boundaries = UNV2.generate(points, elements, boundaryElements, bfaces)

    println("Building connectivity...")
    UNV2.connect!(cells, faces, nodes, boundaries, bfaces)

    # Use the intermediate UNV2.Mesh2 type
    mesh_intermediate = UNV2.Mesh2(cells, faces, boundaries, nodes)
    UNV2.process_geometry!(mesh_intermediate)

    # Convert to the final XCALibre.Mesh.Mesh2 type
    mesh_final = UNV2.update_mesh_format(mesh_intermediate, TI, TF)
    println("Mesh ready!")
    return mesh_final
end

"""
    Gmsh3D_mesh(; scale=1, integer_type=Int64, float_type=Float64) -> Mesh3

Directly convert the active Gmsh model mesh into XCALibre.jl Mesh3 format.
Bypasses intermediate file formats (UNV/MSH).
"""
function Gmsh3D_mesh(; scale=1, integer_type=Int64, float_type=Float64)
    I = integer_type
    F = float_type

    println("Extracting 3D Gmsh mesh data...")

    # 1. Get Nodes
    nodeTags, coords, _ = gmsh.model.mesh.getNodes()
    n_nodes = length(nodeTags)

    # Map Gmsh nodeTags to 1-based indices
    tag_to_idx = Dict{Int, I}()
    points = Vector{UNV3.Point{F, SVector{3, F}}}(undef, n_nodes)
    for i in 1:n_nodes
        tag_to_idx[Int(nodeTags[i])] = I(i)
        points[i] = UNV3.Point(SVector{3, F}(coords[3i-2] * scale, coords[3i-1] * scale, coords[3i] * scale))
    end

    # 2. Get 2D Faces (for boundaries)
    f_types, f_tags, f_nodes = gmsh.model.mesh.getElements(2)
    efaces = UNV3.Face{I, Vector{I}}[]
    face_tag_to_idx = Dict{Int, I}()

    f_counter = 0
    for (t, tags, nodes) in zip(f_types, f_tags, f_nodes)
        npe = Int(gmsh.model.mesh.getElementProperties(t)[4])
        for i in 1:length(tags)
            f_counter += 1
            el_nodes = [tag_to_idx[Int(n)] for n in nodes[(i-1)*npe+1 : i*npe]]
            push!(efaces, UNV3.Face(I(f_counter), I(npe), el_nodes))
            face_tag_to_idx[Int(tags[i])] = I(f_counter)
        end
    end

    # 3. Get 3D Cells
    c_types, c_tags, c_nodes = gmsh.model.mesh.getElements(3)
    cells_UNV = UNV3.Cell_UNV{I, Vector{I}}[]

    c_counter = 0
    for (t, tags, nodes) in zip(c_types, c_tags, c_nodes)
        npe = Int(gmsh.model.mesh.getElementProperties(t)[4])
        for i in 1:length(tags)
            c_counter += 1
            el_nodes = [tag_to_idx[Int(n)] for n in nodes[(i-1)*npe+1 : i*npe]]
            push!(cells_UNV, UNV3.Cell_UNV(I(c_counter), I(npe), el_nodes))
        end
    end

    # 4. Process Physical Groups (Boundaries)
    physGroups = gmsh.model.getPhysicalGroups(2) # 2D boundaries in 3D mesh
    boundaryElements = UNV3.BoundaryElement{String, I, Vector{I}}[]
    for (dim, tag) in physGroups
        name = gmsh.model.getPhysicalName(dim, tag)
        if isempty(name) name = "group_$tag" end

        entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
        group_faces = I[]
        for ent in entities
            _, tags, _ = gmsh.model.mesh.getElements(dim, ent)
            for t_list in tags
                for f_tag in t_list
                    if haskey(face_tag_to_idx, Int(f_tag))
                        push!(group_faces, face_tag_to_idx[Int(f_tag)])
                    end
                end
            end
        end

        loader = UNV3.BoundaryElement(name, I(tag), group_faces)
        push!(boundaryElements, loader)
    end

    @info "Generating mesh connectivity and geometry from Gmsh data..."
    mesh = UNV3._build_UNV3D_mesh_core(points, efaces, cells_UNV, boundaryElements, I, F)

    @info "Mesh constructed: $(length(mesh.nodes)) nodes | $(length(mesh.faces)) faces | $(length(mesh.cells)) cells."
    return mesh
end

end # module
