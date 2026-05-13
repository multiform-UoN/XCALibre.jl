using XCALibre
using SparseArrays

import XCALibre.Solve: assemble_monolithic_system

function tutorial_sparse_matrix(A_csr)
    rows = Vector{Int}(undef, length(A_csr.nzval))
    for r in 1:(length(A_csr.rowptr) - 1)
        for nzi in A_csr.rowptr[r]:(A_csr.rowptr[r + 1] - 1)
            rows[nzi] = r
        end
    end
    return sparse(rows, A_csr.colval, A_csr.nzval, size(A_csr)...)
end

function tutorial_straight_mesh()
    grids_dir = pkgdir(XCALibre, "examples", "0_GRIDS")
    mesh = UNV2D_mesh(joinpath(grids_dir, "quad40.unv"), scale=0.025)
    return mesh, adapt(CPU(), mesh)
end
