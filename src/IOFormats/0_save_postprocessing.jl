export save_postprocessing
export save_vtk

"""
    save_vtk(case_name, mesh_dev, BCs, fields...)

Write a named VTK snapshot to the directory given by the `XC_VTK_DIR` environment
variable. Does nothing when `XC_VTK_DIR` is unset or empty.

Each entry in `fields` is a `(name, field)` tuple accepted by `write_results`.

# Example
```julia
ENV["XC_VTK_DIR"] = "/tmp/vtk"
save_vtk("my_case", mesh_dev, BCs, ("U", U), ("p", p))
```
"""
function save_vtk(case_name::String, mesh_dev, BCs, fields...)
    vtk_dir = get(ENV, "XC_VTK_DIR", "")
    isempty(vtk_dir) && return
    mkpath(vtk_dir)
    fname = replace(case_name, r"[\s/\\]+" => "_") * ".vtk"
    old_dir = pwd()
    try
        cd(vtk_dir)
        meshData = initialise_writer(VTK(), mesh_dev)
        write_results(1, 1, mesh_dev, meshData, BCs, fields...)
        for ext in (".vtk", ".vtu")
            src = "iteration_1" * ext
            isfile(src) && mv(src, fname; force=true)
        end
    finally
        cd(old_dir)
    end
    println("  → wrote $(joinpath(vtk_dir, fname))")
end

function save_postprocessing(postprocess, iteration, time, mesh, meshData, BCs)
    postprocess === nothing && return nothing
    # suffix = "_" * string(postprocess.name)  
    suffix = "_postprocessed"
    
    args = build_args(postprocess)
    write_results(iteration, time, mesh, meshData, BCs, args...; suffix=suffix)
end
function save_postprocessing(postprocess, iteration, time, mesh, meshData::FOAMWriter, BCs)
    postprocess === nothing && return nothing
    args = build_args(postprocess)
    write_results(iteration, time, mesh, meshData, BCs, args...; suffix="")
end


function build_args(pp)
    pp === nothing && return ()
    if hasproperty(pp, :rs)
        return ((getproperty(pp, :name), getproperty(pp, :rs)),)
    elseif hasproperty(pp, :Q)
        return ((getproperty(pp, :name), getproperty(pp, :Q)),)
    elseif hasproperty(pp, :rms)
        return ((getproperty(pp, :name), getproperty(pp, :rms)),)
    elseif hasproperty(pp, :mean)
        return ((getproperty(pp, :name), getproperty(pp, :mean)),)
    else
        return ()
    end
end

function build_args(pp::Vector)
    vector_of_tuples = build_args.(pp)
    return Tuple(first(t) for t in vector_of_tuples if !isempty(t))
end