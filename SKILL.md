---
name: xcalibre-kernels
description: Write idiomatic XCALibre.jl kernels and fused field loops. Use when adding, reviewing, or refactoring Julia code that uses xcal_foreach, KernelAbstractions.@kernel, XCALibre ScalarField/VectorField/Face*Field types, or GPU/CPU backend kernels in XCALibre.jl.
---

# XCALibre Kernel Style

Follow existing XCALibre patterns first. Keep kernels compact, typed where dispatch matters, and naming convection is action-oriented.

## Choose the Kernel Form

- Prefer `xcal_foreach(field, config) do i ... end` for local fused updates over fields, especially when combining several per-cell or per face calculations in models/solvers.
- Use `KernelAbstractions.@kernel` when the operation is reusable, needs explicit launch control, has multiple dispatch variants, touches sparse/raw arrays, or benefits from `@uniform`.
- Launch KA kernels from a public wrapper:

```julia
# API function
function name!(out, a::MyType, config)
    (; backend, workgroup) = config.hardware
    ndrange = length(out)
    kernel! = _name!(_setup(backend, workgroup, ndrange)...)
    kernel!(out, a)
end

@kernel function _name(out, a)
   i = @index(Global)

   @inbounds begin
        ... # variables will stay in bounds
    end
end
```

## Field and Vector Rules

- Pass XCALibre fields directly to kernels; do not pass `field.values` or component arrays unless the code truly needs raw storage.
- Read and write fields by direct indexing: `phi[i]`, `U[i]`, `phif[i] = ...`.
- Use direct vector/matrix algebra with StaticArrays: `a ⋅ b`, `norm(U[i])`, `tr(gradU[i])`, `g + g'`, `0.5*(g - g')`.
- In kernels, write vector dot products with the infix `⋅` operator (`\cdot<Tab>` in Julia editing) instead of `dot(a, b)`.
- Avoid expanding vector operations into `x/y/z` components. The exception is `Atomix.@atomic` on individual vector components; then pass/update each component separately.

## Naming and Dispatch

- Use `function name!(...)` for the public launcher and `@kernel function _name!(...)` for the implementation.
- Keep the kernel name stable across dispatch variants:

```julia
@kernel function _flux!(phif, psif, rhof::ScalarDensity)
    i = @index(Global)
    @uniform faces = phif.mesh.faces
    @inbounds begin
        Sf = faces[i].area * faces[i].normal
        phif[i] = (psif[i] ⋅ Sf) * rhof[i]
    end
end

@kernel function _flux!(phif, psif, ::NoDensity)
    i = @index(Global)
    @uniform faces = phif.mesh.faces
    @inbounds begin
        Sf = faces[i].area * faces[i].normal
        phif[i] = psif[i] ⋅ Sf
    end
end
```

- Prefer succinct action names. Use `update!`, `flux!`, `correct!`, etc. when surrounding types make the meaning clear; add words only when needed to disambiguate.

## Fused Loop Example

```julia
xcal_foreach(k, config) do i
    g = gradU[i]
    divU[i] = tr(g)
    S = 0.5*(g + g') - divU[i]/3*I
    Pk[i] = sum(g .* (2*S))
    normU[i] = norm(U[i])
end
```

## KA Kernel Example

```julia
function flux!(phif::FaceScalarField, psif::FaceVectorField, config)
    (; backend, workgroup) = config.hardware
    ndrange = length(phif)
    kernel! = _flux!(_setup(backend, workgroup, ndrange)...)
    kernel!(phif, psif)
end

@kernel function _flux!(phif, psif)
    i = @index(Global)
    @uniform faces = phif.mesh.faces
    @inbounds begin
        Sf = faces[i].area * faces[i].normal
        phif[i] = psif[i] ⋅ Sf
    end
end
```
