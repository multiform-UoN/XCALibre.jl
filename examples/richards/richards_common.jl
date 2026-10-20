# Richards' equation in mixed form with Newton's method on XCALibre's PDE layer,
# plus a passive tracer driven by the discrete Darcy flux.
#
#   ∂θ(h)/∂t - ∇·(Ks ∇Φ(h)) + ∇·(F_g kr(h)) = 0,   Φ' = kr (Kirchhoff),
#   F_g = Ks ĝ·S_f (gravity, upwinded),
#
# so the diffusive flux between two cells is Ks (Φ_N - Φ_P)/δ = Ks mean(kr) (h_N - h_P)/δ,
# the same face conductivity as the `kirchhoff` option of GMRTFoam's richardsHeadFoam.
# Storage is a nonlinear implicit source θ(h)/Δt with the old content θⁿ/Δt on the
# right-hand side, which keeps the mixed form mass conservative. Every nonlinear
# term carries its analytic derivative (NonlinearMap(f, df)), so linearize_physics
# assembles the exact Jacobian.

using XCALibre
using Accessors
using LinearAlgebra
using Printf
using StaticArrays

include("vanGenuchten.jl")

# gravity face flux F_g = Ks ĝ·n A (water moves along ĝ under gravity)
function gravity_flux(mesh, Ks, ĝ)
    Fg = FaceScalarField(mesh)
    for (i, f) in enumerate(mesh.faces)
        Fg.values[i] = Ks*(ĝ ⋅ f.normal)*f.area
    end
    Fg
end

"""
    richards_operator(soil, Ks, Fg, dt, θold, BCs, setup)

Field-free Richards operator; bind it to the head field with `L(h)`. `dt` is a `Ref`
so that the time step can change without rebuilding the operator; `θold` holds θⁿ/Δt.
"""
function richards_operator(soil, Ks, Fg, dt, rhs, BCs, setup)
    storage = NonLinearSi(h -> θ(soil, h)/dt[], h -> C(soil, h)/dt[])
    kirchhoff = NonlinearMap(h -> Φ(soil, h), h -> kr(soil, h))
    conductivity = NonlinearMap(h -> kr(soil, h), h -> dkr(soil, h))
    ((storage
      - Laplacian{Linear}(ConstantScalar(Ks), kirchhoff)
      + Divergence{Upwind}(Fg, conductivity)
      == Source(rhs)) → BCs) → setup
end

"""
    newton_step!(eqn, φ, BCs, config; htol, maxiter, damping_min, chop=false)

One time step of Newton's method for the field φ (the head h for Richards): exact Jacobian from linearize_physics, a linear
solve per iteration, and a backtracking line search on the residual norm. A step is
accepted at the first length ω = 1, 1/2, ... that decreases |R| (Armijo); if none down
to `damping_min` does, the time step fails (the caller retries with a smaller Δt).
Converged when a full Newton step has max|Δh| < htol. With `chop` (for the head), a cell whose head would cross
h = 0 stops at 0, as in richardsHeadFoam. Returns (converged, iterations, residuals).
"""
function newton_step!(eqn, h, BCs, config; htol=1e-7, maxiter=20, damping_min=1/16, chop=false)
    # h is any scalar field: the head for Richards, a concentration for transport
    res = Float64[]
    for it in 1:maxiter
        bcs, lin, _ = linearize_physics(BCs, eqn; ad_backend=:forwarddiff)
        lin = @set lin.equation.BCs = bcs
        r0 = norm(residual(lin, config))
        push!(res, r0)
        h0 = copy(h.values)
        solve_equation!(lin, config)
        δ = h.values .- h0
        all(isfinite, δ) || (h.values .= h0; return (false, it, res))
        # chop at saturation (Forsyth et al. 1995): a cell whose value would cross 0 stops
        # there, so an update never jumps over the kink of C and kr at h = 0 (Richards only)
        if chop
            for i in eachindex(δ)
                (h0[i] < 0 && h0[i] + δ[i] > 0 || h0[i] > 0 && h0[i] + δ[i] < 0) && (δ[i] = -h0[i])
            end
        end
        # a full step below the tolerance converges; its residual is at round-off
        maximum(abs, δ) < htol && return (true, it, res)
        ω = 1.0
        accepted = false
        while ω >= damping_min
            @. h.values = h0 + ω*δ
            if norm(residual(eqn, config)) <= (1 - 1e-4ω)*r0     # Armijo
                accepted = true
                break
            end
            ω /= 2
        end
        accepted || (h.values .= h0; return (false, it, res))
    end
    return (false, maxiter, res)
end

# Darcy face flux from h, the same discrete fluxes the Richards operator assembles;
# `bvals` maps Dirichlet boundary faces to their head, other boundary faces are zero-gradient
function darcy_flux!(F, mesh, soil, Ks, Fg, h, bvals)
    gD = mesh.face_gDiff
    for (i, f) in enumerate(mesh.faces)
        P, N = f.ownerCells
        if P != N
            hP, hN = h.values[P], h.values[N]
            up = Fg.values[i] > 0 ? hP : hN
            F.values[i] = -Ks*gD[i]*kr_integral(soil, hP, hN) + Fg.values[i]*kr(soil, up)
        else
            hb = get(bvals, i, nothing)
            if hb === nothing
                # zero-gradient (free drainage) or empty: gravity flux at the cell's kr
                F.values[i] = Fg.values[i]*kr(soil, h.values[P])
            else
                hP = h.values[P]
                up = Fg.values[i] > 0 ? hP : hb
                F.values[i] = -Ks*gD[i]*kr_integral(soil, hP, hb) + Fg.values[i]*kr(soil, up)
            end
        end
    end
    F
end

# face ids of named Dirichlet patches => boundary head
function dirichlet_faces(mesh, values::Dict{Symbol,Float64})
    d = Dict{Int,Float64}()
    for b in mesh.boundaries
        haskey(values, b.name) || continue
        for i in b.IDs_range
            d[i] = values[b.name]
        end
    end
    d
end

# cell Darcy velocity by Gauss reconstruction, U_P = Σ_f F_f (x_f - x_P)/V_P, and its magnitude
function cell_speed(mesh, F)
    U = [zero(mesh.cells[1].centre) for _ in mesh.cells]
    for (i, f) in enumerate(mesh.faces)
        P, N = f.ownerCells
        U[P] += F.values[i]*(f.centre - mesh.cells[P].centre)
        P != N && (U[N] -= F.values[i]*(f.centre - mesh.cells[N].centre))
    end
    [norm(U[k])/mesh.cells[k].volume for k in eachindex(U)]
end
