using ForwardDiff
using Enzyme
using Accessors

export linearize_physics, linearize_bcs

# DSL constructor overloads: three-argument form returns NonlinearOperator wrapper
# (extending constructors defined in ModelFramework)
ModelFramework.Laplacian{T}(flux, func::Function, phi) where T =
    NonlinearOperator(Operator(flux, phi, 1, Laplacian{T}()), func)

ModelFramework.Divergence{T}(flux, func::Function, phi) where T =
    NonlinearOperator(Operator(flux, phi, 1, Divergence{T}()), func)

# Two-argument constructor: NonLinearSi(func, phi) → Operator with NonLinearSi type tag
ModelFramework.NonLinearSi(func::Function, phi) =
    Operator(nothing, phi, 1, NonLinearSi(func))

# ---------------------------------------------------------------------------
# linearize_bcs: replace NonLinearRobin BCs with their Newton-linearised Robin
# ---------------------------------------------------------------------------
function linearize_bcs(BCs, phi::ScalarField)
    return Discretise.update_nonlinear_robin(BCs, phi)
end

# ---------------------------------------------------------------------------
# linearize_physics: convert all NonlinearOperator terms into standard Operators
# using automatic differentiation (ForwardDiff default, Enzyme optional)
# ---------------------------------------------------------------------------
"""
    linearize_physics(BCs, model_eqn; susp=true, ad_backend=:forwarddiff)

Pre-pass Newton linearisation. Detects `NonlinearOperator` and `NonLinearSi` terms, 
differentiates the stored functions at current field values, scales the operator 
flux, and returns a plain `ModelEquation` with standard `Operator` and `Source` terms.
"""
function linearize_physics(BCs, model_eqn::ModelEquation; susp=true, ad_backend=:forwarddiff)
    phi = get_phi(model_eqn)
    vals = phi.values
    mesh = phi.mesh
    backend = _get_backend(mesh)

    # 1. Linearise BCs (NonLinearRobin → Robin)
    new_bcs = linearize_bcs(BCs, phi)

    # 2. Linearise model terms
    # We collect new terms and additional sources
    extra_sources = []
    
    new_terms = map(model_eqn.model.terms) do term
        # Case A: Non-linear Differential Operators (Wrapped in NonlinearOperator)
        # Must check this BEFORE NonLinearSi since NonlinearOperator has no .type field
        if term isa NonlinearOperator
            func = term.func
            inner_op = term.op

            # Compute df/dphi at each cell
            df = ScalarField(mesh)
            s_exp_raw = ScalarField(mesh)
            for i in eachindex(vals)
                v0 = vals[i]
                if ad_backend === :forwarddiff
                    df.values[i] = ForwardDiff.derivative(func, v0)
                    r0 = func(v0)
                else # :enzyme
                    r0 = func(v0)
                    df.values[i] = Enzyme.autodiff(Enzyme.Reverse, func, Enzyme.Active, Enzyme.Active(v0))[1][1]
                end
                s_exp_raw.values[i] = r0 - df.values[i] * v0
            end

            # Scale original flux by derivative (interpolate to faces)
            df_f = FaceScalarField(mesh)
            for fID in eachindex(df_f.values)
                oc = mesh.faces[fID].ownerCells
                df_f.values[fID] = length(oc) > 1 ?
                    0.5*(df.values[oc[1]] + df.values[oc[2]]) :
                    df.values[oc[1]]
            end

            # Create linearized flux
            lin_flux = FaceScalarField(mesh)
            if typeof(inner_op.flux) <: FaceScalarField
                lin_flux.values .= inner_op.flux.values .* df_f.values
            elseif typeof(inner_op.flux) <: ScalarField
                for fID in eachindex(lin_flux.values)
                    oc = mesh.faces[fID].ownerCells
                    f_val = length(oc) > 1 ? 0.5*(inner_op.flux.values[oc[1]] + inner_op.flux.values[oc[2]]) : inner_op.flux.values[oc[1]]
                    lin_flux.values[fID] = f_val * df_f.values[fID]
                end
            else # ConstantScalar
                lin_flux.values .= inner_op.flux.values .* df_f.values
            end

            push!(extra_sources, Source(s_exp_raw))
            return Operator(lin_flux, inner_op.phi, inner_op.sign, inner_op.type)

        # Case B: Standard Non-linear Implicit Source (NonLinearSi)
        elseif hasproperty(term, :type) && typeof(term.type) <: NonLinearSi
            func = term.type.func
            k_imp = ScalarField(mesh)
            s_exp = ScalarField(mesh)
            
            for i in eachindex(vals)
                v0 = vals[i]
                if ad_backend === :forwarddiff
                    dv = ForwardDiff.derivative(func, v0)
                    r0 = func(v0)
                else  # :enzyme
                    r0 = func(v0)
                    dv = Enzyme.autodiff(Enzyme.Reverse, func, Enzyme.Active, Enzyme.Active(v0))[1][1]
                end
                
                # Equation: ... + R(phi) = ...
                # Newton: R(phi) ≈ dv * phi + (r0 - dv * v0)
                # SuSp logic: only keep dv implicit if it helps diagonal dominance
                if !susp || dv > 0
                    k_imp.values[i] = dv
                    s_exp.values[i] -= (r0 - dv * v0)
                else
                    k_imp.values[i] = 0.0
                    s_exp.values[i] -= r0
                end
            end
            push!(extra_sources, Source(s_exp))
            return Si(k_imp, phi)

        else
            return term
        end
    end

    # 3. Rebuild Model and Equation
    TN = typeof(model_eqn.model).parameters[1]
    SN = typeof(model_eqn.model).parameters[2]
    new_sources = (model_eqn.model.sources..., extra_sources...)
    new_model = Model{TN,SN}(Tuple(new_terms), new_sources)
    
    return new_bcs, @set model_eqn.model = new_model
end
