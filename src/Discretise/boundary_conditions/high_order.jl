@define_boundary Zerogradient Biharmonic{Linear} begin
    0.0, 0.0
end

@define_boundary Dirichlet Biharmonic{Linear} begin
    J = term.flux[fID]
    flux = J * face.area / face.delta
    ap = term.sign * (-flux)
    ap, ap * bc.value
end

@define_boundary Robin Biharmonic{Linear} begin
    J = term.flux[fID]
    (; a, b, value) = bc.value
    denom = a*face.delta + b
    coeff = J*face.area/denom
    ap = term.sign*(-coeff*a)
    bp = term.sign*(-coeff*value)
    ap, bp
end
