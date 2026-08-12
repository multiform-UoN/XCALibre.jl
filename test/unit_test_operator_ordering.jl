using XCALibre
using Test

# Regression test for the PDE operator DSL's "time term must come first"
# invariant (see TODO.md, "Hidden operator-ordering rules").
# The legacy generated discretisation code relies on `terms[1]` being the
# time term when a transient term is present. Composition of raw
# TemplateTerm objects (Laplacian, Divergence, Time, ...) must place the time
# term first regardless of the order the user writes the algebraic
# expression in.
@testset "PDE operator: time-term ordering invariance" begin
    mu = ConstantScalar(1.0)
    L = Laplacian{Linear}(mu)
    Tterm = Time{Euler}(ConstantScalar(1.0))

    # raw TemplateTerm + TemplateTerm, both orders
    L_plus_T = L + Tterm
    T_plus_L = Tterm + L
    @test first(L_plus_T.templates).type isa Time
    @test first(T_plus_L.templates).type isa Time
    @test [t.type for t in L_plus_T.templates] == [t.type for t in T_plus_L.templates]

    # PDEOperator + Time / Time + PDEOperator, both orders
    L_pde = (-L == Source(ConstantScalar(0.0)))
    pde_plus_T = L_pde + Tterm
    T_plus_pde = Tterm + L_pde
    @test first(pde_plus_T.templates).type isa Time
    @test first(T_plus_pde.templates).type isa Time
    @test [t.type for t in pde_plus_T.templates] == [t.type for t in T_plus_pde.templates]

    # Chained composition with a second non-time term, still normalises
    D = Divergence{Linear}(ConstantScalar(1.0))
    chained = (L + D) + Tterm
    @test first(chained.templates).type isa Time
    @test length(chained.templates) == 3
end
