# van Genuchten (1980) retention and Mualem (1976) conductivity with analytic
# derivatives, and the Kirchhoff potential Phi(h) = ∫ kr dh used to write the
# Richards diffusion term as a Laplacian of a function of h.
#
#   Se(h) = [1 + (α|h|)^n]^(-m),  m = 1 - 1/n,  Se = 1 for h ≥ 0
#   θ(h)  = θr + (θs - θr) Se,     C(h) = dθ/dh
#   kr(h) = Se^l [1 - (1 - Se^(1/m))^m]^2

struct VanGenuchten{F}
    θr::F
    θs::F
    α::F
    n::F
    l::F
end
VanGenuchten(; θr, θs, α, n, l=0.5) = VanGenuchten(promote(θr, θs, α, n, l)...)
Base.broadcastable(p::VanGenuchten) = Ref(p)

_m(p::VanGenuchten) = 1 - 1/p.n

Se(p::VanGenuchten, h) = h >= 0 ? one(h) : (1 + (p.α*abs(h))^p.n)^(-_m(p))

# dSe/dh, positive for h < 0
function dSe(p::VanGenuchten, h)
    h >= 0 && return zero(h)
    m = _m(p); ah = p.α*abs(h)
    return p.α*m*p.n*ah^(p.n - 1)*(1 + ah^p.n)^(-m - 1)
end

θ(p::VanGenuchten, h) = p.θr + (p.θs - p.θr)*Se(p, h)
C(p::VanGenuchten, h) = (p.θs - p.θr)*dSe(p, h)

function kr(p::VanGenuchten, h)
    h >= 0 && return one(h)
    m = _m(p); se = Se(p, h)
    f = 1 - (1 - se^(1/m))^m
    return se^p.l*f^2
end

function dkr(p::VanGenuchten, h)
    h >= 0 && return zero(h)
    m = _m(p); se = Se(p, h)
    g = 1 - se^(1/m)
    f = 1 - g^m
    df = g^(m - 1)*se^(1/m - 1)          # df/dSe
    dkr_dSe = p.l*se^(p.l - 1)*f^2 + 2*se^p.l*f*df
    return dkr_dSe*dSe(p, h)
end

# ∫_a^b kr dh by composite 8-point Gauss-Legendre: in s = ln(-h) for h < -hsmall
# (kr is smooth in ln|h| over many decades), directly on [-hsmall, 0], exactly
# for h > 0. The same rule is used by GMRTFoam's richardsHeadFoam.
const _GLx = (-0.9602898564975363, -0.7966664774136267, -0.5255324099163290,
    -0.1834346424956498, 0.1834346424956498, 0.5255324099163290,
    0.7966664774136267, 0.9602898564975363)
const _GLw = (0.1012285362903763, 0.2223810344533745, 0.3137066458778873,
    0.3626837833783620, 0.3626837833783620, 0.3137066458778873,
    0.2223810344533745, 0.1012285362903763)

function kr_integral(p::VanGenuchten, a, b; hsmall=1e-3, dsmax=0.25)
    b < a && return -kr_integral(p, b, a; hsmall=hsmall, dsmax=dsmax)
    I = zero(promote_type(typeof(a), typeof(b)))
    b > 0 && (I += b - max(a, zero(a)))
    lo, hi = max(a, -hsmall), min(b, zero(b))
    if hi > lo
        I += (hi - lo)/2*sum(w*kr(p, (lo + hi)/2 + (hi - lo)/2*x) for (x, w) in zip(_GLx, _GLw))
    end
    lo, hi = a, min(b, -hsmall)
    if hi > lo
        s0, s1 = log(-hi), log(-lo)
        nsub = max(1, ceil(Int, (s1 - s0)/dsmax))
        ds = (s1 - s0)/nsub
        for j in 0:nsub-1
            sa = s0 + j*ds
            I += ds/2*sum(w*kr(p, -exp(sa + ds/2*(1 + x)))*exp(sa + ds/2*(1 + x))
                for (x, w) in zip(_GLx, _GLw))
        end
    end
    return I
end

# Kirchhoff potential relative to href; only differences of Phi enter the fluxes
Φ(p::VanGenuchten, h; href=-1.0) = kr_integral(p, href, h)
