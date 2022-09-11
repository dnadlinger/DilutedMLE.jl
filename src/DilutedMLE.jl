module DilutedMLE

export diluted_mle

using LinearAlgebra
using QuantumOpticsBase

"""
Calculate expectation value of `povm` given state `ρ`.

This is an optimised version of QuantumOptics.expect for dense operators, which
significantly increases `diluted_mle()` performance.
"""
function expect_inline(povm, ρ)
    sum = zero(real(povm[1, 1]))
    n = size(povm)[1]
    #@assert size(povm) == (n, n)
    #@assert size(ρ) == (n, n)
    for i in 1:n
        for j in 1:n
            @inbounds sum += real(povm[i, j] * ρ[j, i])
        end
    end
    #@assert imag(sum) ≈ 0
    sum
end

function normalize_tr!(r)
    r .*= 1 / tr(r)
end

"""
Estimate density matrix from given POVM elements and associated weights using a
diluted fixed point iteration method.

Returns a tuple `(ρ, did_converge, number_of_iterations)`.
"""
function diluted_mle(povm_ops::Vector{T}, obs_weights::Vector{Float64},
       ρ0::T; maxit=100000, tol=1e-9, ϵ=10
) where T <: DenseOpType
    # TODO: Clean up while keeping in-place properties.
    @assert length(povm_ops) == length(obs_weights)
    povms = [p.data for p in povm_ops]
    ρ = ρ0.data
    normalize_tr!(ρ)
    r = copy(ρ)
    reye = Matrix{eltype(r)}(I, size(r))
    ρtmp = copy(r)
    ρprev = copy(r)
    for it in 1:maxit
        copyto!(ρprev, ρ)

        copyto!(r, reye)
        @inbounds for i in eachindex(povms)
            factor = ϵ * obs_weights[i] / expect_inline(povms[i], ρ)
            axpy!(factor, povms[i], r)
        end
        # Elided normalization of r, as we normalize_tr!(ρ) anyway.
        mul!(ρtmp, ρ, r)
        mul!(ρ, r, ρtmp)
        normalize_tr!(ρ)

        if norm(ρ - ρprev) < tol
            return DenseOperator(ρ0.basis_l, ρ0.basis_r, ρ), true, it
        end
    end
    return DenseOperator(ρ0.basis_l, ρ0.basis_r, ρ), false, maxit
end

end
