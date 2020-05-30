module DilutedMLE

export diluted_mle

using LinearAlgebra
using QuantumOptics

"""
Calculate expectation value of `povm` given state `ρ`.

This is an optimised version of QuantumOptics.expect for dense operators, which
significantly increases `diluted_mle()` performance.
"""
function expect_inline(povm, ρ)
    sum = zero(povm[1, 1])
    n = size(povm)[1]
    #@assert size(povm) == (n, n)
    #@assert size(ρ) == (n, n)
    for i in 1:n
        for j in 1:n
            @inbounds sum += povm[i, j] * ρ[j, i]
        end
    end
    #@assert imag(sum) ≈ 0
    real(sum)
end


"""
Estimate density matrix from given POVM elements and associated weights using a
diluted fixed point iteration method.

Returns a tuple `(ρ, did_converge, number_of_iterations)`.
"""
function diluted_mle(povms::Vector{DenseOperator}, obs_weights::Vector{Float64},
	ρ0::DenseOperator; maxit=100000, tol=1e-9, ϵ=10
)
    # TODO: Clean up while keeping in-place properties.
    ρ = copy(ρ0)
    normalize!(ρ)
    r = copy(ρ)
    reye = Matrix{eltype(r.data)}(I, size(r.data))
    ρtmp = copy(ρ)
    ρprev = copy(ρ)
    for it in 1:maxit
        copyto!(ρprev.data, ρ.data)

        copyto!(r.data, reye)
        for (povm, obs_weight) in zip(povms, obs_weights)
            factor = ϵ * obs_weight / expect_inline(povm.data, ρ.data)
            for i in 1:length(r.data)
                @inbounds r.data[i] += factor * povm.data[i]
            end
        end
        r.data ./= 1 + ϵ
        mul!(ρtmp.data, ρ.data, r.data)
        mul!(ρ.data, r.data, ρtmp.data)
        normalize!(ρ)

        if norm((ρ - ρprev).data) < tol
            return ρ, true, it
        end
    end
    return ρ, false, maxit
end

end
