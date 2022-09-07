# Generates random states, simulates tomography experiments, and verifies the
# diluted MLE procedure converges to something close to the ground truth.
#
# FIXME: This currently outputs plots. Instead, we should compare the distribution of
# MLE result/ground truth distances to some bounds for automated regression testing.

using DilutedMLE, Distributions, LinearAlgebra, QuantumOpticsBase, RandomQuantum

trnorm(ρ) = norm(svdvals(ρ.data), 1)

function random_mle(projectors, asymptotic=true, shot_count=2^16 * length(projectors))
    pure_frac = rand()
    basis = projectors[1].basis_l

    a = projector(Ket(basis, rand(FubiniStudyPureState(4))))
    b = DenseOperator(basis, rand(HilbertSchmidtMixedState(4)))

    ρ = pure_frac * a + (1 - pure_frac) * b

    means = [real(expect(p, ρ)) for p in projectors]

    weights = asymptotic ? means : rand(Multinomial(shot_count, means / sum(means)))
    ρest, converged, iters_taken = diluted_mle(projectors, Float64.(weights), identityoperator(DenseOpType, basis))

    return trnorm(ρ - ρest), converged, iters_taken, ρest
end

function test_mle(asymptotic, iter_count=1_000)
    trdists = Array{Float64}(undef, iter_count)
    iters_takens = Array{Int64}(undef, iter_count)

    for i = 1:iter_count
        b = SpinBasis(1 // 2)
        d = spindown(b)
        u = spinup(b)
        states = [d, u, normalize(d + u), normalize(d - u), normalize(d + 1im * u), normalize(d - 1im * u)]
        projectors = [identityoperator(DenseOpType, b); [projector(s) for s in states]]
        two_qubit_projectors = reduce(vcat, [dense(l ⊗ r) for l in projectors] for r in projectors)

        trdist, converged, iters_taken, ρest = random_mle(two_qubit_projectors, asymptotic)
        trdists[i] = trdist
        iters_takens[i] = iters_taken
    end

    trdists, iters_takens
end


# Run tests, both with ideal obervable values and with experimental sampling noise.
using PyPlot
trdists_as, iters_as = test_mle(true)
trdists_mn, iters_mn = test_mle(false)

figure()
ploth(data) = plt."hist"(data, "auto")
ploth(trdists_as)
figure()
ploth(trdists_mn)

figure()
ploth(iters_as)
ploth(iters_mn)

show()
