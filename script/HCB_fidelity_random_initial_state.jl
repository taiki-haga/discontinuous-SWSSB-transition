# ────────────────────────────────────────────────────────────────────────────────
#  Time evolution of Fidelity correlator 
#  RK4 solver for the Lindblad master equation
# ────────────────────────────────────────────────────────────────────────────────
using LinearAlgebra
using SparseArrays
using Combinatorics
using Statistics
using Random
using JLD
using Plots
using ProgressMeter

# ───────────────────────────────────────────────
# 1. Parameters
# ───────────────────────────────────────────────
const L = 14           # lattice size
const Npart = 7        # particle number (hard-core)
const J = 1.0          # hopping amplitude
const γ = 1.0          # dephasing rate
const dt = 0.01        # RK4 step
const tmax = 4.0

# ───────────────────────────────────────────────
# 2. Fixed-N Hilbert space
# ───────────────────────────────────────────────
"""
    build_basis(L, N)

Return list of bitstrings (as Int) whose binary form has exactly N set bits.
Also return Dicts to map <--> index.
"""
function build_basis(L::Int, N::Int)
    states = [sum((1 << (i - 1)) for i in occ) for occ in combinations(1:L, N)]
    state2idx = Dict(s => i for (i, s) in enumerate(states))
    return states, state2idx
end

const BASIS, STATE2IDX = build_basis(L, Npart)
const DIM = length(BASIS)

# ───────────────────────────────────────────────
# 3. hopping operator  b†_i b_j  (i=j ⇒ n_i)
# ───────────────────────────────────────────────
function move_op(i::Int, j::Int)
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    if i == j
        # number operator: diagonal matrix
        for (idx, bits) in enumerate(BASIS)
            occ = (bits >> (i - 1)) & 1
            if occ == 1
                push!(rows, idx)
                push!(cols, idx)
                push!(vals, 1.0)
            end
        end
    else
        for (ket_idx, bits) in enumerate(BASIS)
            has_i = (bits >> (i - 1)) & 1 == 1     # destination must be empty
            has_j = (bits >> (j - 1)) & 1 == 1     # source must be occupied
            if !has_i && has_j
                # move particle j → i
                new_bits = (bits | (1 << (i - 1))) & ~(1 << (j - 1))
                bra_idx = STATE2IDX[new_bits]    # stays inside fixed-N space
                push!(rows, bra_idx)
                push!(cols, ket_idx)
                push!(vals, 1.0)                  # hard-core boson ⇒ +1
            end
        end
    end
    return sparse(rows, cols, ComplexF64.(vals), DIM, DIM)
end

# Pre-compute
const N_OPS = [move_op(i, i) for i in 1:L]              # n_i
const BdB = [move_op(i, j) for i in 1:L, j in 1:L]    # all b†_i b_j

# ───────────────────────────────────────────────
# 4. Hamiltonian  H = −J Σ_i (b†_i b_{i+1} + h.c.)
# ───────────────────────────────────────────────
function build_H()
    H = spzeros(ComplexF64, DIM, DIM)
    for i in 1:L
        j = (i == L ? 1 : i + 1)
        T = move_op(i, j)                  # b†_i b_{i+1}
        H .-= J .* (T + T')                # + hermitian conjugate
    end
    return H
end
const H = build_H()

# ───────────────────────────────────────────────
# 5. Liouvillian right-hand side   dρ/dt = 𝓛(ρ)
# ───────────────────────────────────────────────
function L_rhs!(dρ::Matrix{ComplexF64}, ρ::Matrix{ComplexF64})
    dρ .= -1im * (H * ρ - ρ * H)                               # coherent
    for n in N_OPS                                             # dephasing
        dρ .+= γ .* (n * ρ * n - 0.5 * (n * ρ + ρ * n))
    end
    return dρ
end

# ───────────────────────────────────────────────
# 6. 4-th-order Runge-Kutta integrator
# ───────────────────────────────────────────────
function rk4_step!(ρ::Matrix{ComplexF64}, dt::Float64,
    k_rhs::Matrix{ComplexF64}, k_mid::Matrix{ComplexF64})
    # k_rhs : holds derivatives (k₁,k₂,k₃,k₄)
    # k_mid : reused buffer for intermediate density matrices
    L_rhs!(k_rhs, ρ)
    k1 = copy(k_rhs)

    k_mid .= ρ .+ 0.5dt .* k1
    L_rhs!(k_rhs, k_mid)
    k2 = copy(k_rhs)

    k_mid .= ρ .+ 0.5dt .* k2
    L_rhs!(k_rhs, k_mid)
    k3 = copy(k_rhs)

    k_mid .= ρ .+ dt .* k3
    L_rhs!(k_rhs, k_mid)
    k4 = copy(k_rhs)

    ρ .+= dt / 6 .* (k1 .+ 2k2 .+ 2k3 .+ k4)

    # enforce Hermiticity & trace = 1
    #ρ .= 0.5 .* (ρ + ρ')
    #ρ ./= real(tr(ρ))
    return nothing
end

# ───────────────────────────────────────────────
# 7. Fidelity correlator
# ───────────────────────────────────────────────

# Calculate averaged Fidelity Correlator over all pairs
# F_avg = (1/L^2) * sum_{i,j} F(ρ, b†_i b_j ρ b†_j b_i)
# F(ρ, O ρ O†) = tr √ (√ρ O ρ O† √ρ)
function FidelityBar(ρ::Matrix{ComplexF64})

    # Compute sqrt(ρ)
    evals, evecs = eigen(Hermitian(ρ))
    # Numerical safety: clamp small negative eigenvalues to 0
    evals .= max.(evals, 0.0)
    sqrt_rho = evecs * Diagonal(sqrt.(evals)) * evecs'

    acc_F = 0.0

    for i in 1:L
        for j in 1:L
            # Compute B = √ρ O ρ O† √ρ
            B = sqrt_rho * BdB[i, j] * ρ * BdB[j, i] * sqrt_rho

            # Compute tr √B
            evals, evecs = eigen(Hermitian(B))
            # Numerical safety: clamp small negative eigenvalues to 0
            evals .= max.(evals, 0.0)
            f = sum(sqrt.(evals))

            acc_F += f
        end
    end

    return acc_F / (L^2)
end

# ───────────────────────────────────────────────
# 8. One trajectory from a random pure state
# ───────────────────────────────────────────────

function fidelity_evolution(steps::Int, seed::Int)
    Random.seed!(seed)
    ψ = randn(ComplexF64, DIM)
    ψ /= norm(ψ)
    ρ = ψ * ψ'

    k_rhs = zeros(ComplexF64, DIM, DIM) # derivative buffer
    k_mid = similar(k_rhs)              # mid-state buffer

    samp_int = 4
    times = (0:Int(steps / samp_int)) * samp_int * dt
    F = zeros(Int(steps / samp_int) + 1)

    @showprogress for idx in 0:steps
        if idx % samp_int == 0
            F[Int(idx / samp_int)+1] = FidelityBar(ρ)
        end
        rk4_step!(ρ, dt, k_rhs, k_mid)
    end

    return times, F, ρ
end

# ───────────────────────────────────────────────
# 9.  Main simulation
# ───────────────────────────────────────────────

steps = Int(round(tmax / dt))

println(" L = ", L)
println(" N = ", Npart)

for seed in 6:10
    println("  seed = ", seed)
    t, F, ρ_f = fidelity_evolution(steps, seed)
    save("data//HCB//random_initial_state//Fidelity_J=$(J)_L=$(L)_N=$(Npart)_seed=$(seed).jld", "t", t, "F", F)
end
