# ────────────────────────────────────────────────────────────────────────────────
# TFIM + (x or z)-dephasing : Evolution of Renyi-2 correlator
# Random initial state
# ────────────────────────────────────────────────────────────────────────────────
using LinearAlgebra
using SparseArrays
using Random
using JLD
using Plots
using ProgressMeter

# -------- embed a 2×2 operator at site j (1..L) into a 2^L×2^L operator --------
function embed_site(op::SparseMatrixCSC{ComplexF64,Int}, j::Int, L::Int)
    I2 = spdiagm(0 => ones(ComplexF64, 2))
    M = sparse(ComplexF64[1.0;;])         # start from 1×1 identity
    @inbounds for s in 1:L
        M = kron(M, s == j ? op : I2)
    end
    return M
end

# -------- build TFIM operators and Hamiltonian --------
# H = -J Σ σ^z_j σ^z_{j+1} - h Σ σ^x_j  (PBC)
function build_TFIM(L::Int, J::Float64, h::Float64)
    # single-site Pauli (as sparse)
    σx = sparse(ComplexF64[0 1; 1 0])
    σz = sparse(ComplexF64[1 0; 0 -1])

    Z = [embed_site(σz, j, L) for j in 1:L]
    X = [embed_site(σx, j, L) for j in 1:L]
    D = size(Z[1], 1)
    H = spzeros(ComplexF64, D, D)
    @inbounds for j in 1:L
        jp1 = (j == L ? 1 : j + 1)
        H .-= J .* (Z[j] * Z[jp1])
        H .-= h .* X[j]
    end

    return H, Z, X
end

# -------- Lindblad set: L_j = σ^α_j  (α = :z or :x) --------
build_L_ops(Z, X; axis::Symbol=:z) = (axis == :z ? Z : X)

# -------- Z_2 symmetry operator U = Π X_n
function build_U(X)
    L = length(X)
    U = X[1]
    for i in 2:L
        U *= X[i]
    end
    return U
end

# -------- RHS: dρ/dt = -i[H,ρ] + γ Σ_j (L_j ρ L_j - ρ) --------
function rhs(ρ::Matrix{ComplexF64},
    H::SparseMatrixCSC{ComplexF64,Int},
    Ls::Vector{SparseMatrixCSC{ComplexF64,Int}},
    γ::Float64)
    # coherent part: -i(Hρ - ρH)
    comm = -1im .* (H * ρ - ρ * H)

    # dissipative part: γ Σ (L ρ L - ρ)
    diss = zeros(ComplexF64, size(ρ))
    for Lk in Ls
        diss .+= γ .* (Lk * ρ * Lk .- ρ)
    end

    return comm .+ diss
end

# ----- RK4 -----
function rk4_step!(ρ::Matrix{ComplexF64}, dt::Float64,
    H::SparseMatrixCSC{ComplexF64,Int},
    Ls::Vector{SparseMatrixCSC{ComplexF64,Int}},
    γ::Float64)
    k1 = rhs(ρ, H, Ls, γ)
    k2 = rhs(ρ .+ 0.5dt .* k1, H, Ls, γ)
    k3 = rhs(ρ .+ 0.5dt .* k2, H, Ls, γ)
    k4 = rhs(ρ .+ dt .* k3, H, Ls, γ)

    ρ .+= (dt / 6) .* (k1 .+ 2k2 .+ 2k3 .+ k4)
    return nothing
end

# ----- Initial states -----
# Build random state in symmetric sector
function build_random_state(L::Int, U::SparseMatrixCSC{ComplexF64,Int}, seed::Int)
    Random.seed!(seed)
    ψ = randn(ComplexF64, 2^L)
    ψ = ψ + U * ψ
    ψ /= norm(ψ)
    return ψ * ψ'
end

# ----- R̄2(t) -----
function Rbar2(ρ::Matrix{ComplexF64}, L::Int)
    dim = size(ρ, 1)

    P = real(tr(ρ * ρ))

    numerator = 0.0

    @inbounds for j in 1:dim
        for i in 1:dim
            dist = count_ones((i - 1) ⊻ (j - 1))
            factor = (L - 2 * dist)^2
            numerator += abs2(ρ[i, j]) * factor
        end
    end

    return numerator / (L^2 * P)
end

function simulate_Rbar2(seed::Integer;
    L::Int=5, J::Float64=1.0, h::Float64=1.0,
    γ::Float64=1.0, axis::Symbol=:x,
    dt::Float64=0.02, tmax::Float64=5.0,
    stride::Int=1)

    # operators and Hamiltonian (sparse)
    H, Z, X = build_TFIM(L, J, h)
    Ls = build_L_ops(Z, X; axis=axis)

    # Z_2 symmetry operator U = Π X_n
    U = build_U(X)

    # time grid
    nsteps = Int(round(tmax / dt))
    times = collect(0:nsteps) .* dt
    keepix = 1:stride:length(times)
    tkeep = times[keepix]
    Rbar = zeros(Float64, length(tkeep))

    # initial pure state
    ρ = build_random_state(L, U, seed)

    # record R̄2 along time
    idx = 1
    Rbar[idx] = Rbar2(ρ, L)

    @showprogress for step in 1:nsteps
        rk4_step!(ρ, dt, H, Ls, γ)
        if step % stride == 0
            idx += 1
            Rbar[idx] = Rbar2(ρ, L)
        end
    end

    return tkeep, Rbar, ρ
end

# ======== user parameters ========
# chain length (Hilbert dim = 2^L)
L = 12
# Ising zz coupling
J = 1.0
# transverse field along x
h = 0.0
# dephasing rate
γ = 1.0
# :z or :x  (choose Lindblad σ^z or σ^x)
axis = :x
# RK4 step
dt = 0.01
# total time
tmax = 2.0
# compute R̄2 every 'stride' steps (>=1)
stride = 4
# =================================

# number of samples for each L
nsample = Dict([("5", 200), ("6", 200), ("7", 100), ("8", 100), ("9", 50), ("10", 50)])

println(" J = ", J)
println(" h = ", h)
println(" L = ", L)

for seed = 1:1 #nsample["$(L)"]
    println("  seed = ", seed)
    t, R2, ρ_f = simulate_Rbar2(seed;
        L=L, J=J, h=h, γ=γ, axis=axis, dt=dt, tmax=tmax, stride=stride)

    save("data//Ising//random_initial_state//R2_J=$(J)_h=$(h)_L=$(L)_seed=$(seed).jld", "t", t, "R2", R2)
end
