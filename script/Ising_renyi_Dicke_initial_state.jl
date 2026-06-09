# ────────────────────────────────────────────────────────────────────────────────
# TFIM + (x or z)-dephasing : Evolution of Renyi-2 correlator
# Initial state: Dicke state (L/2 excitations)
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
# H = -J Σ σ^z_j σ^z_{j+1} - hx Σ σ^x_j - hz Σ σ^z_j  (PBC)
function build_TFIM(L::Int, J::Float64, hx::Vector{Float64}, hz::Vector{Float64})
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
        H .-= hx[j] .* X[j]
        H .-= hz[j] .* Z[j]
    end

    return H, Z, X
end

# -------- Lindblad set: L_j = σ^α_j  (α = :z or :x) --------
build_L_ops(Z, X; axis::Symbol=:z) = (axis == :z ? Z : X)

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
function build_dicke_state(L)
    dim = 2^L
    ψ = zeros(ComplexF64, dim)

    # Target number of excitations (Down spins |1>)
    # Assuming standard Dicke state is "half-filled"
    k = div(L, 2)

    for i in 0:dim-1
        # Check Hamming weight (number of 1s in binary rep)
        # 0 corresponds to |0> (Up), 1 corresponds to |1> (Down)
        if count_ones(i) == k
            ψ[i+1] = 1.0
        end
    end

    normalize!(ψ)
    return ψ * ψ'
end

# ----- R̄2(t) -----
# Precompute all Z_j, and pair strings Z_j Z_k to speed up Tr[ρ A ρ A]
function precompute_Zpairs(Z::Vector{SparseMatrixCSC{ComplexF64,Int}})
    L = length(Z)
    Zpair = [Z[j] * Z[k] for j in 1:L, k in 1:L]  # sparse
    return Zpair
end

function Rbar2(ρ::Matrix{ComplexF64}, Zpair::Array{SparseMatrixCSC{ComplexF64,Int},2})
    L = size(Zpair, 1)
    P = real(tr(ρ * ρ))   # purity
    acc = 0.0
    for j in 1:L, k in 1:L
        A = Zpair[j, k]
        acc += real(tr(ρ * Matrix(A) * ρ * Matrix(A))) / P
    end
    return acc / (L^2)
end

function simulate_Rbar2(ρ0; L::Int=5, J::Float64=1.0, hx::Vector{Float64}, hz::Vector{Float64},
    γ::Float64=1.0, axis::Symbol=:x,
    dt::Float64=0.02, tmax::Float64=5.0,
    stride::Int=1)

    # operators and Hamiltonian (sparse)
    H, Z, X = build_TFIM(L, J, hx, hz)
    Ls = build_L_ops(Z, X; axis=axis)

    # precompute A = Z_m Z_n (sparse)
    Zpair = precompute_Zpairs(Z)

    # time grid
    nsteps = Int(round(tmax / dt))
    times = collect(0:nsteps) .* dt
    keepix = 1:stride:length(times)
    tkeep = times[keepix]
    Rbar = zeros(Float64, length(tkeep))

    # initial pure state
    ρ = copy(ρ0)

    # record R̄2 along time
    idx = 1
    Rbar[idx] = Rbar2(ρ, Zpair)

    @showprogress for step in 1:nsteps
        rk4_step!(ρ, dt, H, Ls, γ)
        if (step + 1) % stride == 0
            idx += 1
            Rbar[idx] = Rbar2(ρ, Zpair)
        end
    end

    return tkeep, Rbar, ρ
end

# ======== user parameters ========
# chain length (Hilbert dim = 2^L)
L = 12 # even
# Ising zz coupling
J = 1.0
# transverse field along x
hx = 0.0 * ones(L)
# transverse field along z
hz = 0.0 * ones(L)
# dephasing rate
γ = 1.0
# :z or :x  (choose Lindblad σ^z or σ^x)
axis = :x
# RK4 step
dt = 0.02
# total time
tmax = 3.0
# compute R̄2 every 'stride' steps (>=1)
stride = 2
# =================================

println(" L = ", L)

ρ0 = build_dicke_state(L)

t, R2, ρ_f = simulate_Rbar2(ρ0; L=L, J=J, hx=hx, hz=hz, γ=γ, axis=axis, dt=dt, tmax=tmax, stride=stride)

save("data//Ising//Dicke_initial_state//R2_J=1.0_h=0.0_L=$(L).jld", "t", t, "R2", R2)

#plot(t, R2)
