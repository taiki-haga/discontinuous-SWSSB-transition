# ────────────────────────────────────────────────────────────────────────────────
# TFIM + (x or z)-dephasing : Evolution of Fidelity Correlator
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

# ----- Fidelity Correlator F_O(x, y) -----
# Calculate averaged Fidelity Correlator over all pairs
# F_avg = (1/L^2) * sum_{x,y} F(ρ, Z_x Z_y ρ Z_x Z_y)
# F(ρ, U ρ U†) = tr √ (√ρ U ρ U† √ρ) = tr √ ( (√ρ U √ρ) (√ρ U √ρ)† )
# Since U = Z_x Z_y is Hermitian and Unitary, let B = √ρ U √ρ.
# B is Hermitian. We need tr √ (B^2) = sum(abs(eigvals(B))).
function FidelityBar(ρ::Matrix{ComplexF64}, Z::Vector{SparseMatrixCSC{ComplexF64,Int}})
    L = length(Z)
    dim = size(ρ, 1)

    # 1. Compute sqrt(ρ)
    evals, evecs = eigen(Hermitian(ρ))
    # Numerical safety: clamp small negative eigenvalues to 0
    evals .= max.(evals, 0.0)
    sqrt_rho = evecs * Diagonal(sqrt.(evals)) * evecs'

    acc_F = 0.0

    # Pre-calculate diagonals of Z operators to speed up multiplication
    Z_diags = [diag(Z[i]) for i in 1:L]

    for x in 1:L
        for y in 1:L
            if x == y
                # F(ρ, Z_x Z_x ρ Z_x Z_x) = F(ρ, ρ) = 1
                acc_F += 1.0
                continue
            end

            # Construct diagonal of U = Z_x * Z_y
            U_diag = Z_diags[x] .* Z_diags[y]

            # Compute B = √ρ * U * √ρ
            B = sqrt_rho * (Diagonal(U_diag) * sqrt_rho)

            # B is Hermitian. Fidelity is sum of absolute values of eigenvalues.
            f_xy = sum(abs.(eigvals(Hermitian(B))))

            acc_F += f_xy
        end
    end

    return acc_F / (L^2)
end

function simulate_Fidelity(seed::Integer;
    L::Int=5, J::Float64=1.0, h::Float64=1.0,
    γ::Float64=1.0, axis::Symbol=:x,
    dt::Float64=0.02, tmax::Float64=3.0,
    stride::Int=1)

    # operators and Hamiltonian (sparse)
    H, Z, X = build_TFIM(L, J, h)
    Ls = build_L_ops(Z, X; axis=axis)

    # Z_2 symmetry operator U = Π X_n
    U_sym = build_U(X)

    # time grid
    nsteps = Int(round(tmax / dt))
    times = collect(0:nsteps) .* dt
    keepix = 1:stride:length(times)
    tkeep = times[keepix]
    Fbar = zeros(Float64, length(tkeep))

    # initial pure state
    ρ = build_random_state(L, U_sym, seed)

    # record Fidelity along time
    idx = 1
    Fbar[idx] = FidelityBar(ρ, Z)

    @showprogress for step in 1:nsteps
        rk4_step!(ρ, dt, H, Ls, γ)

        if step % stride == 0
            idx += 1
            Fbar[idx] = FidelityBar(ρ, Z)
        end
    end

    return tkeep, Fbar, ρ
end

# ======== user parameters ========
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
dt = 0.02
# total time
tmax = 2.0
# compute Fidelity every 'stride' steps
stride = 2
# =================================

# number of samples for each L
nsample = Dict([("5", 100), ("6", 100), ("7", 50), ("8", 50), ("9", 25), ("10", 25)])

println(" L = ", L)
println(" J = ", J)
println(" h = ", h)

for seed = 9:10 #nsample["$(L)"]
    println("  seed = ", seed)
    t, F_vals, ρ_f = simulate_Fidelity(seed;
        L=L, J=J, h=h, γ=γ, axis=axis, dt=dt, tmax=tmax, stride=stride)

    save("data//Ising//random_initial_state//Fidelity_J=$(J)_h=$(h)_L=$(L)_seed=$(seed).jld", "t", t, "F", F_vals)
end
