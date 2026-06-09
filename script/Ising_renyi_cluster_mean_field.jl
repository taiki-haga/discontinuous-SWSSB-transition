using LinearAlgebra
using CairoMakie

# ==============================================================================
# 1. Parameters
# ==============================================================================
const J = 1.0      # Interaction
const γ = 1.0      # Dephasing rate
const dt = 0.01
const T_max = 2.0
const STEPS = Int(T_max / dt)
const L_list = [8, 16, 32, 64] # System sizes

const σx = [0.0 1.0; 1.0 0.0]
const σz = [1.0 0.0; 0.0 -1.0]
const Id2 = I(2)

# ==============================================================================
# 2. Hamiltonian & Jump Operators
# ==============================================================================

# Operators on the 2-site Hilbert space (dim=4)
# Basis: |00>, |01>, |10>, |11>
const Id4 = I(4)
const H_pair_1copy = J * kron(σz, σz) # Heisenberg picture

# PairA(dim4) ⊗ PairB(dim4) => dim 16
# Hamiltonian H_total = H_A + H_B
const H_tot = kron(H_pair_1copy, Id4) + kron(Id4, H_pair_1copy)

const X1_pair = kron(σx, Id2)
const X2_pair = kron(Id2, σx)
const X1_X2_pair = kron(σx, σx)

# Lindblad operators
const Ops_diss = [
    kron(X1_pair, Id4), # X on 1A
    kron(X2_pair, Id4), # X on 2A
    kron(Id4, X1_pair), # X on 1B
    kron(Id4, X2_pair)  # X on 2B
]

const Ops_obs = [
    kron(Id4, Id4),              # I ⊗ I
    kron(X1_X2_pair, Id4),       # X ⊗ I
    kron(Id4, X1_X2_pair),       # I ⊗ X
    kron(X1_X2_pair, X1_X2_pair) # X ⊗ X
]

# ==============================================================================
# 3. Master equation
# ==============================================================================

# dPsi/dt = -i [H, Psi] + gamma * sum( Op * Psi * Op - Psi )
function dPsi_dt(Psi::Matrix{ComplexF64})
    # 1. Coherent part: -i (H Ψ - Ψ H)
    dPsi = -1im * (H_tot * Psi - Psi * H_tot)

    # 2. Dissipative part: γ Σ (X Ψ X - Ψ)
    diss_term = zeros(ComplexF64, 16, 16)
    for Op in Ops_diss
        diss_term += (Op * Psi * Op - Psi)
    end

    dPsi += γ * diss_term

    return dPsi
end

# RK4 step
function rk4_step(X, dt, f)
    k1 = f(X)
    k2 = f(X + 0.5 * dt * k1)
    k3 = f(X + 0.5 * dt * k2)
    k4 = f(X + dt * k3)
    return X + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
end

# ==============================================================================
# 4. Initial condition
# ==============================================================================

# 2-site Swap operator (16x16 Matrix)
function get_swap_matrix()
    S = zeros(ComplexF64, 16, 16)
    for i in 1:4
        for j in 1:4
            # Input basis index (Pair A=i, Pair B=j) -> row index
            # Basis order is i * 4 + j (roughly, using 1-based indexing)
            # Correct linear index for tensor product:
            # col (input) = (i-1)*4 + j
            # row (output, swapped) = (j-1)*4 + i

            col = (i - 1) * 4 + j
            row = (j - 1) * 4 + i
            S[row, col] = 1.0
        end
    end
    return S
end

const S_op = get_swap_matrix()

# ==============================================================================
# 5. Main
# ==============================================================================

function compute_observables()

    Psi = copy(S_op)

    times = Float64[]
    tau_II_vals = Float64[]
    tau_XI_vals = Float64[]
    tau_IX_vals = Float64[]
    tau_XX_vals = Float64[]

    curr_t = 0.0
    for s in 0:STEPS
        tau_II = real(tr(S_op * Ops_obs[1] * Psi))
        tau_XI = real(tr(S_op * Ops_obs[2] * Psi))
        tau_IX = real(tr(S_op * Ops_obs[3] * Psi))
        tau_XX = real(tr(S_op * Ops_obs[4] * Psi))

        push!(times, curr_t)
        push!(tau_II_vals, tau_II)
        push!(tau_XI_vals, tau_XI)
        push!(tau_IX_vals, tau_IX)
        push!(tau_XX_vals, tau_XX)

        Psi = rk4_step(Psi, dt, dPsi_dt)
        curr_t += dt
    end

    return times, tau_II_vals, tau_XI_vals, tau_IX_vals, tau_XX_vals
end

times, tau_II_vals, tau_XI_vals, tau_IX_vals, tau_XX_vals = compute_observables()

# ==============================================================================
# 6. Plot
# ==============================================================================

function calc_a(L, tau_II, tau_XI, tau_IX, tau_XX)
    L_half = L / 2.0

    t_II = (tau_II / 4.0)^L_half
    t_XI = (tau_XI / 4.0)^L_half
    t_IX = (tau_IX / 4.0)^L_half
    t_XX = (tau_XX / 4.0)^L_half

    return (t_II + t_XI + t_IX + t_XX) / 2.0
end

function calc_R2_bar(L, a)
    term_diag = 1.0 / L
    term_offdiag = ((L - 1.0) / L) / (1.0 + a)
    return term_diag + term_offdiag
end

function calc_normalized_S2(L, a)
    D = 2.0^(L - 1)

    P = (1.0 + a) / (D + 1.0)

    return -log(P) / ((L - 1) * log(2.0))
end

R2_dict = Dict{Int,Vector{Float64}}()
S2_dict = Dict{Int,Vector{Float64}}()

for L in L_list
    R2_vals = zeros(length(times))
    S2_vals = zeros(length(times))
    for t in eachindex(times)
        a = calc_a(L, tau_II_vals[t], tau_XI_vals[t], tau_IX_vals[t], tau_XX_vals[t])
        R2_vals[t] = calc_R2_bar(L, a)
        S2_vals[t] = calc_normalized_S2(L, a)
    end
    R2_dict[L] = R2_vals
    S2_dict[L] = S2_vals
end

function find_intersection(t_array, y1, y2)
    diff = y1 .- y2
    intersections = Float64[]
    for i in 1:(length(diff)-1)
        if diff[i] * diff[i+1] < 0.0
            t0, t1 = t_array[i], t_array[i+1]
            d0, d1 = diff[i], diff[i+1]
            t_c = t0 - d0 * (t1 - t0) / (d1 - d0)
            push!(intersections, t_c)
        end
    end
    return intersections
end

println("=== Critical Time (t_c) Evaluation ===")
t_c_list = Float64[]
for i in 1:(length(L_list)-1)
    L1 = L_list[i]
    L2 = L_list[i+1]
    t_c_vals = find_intersection(times, R2_dict[L1], R2_dict[L2])

    if !isempty(t_c_vals)
        t_c = t_c_vals[1]
        push!(t_c_list, t_c)
        println("Intersection L=$(L1) & L=$(L2): t_c ≈ $(round(t_c, digits=4))")
    else
        println("Intersection L=$(L1) & L=$(L2): Not found")
    end
end

mean_tc = isempty(t_c_list) ? NaN : sum(t_c_list) / length(t_c_list)
if !isnan(mean_tc)
    println("=> Estimated mean t_c: $(round(mean_tc, digits=4))")
end
println("======================================")

set_theme!(
    fontsize=28,
    Legend=(labelsize=28,),
    Axis=(
        xgridvisible=false,
        ygridvisible=false,
    )
)

fig = Figure(size=(1000, 400), figure_padding=(20, 30, 20, 20))
label_list = [L"L = 8", L"L = 16", L"L = 32", L"L = 64"]
cmap = cgrad(:plasma)
color_val = [cmap[(i-1)/5] for i in 1:5]

# Plot Renyi-2 correlator
ax1 = Axis(fig[1, 1],
    xlabel=L"\gamma t",
    ylabel=L"\bar{R}(t)",
    xticks=([0.0, 0.5, 1.0, 1.5, 2.0], [L"0", L"0.5", L"1", L"1.5", L"2.0"]),
    yticks=([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [L"0", L"0.2", L"0.4", L"0.6", L"0.8", L"1"])
)
xlims!(ax1, 0, 2)

for (i, L) in enumerate(L_list)
    lines!(ax1, times, R2_dict[L], label=label_list[i], linewidth=3, color=color_val[i])
end

if !isnan(mean_tc)
    vlines!(ax1, [mean_tc], color=:gray, linestyle=:dash, linewidth=2)
end
axislegend(ax1, position=:rb)
Label(fig[1, 1, TopLeft()], L"\mathrm{(a)}"; fontsize=25, halign=:left, padding=(-10, 0, 0, 0))

# Plot Renyi-2 entropy
ax2 = Axis(fig[1, 2],
    xlabel=L"\gamma t",
    ylabel=L"\tilde{S}_2(t)",
    xticks=([0.0, 0.5, 1.0, 1.5, 2.0], [L"0", L"0.5", L"1", L"1.5", L"2.0"]),
    yticks=([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [L"0", L"0.2", L"0.4", L"0.6", L"0.8", L"1"])
)
xlims!(ax2, 0, 2)

for (i, L) in enumerate(L_list)
    lines!(ax2, times, S2_dict[L], label=label_list[i], linewidth=3, color=color_val[i])
end

if !isnan(mean_tc)
    vlines!(ax2, [mean_tc], color=:gray, linestyle=:dash, linewidth=2)
end
axislegend(ax2, position=:rb)
Label(fig[1, 2, TopLeft()], L"\mathrm{(b)}"; fontsize=25, halign=:left, padding=(-10, 0, 0, 0))

colgap!(fig.layout, 1, Relative(0.05))

display(fig)
