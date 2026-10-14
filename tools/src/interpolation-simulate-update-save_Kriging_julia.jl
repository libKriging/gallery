# %% [markdown]
# # Julia · `Kriging`, noise-free — interpolation, conditional simulation, update, save/load
#
# Branin on [0,1]² (n = 30 LHS points unless stated, Q² on 500 uniform test
# points), through JLibKriging. Checks (error-level unless stated):
#
# * interpolation (max |m(X) − y| / sd(y) < 1e-6) and Q² > 0.95 on 500 test points
# * conditional simulations through the data (relative to σ)
# * stdev > 0 off the design
# * `update` without refit = a fit on all the points at the same θ, σ²
# * `save` then `load`: the same predicted mean
#
# The same checks as `interpolation-simulate-update-save_Kriging_R.ipynb`, with the Julia binding.

# %%
import JLibKriging as LK
using Random
include("tools/julia/checks.jl")
s = Suite("interpolation-simulate-update-save_Kriging_julia")

branin(U) = begin
    x1 = 15 .* U[:, 1] .- 5; x2 = 15 .* U[:, 2]
    (x2 .- 5.1 / (4π^2) .* x1 .^ 2 .+ 5 / π .* x1 .- 6) .^ 2 .+ 10 * (1 - 1 / (8π)) .* cos.(x1) .+ 10
end
rng = MersenneTwister(1)
lhs(n, d) = hcat([(randperm(rng, n) .- rand(rng, n)) ./ n for _ in 1:d]...)
X = lhs(30, 2); y = branin(X)
Xt = rand(rng, 500, 2); yt = branin(Xt)
sd(v) = sqrt(sum((v .- sum(v) / length(v)) .^ 2) / (length(v) - 1))
finite_count(v...) = count(!isfinite, vcat(v...))

function check_runs!(f, s, check)
    try
        f()
    catch e
        add_check!(s, check, false; value=string(typeof(e)), expected="no error", detail=first(sprint(showerror, e), 200))
    end
end

function roundtrip!(s, label, k, Xn)
    check_runs!(s, "$label: save + load") do
        f = joinpath(mktempdir(), "k.json")
        LK.save(k, f)
        k2 = LK.load(f)
        check_close!(s, "$label: load(save(k)) predicts the same mean",
                     LK.predict(k2, Xn; return_stdev=false).mean, LK.predict(k, Xn; return_stdev=false).mean; rtol=1e-10)
    end
end

# %% [markdown]
# ## `Kriging`, noise-free

# %%
k = LK.Kriging(y, X, "matern5_2")
p = LK.predict(k, Xt; return_stdev=true)
add_metric!(s, "branin", "Kriging"; q2=q2(yt, p.mean))
add_check!(s, "Kriging: Q2", q2(yt, p.mean) > 0.95; value=q2(yt, p.mean), expected="> 0.95")
e = maximum(abs.(LK.predict(k, X; return_stdev=false).mean .- y)) / sd(y)
add_check!(s, "Kriging: interpolates (max |m(X) - y| / sd(y))", e < 1e-6; value=e, expected="< 1e-6")
S = LK.simulate(k, 5, 2, X)
e = maximum(abs.(S .- y)) / sqrt(LK.sigma2(k))
add_check!(s, "Kriging: conditional simulations go through the data (max |sim(X) - y| / sigma)", e < 1e-4;
           value=e, expected="< 1e-4")
add_check!(s, "Kriging: stdev > 0 off the design", minimum(p.stdev) > 0; value=minimum(p.stdev), expected="> 0")
Xu = lhs(5, 2); yu = branin(Xu)
ku = copy(k)
LK.update!(ku, yu, Xu; refit=false)
k35 = LK.Kriging(vcat(y, yu), vcat(X, Xu), "matern5_2"; optim="none",
                 theta=vec(LK.theta(k)), is_theta_estim=false, sigma2=LK.sigma2(k), is_sigma2_estim=false)
check_close!(s, "Kriging: update!(refit=false) == fit on all points (same theta, sigma2)",
             LK.predict(ku, Xt; return_stdev=false).mean, LK.predict(k35, Xt; return_stdev=false).mean;
             rtol=1e-6, atol=1e-8 * sd(y))
roundtrip!(s, "Kriging", k, Xt)

# %%
foreach(r -> r[5] == "q2" && println(rpad(r[3], 24), rpad(r[4], 28), r[6]), s.metrics)

# %%
finalize!(s)
