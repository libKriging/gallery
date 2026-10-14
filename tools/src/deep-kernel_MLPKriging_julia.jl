# %% [markdown]
# # Julia · `MLPKriging` — a joint feature map ℝ² → ℝ² before the kernel
#
# Branin on [0,1]² (n = 30 LHS points unless stated, Q² on 500 uniform test
# points), through JLibKriging. Checks (error-level unless stated):
#
# * 10 restarts (`BFGS10`): finite predictions, Q² > 0.9 (warn-level)
# * `save` then `load`: the same predicted mean
# * the default optimizer as good as `BFGS10` (warn-level; see libKriging/libKriging#382)
#
# The same checks as `deep-kernel_MLPKriging_R.ipynb`, with the Julia binding.

# %%
import JLibKriging as LK
using Random
include("tools/julia/checks.jl")
s = Suite("deep-kernel_MLPKriging_julia")

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
# ## `MLPKriging`: a joint feature map ℝ² → ℝ² before the kernel

# %%
check_runs!(s, "MLPKriging: fit + predict") do
    km = LK.MLPKriging(y, X, [8], 2; activation="tanh", kernel="matern5_2", optim="BFGS10")
    pm = LK.predict(km, Xt; return_stdev=true)
    add_check!(s, "MLPKriging: finite predictions", finite_count(pm.mean, pm.stdev) == 0;
               value=finite_count(pm.mean, pm.stdev), expected="0 non-finite")
    add_metric!(s, "branin", "MLPKriging tanh [8] -> 2, BFGS10"; q2=q2(yt, pm.mean))
    add_check!(s, "MLPKriging: Q2", q2(yt, pm.mean) > 0.9; value=q2(yt, pm.mean), expected="> 0.9", severity="warn")
    roundtrip!(s, "MLPKriging", km, Xt)
    kd = LK.MLPKriging(y, X, [8], 2; activation="tanh", kernel="matern5_2")   # default optimizer
    qd = q2(yt, LK.predict(kd, Xt; return_stdev=false).mean)
    add_metric!(s, "branin", "MLPKriging tanh [8] -> 2, default optim"; q2=qd)
    add_check!(s, "MLPKriging: Q2(default optim) - Q2(BFGS10)", qd - q2(yt, pm.mean) > -0.05;
               value=qd - q2(yt, pm.mean), expected="> -0.05", severity="warn")
end

# %%
foreach(r -> r[5] == "q2" && println(rpad(r[3], 24), rpad(r[4], 28), r[6]), s.metrics)

# %%
finalize!(s)
