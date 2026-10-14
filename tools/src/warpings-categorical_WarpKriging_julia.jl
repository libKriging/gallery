# %% [markdown]
# # Julia · `WarpKriging` — continuous warpings, a categorical input
#
# Branin on [0,1]² (n = 30 LHS points unless stated, Q² on 500 uniform test
# points), through JLibKriging. Checks (error-level unless stated):
#
# * `kumaraswamy` on both inputs, then x₂ restricted to 5 levels with `categorical(5,2)`: finite predictions
# * Q² floors (warn-level)
# * `save` then `load`: the same predicted mean
#
# The same checks as `warpings-categorical_WarpKriging_R.ipynb`, with the Julia binding.

# %%
import JLibKriging as LK
using Random
include("tools/julia/checks.jl")
s = Suite("warpings-categorical_WarpKriging_julia")

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
# ## `WarpKriging`: continuous warpings, then a categorical input
#
# The categorical case: x₂ restricted to 5 levels (coded 0, …, 4), embedded in ℝ².

# %%
check_runs!(s, "WarpKriging(kumaraswamy, kumaraswamy): fit + predict") do
    kw = LK.WarpKriging(y, X, ["kumaraswamy", "kumaraswamy"], "matern5_2")
    pw = LK.predict(kw, Xt; return_stdev=true)
    add_check!(s, "WarpKriging(kumaraswamy): finite predictions", finite_count(pw.mean, pw.stdev) == 0;
               value=finite_count(pw.mean, pw.stdev), expected="0 non-finite")
    add_metric!(s, "branin", "WarpKriging kumaraswamy"; q2=q2(yt, pw.mean))
    add_check!(s, "WarpKriging(kumaraswamy): Q2", q2(yt, pw.mean) > 0.9; value=q2(yt, pw.mean), expected="> 0.9",
               severity="warn")
    roundtrip!(s, "WarpKriging(kumaraswamy)", kw, Xt)
end
lev(U) = hcat(U[:, 1], round.(U[:, 2] .* 4))
unlev(L) = hcat(L[:, 1], L[:, 2] ./ 4)
Xc = lev(lhs(40, 2)); yc = branin(unlev(Xc))
Xtc = lev(Xt); ytc = branin(unlev(Xtc))
check_runs!(s, "WarpKriging(none, categorical(5,2)): fit + predict") do
    kc = LK.WarpKriging(yc, Xc, ["none", "categorical(5,2)"], "matern5_2")
    pc = LK.predict(kc, Xtc; return_stdev=true)
    add_check!(s, "WarpKriging(categorical): finite predictions", finite_count(pc.mean, pc.stdev) == 0;
               value=finite_count(pc.mean, pc.stdev), expected="0 non-finite")
    add_metric!(s, "branin, x2 categorical", "WarpKriging categorical"; q2=q2(ytc, pc.mean))
    add_check!(s, "WarpKriging(categorical): Q2", q2(ytc, pc.mean) > 0.8; value=q2(ytc, pc.mean), expected="> 0.8",
               severity="warn")
    roundtrip!(s, "WarpKriging(categorical)", kc, Xtc)
end

# %%
foreach(r -> r[5] == "q2" && println(rpad(r[3], 24), rpad(r[4], 28), r[6]), s.metrics)

# %%
finalize!(s)
