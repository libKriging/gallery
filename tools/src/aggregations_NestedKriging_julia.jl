# %% [markdown]
# # Julia · `NestedKriging` — `NK` and `PoE` aggregations, 4 groups of 100 points
#
# Branin on [0,1]² (n = 30 LHS points unless stated, Q² on 500 uniform test
# points), through JLibKriging. Checks (error-level unless stated):
#
# * stdev ≥ 0, Q² > 0.95 (warn-level)
# * `NK` interpolates the data (it is a proper kriging predictor)
# * `save` then `load`: the same predicted mean
#
# The same checks as `aggregations_NestedKriging_R.ipynb`, with the Julia binding.

# %%
import JLibKriging as LK
using Random
include("tools/julia/checks.jl")
s = Suite("aggregations_NestedKriging_julia")

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
# ## `NestedKriging`: 4 groups of 100 points, `NK` and `PoE` aggregations

# %%
Xl = lhs(400, 2); yl = branin(Xl)
for agg in ("NK", "PoE")
    label = "NestedKriging($agg)"
    check_runs!(s, "$label: fit + predict") do
        kk = LK.NestedKriging(yl, Xl, "matern5_2", 4; aggregation=agg, seed=3)
        pk = LK.predict(kk, Xt; return_stdev=true)
        add_check!(s, "$label: stdev >= 0", all(pk.stdev .>= 0); value=minimum(pk.stdev), expected=">= 0")
        add_metric!(s, "branin n=400", "NestedKriging $agg"; q2=q2(yt, pk.mean))
        add_check!(s, "$label: Q2", q2(yt, pk.mean) > 0.95; value=q2(yt, pk.mean), expected="> 0.95", severity="warn")
        if agg == "NK"
            e = maximum(abs.(LK.predict(kk, Xl; return_stdev=false).mean .- yl)) / sd(yl)
            add_check!(s, "NestedKriging(NK): interpolates (max |m(X) - y| / sd(y))", e < 1e-3; value=e, expected="< 1e-3")
        end
        roundtrip!(s, label, kk, Xt)
    end
end

# %%
foreach(r -> r[5] == "q2" && println(rpad(r[3], 24), rpad(r[4], 28), r[6]), s.metrics)

# %%
finalize!(s)
