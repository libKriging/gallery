# %% [markdown]
# # Julia · `Kriging` with an approximate fit — Vecchia, Nyström (n = 400)
#
# Branin on [0,1]² (n = 30 LHS points unless stated, Q² on 500 uniform test
# points), through JLibKriging. Checks (error-level unless stated):
#
# * the fit method is reported by `method()`
# * `predict` with the fit method: finite mean and stdev
# * Q² > 0.95 (warn-level)
#
# The same checks as `large-n_Kriging-vecchia-nystrom_R.ipynb`, with the Julia binding.

# %%
import JLibKriging as LK
using Random
include("tools/julia/checks.jl")
s = Suite("large-n_Kriging-vecchia-nystrom_julia")

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
# ## `Kriging` with an approximate fit: Vecchia, Nyström (n = 400)

# %%
Xl = lhs(400, 2); yl = branin(Xl)
for m in ("vecchia(20)", "nystrom(60)")
    short = split(m, "(")[1]
    label = "Kriging(method = '$m')"
    check_runs!(s, "$label: fit + predict") do
        ka = LK.Kriging(yl, Xl, "matern5_2"; method=m)
        add_check!(s, "$label: method()", LK.method(ka) == m; value=LK.method(ka), expected=m)
        pa = LK.predict(ka, Xt; return_stdev=true, method=String(short))
        add_check!(s, "$label: finite predictions", finite_count(pa.mean, pa.stdev) == 0;
                   value=finite_count(pa.mean, pa.stdev), expected="0 non-finite")
        add_metric!(s, "branin n=400", "Kriging $m"; q2=q2(yt, pa.mean))
        add_check!(s, "$label: Q2", q2(yt, pa.mean) > 0.95; value=q2(yt, pa.mean), expected="> 0.95", severity="warn")
    end
end

# %%
foreach(r -> r[5] == "q2" && println(rpad(r[3], 24), rpad(r[4], 28), r[6]), s.metrics)

# %%
finalize!(s)
