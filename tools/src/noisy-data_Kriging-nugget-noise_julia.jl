# %% [markdown]
# # Julia · `Kriging` with noise — estimated nugget, known variances
#
# Branin on [0,1]² (n = 30 LHS points unless stated, Q² on 500 uniform test
# points), through JLibKriging. Checks (error-level unless stated):
#
# * `noise = "nugget"` (10 restarts): nugget > 0, Q² > 0.95 on the noise-free function, smoothing next to the design points
# * the default single start reaches the multistart likelihood (warn-level)
# * `save` then `load`: the same predicted mean
# * `noise = <variances>`: the stored variances, finite predictions
#
# The same checks as `noisy-data_Kriging-nugget-noise_R.ipynb`, with the Julia binding.

# %%
import JLibKriging as LK
using Random
include("tools/julia/checks.jl")
s = Suite("noisy-data_Kriging-nugget-noise_julia")

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
# ## `Kriging` with noise: estimated nugget, known variances
#
# Noise of standard deviation 5% of sd(y). The nugget model is fitted with 10
# restarts (`optim="BFGS10"`); the default single start is compared with it
# (warn-level: a lower likelihood means a local optimum). As DiceKriging's
# nugget, the predictor returns y exactly *at* the design points (the nugget is
# a micro-scale variation): smoothing is checked next to them.

# %%
sd_noise = 0.05 * sd(y)
yn = y .+ sd_noise .* randn(rng, length(y))
kn = LK.Kriging(yn, X, "matern5_2"; noise="nugget", optim="BFGS10")
mn = LK.predict(kn, Xt; return_stdev=false).mean
add_metric!(s, "branin noisy", "Kriging nugget BFGS10"; q2=q2(yt, mn))
add_check!(s, "Kriging(nugget): estimated nugget > 0", LK.nugget(kn) > 0; value=LK.nugget(kn), expected="> 0")
add_check!(s, "Kriging(nugget): Q2", q2(yt, mn) > 0.95; value=q2(yt, mn), expected="> 0.95")
r = sqrt(sum((LK.predict(kn, min.(X .+ 1e-7, 1.0); return_stdev=false).mean .- yn) .^ 2) / length(yn)) / sd_noise
add_check!(s, "Kriging(nugget): smooths the noisy data next to the design points (rms residual / noise sd)", r > 0.3;
           value=r, expected="> 0.3")
kn1 = LK.Kriging(yn, X, "matern5_2"; noise="nugget")
add_metric!(s, "branin noisy", "Kriging nugget BFGS"; q2=q2(yt, LK.predict(kn1, Xt; return_stdev=false).mean))
dll = LK.log_likelihood(kn1) - LK.log_likelihood(kn)
add_check!(s, "Kriging(nugget): LL(single start) - LL(10 restarts)", dll > -1; value=dll, expected="> -1", severity="warn")
roundtrip!(s, "Kriging(nugget)", kn, Xt)
v = fill(sd_noise^2, length(yn))
kh = LK.Kriging(yn, X, "matern5_2"; noise=v)
check_close!(s, "Kriging(noise = variances): noise() returns them", LK.noise(kh), v; rtol=1e-12)
ph = LK.predict(kh, Xt; return_stdev=true)
add_check!(s, "Kriging(noise = variances): finite predictions", finite_count(ph.mean, ph.stdev) == 0;
           value=finite_count(ph.mean, ph.stdev), expected="0 non-finite")
add_metric!(s, "branin noisy", "Kriging known noise"; q2=q2(yt, ph.mean))

# %%
foreach(r -> r[5] == "q2" && println(rpad(r[3], 24), rpad(r[4], 28), r[6]), s.metrics)

# %%
finalize!(s)
