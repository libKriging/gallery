# %% [markdown]
# # Julia · all classes — every libKriging model class and fit method through JLibKriging
#
# Same tour as `R_all-classes_api-tour.ipynb` (see there for the checks): Branin
# on [0,1]², every class and fit method of the Julia binding, each with the
# checks that hold for it whatever the data.
#
# The module is `JLibKriging` (the packaged binding, [JLibKriging.jl](https://github.com/libKriging/JLibKriging.jl)),
# imported under a short name so that its exported accessors (`X`, `y`, `theta`, ...)
# do not clash with the variables of the notebook.

# %%
import JLibKriging as LK
using Random
include("tools/julia/checks.jl")
s = Suite("julia_all-classes_api-tour")

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

# %% [markdown]
# ## `NestedKriging`: 4 groups of 100 points, `NK` and `PoE` aggregations

# %%
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
