# %% [markdown]
# # Julia · all classes — cross-binding consistency: the Julia side
#
# **Idea.** The Julia side of the cross-binding protocol: JLibKriging must give the
# numbers of pylibkriging on the same data, for every model class and fit method.
#
# **Reference.** The Python values (`cross-binding-reference_all-
# classes_python.ipynb`), compared by `tools/report.py`.
#
# Same protocol as `cross-binding-reference_all-classes_python.ipynb` (see there), on
# the shared datasets of `data/`; writes `results/xbind_julia.csv`, compared with
# the Python values by `tools/report.py`.
#
# The module is `JLibKriging` (the packaged binding, [JLibKriging.jl](https://github.com/libKriging/JLibKriging.jl)),
# imported under a short name so that its exported accessors (`X`, `y`, `theta`, ...)
# do not clash with the variables of the notebook.

# %%
import JLibKriging as LK
include("tools/julia/gallery.jl")
s = Suite("cross-binding_all-classes_julia")
rows = Vector{Vector{String}}()
record!(dataset, quantity, values) =
    for (i, v) in enumerate(vec(Float64.(collect(values))))
        push!(rows, ["julia", dataset, quantity, string(i), @sprintf("%.17g", v)])
    end

# %%
for name in ("branin", "hartmann3", "borehole")
    Xd = read_data(name, "X"); yd = vec(read_data(name, "y")); Xt = read_data(name, "Xt")[1:50, :]
    d = size(Xd, 2)
    theta0 = fill(0.3, d)
    kf = LK.Kriging(yd, Xd, "matern5_2"; regmodel="constant", optim="none", theta=theta0, is_theta_estim=false)
    record!(name, "fixed_sigma2", LK.sigma2(kf))
    record!(name, "fixed_beta", LK.beta(kf))
    record!(name, "fixed_ll", LK.log_likelihood(kf))
    record!(name, "fixed_loo", LK.leave_one_out(kf))
    p = LK.predict(kf, Xt; return_stdev=true)
    record!(name, "fixed_mean", p.mean)
    record!(name, "fixed_stdev", p.stdev)
    llf = LK.log_likelihood_fun(kf, theta0; return_grad=true)
    record!(name, "llfun", llf.ll)
    record!(name, "llfun_grad", llf.grad)
    check_close!(s, "$name: log_likelihood() == log_likelihood_fun(theta0)", LK.log_likelihood(kf), llf.ll; rtol=1e-10)
    k = LK.Kriging(yd, Xd, "matern5_2")
    record!(name, "fit_theta", LK.theta(k))
    record!(name, "fit_ll", LK.log_likelihood(k))
    record!(name, "fit_mean", LK.predict(k, Xt; return_stdev=false).mean)
    record!(name, "sim", LK.simulate(kf, 3, 123, Xt[1:5, :]))
end

# %% [markdown]
# ## Other classes and fit methods (Branin)

# %%
name = "branin"
Xd = read_data(name, "X"); yd = vec(read_data(name, "y")); Xt = read_data(name, "Xt")[1:50, :]
theta0 = [0.3, 0.3]
s2 = sum((yd .- sum(yd) / length(yd)) .^ 2) / length(yd); nug = 0.01 * s2   # population variance, as numpy.var
kn = LK.Kriging(yd, Xd, "matern5_2"; noise="nugget", optim="none", theta=theta0, is_theta_estim=false,
                sigma2=s2, is_sigma2_estim=false, nugget=nug, is_nugget_estim=false)
p = LK.predict(kn, Xt; return_stdev=true)
record!(name, "nugget_fixed_mean", p.mean); record!(name, "nugget_fixed_stdev", p.stdev)
kh = LK.Kriging(yd, Xd, "matern5_2"; noise=fill(nug, length(yd)), optim="none", theta=theta0, is_theta_estim=false,
                sigma2=s2, is_sigma2_estim=false)
p = LK.predict(kh, Xt; return_stdev=true)
record!(name, "noise_fixed_mean", p.mean); record!(name, "noise_fixed_stdev", p.stdev)
kf = LK.Kriging(yd, Xd, "matern5_2"; noise="nugget")
record!(name, "nugget_fit_ll", LK.log_likelihood(kf))
record!(name, "nugget_fit_mean", LK.predict(kf, Xt; return_stdev=false).mean)
for meth in ("vecchia", "nystrom")
    ka = LK.Kriging(yd, Xd, "matern5_2"; optim="none", theta=theta0, is_theta_estim=false, method="$meth(10)")
    record!(name, "$(meth)_fixed_ll", LK.log_likelihood(ka; method=meth))
    record!(name, "$(meth)_fixed_mean", LK.predict(ka, Xt; return_stdev=false, method=meth).mean)
end
kw = LK.WarpKriging(yd, Xd, ["kumaraswamy", "kumaraswamy"], "matern5_2")
record!(name, "warp_fit_mean", LK.predict(kw, Xt; return_stdev=false).mean)
km = LK.MLPKriging(yd, Xd, [8], 2; activation="tanh", kernel="matern5_2", optim="BFGS10")
record!(name, "mlp_fit_mean", LK.predict(km, Xt; return_stdev=false).mean)
kk = LK.NestedKriging(yd, Xd, "matern5_2", 2; aggregation="NK", seed=1)
record!(name, "nested_fit_mean", LK.predict(kk, Xt; return_stdev=false).mean)

# %%
open(joinpath(RESULTS_DIR, "xbind_julia.csv"), "w") do io
    println(io, "lang,dataset,quantity,index,value")
    for r in rows; println(io, join(r, ",")); end
end
println(length(rows), " values written")

# %% [markdown]
# ## The values written
#
# First value of each quantity and dataset (each compared with Python by the
# report; plots are left to the Python reference, to avoid a plotting package here).

# %%
for key in unique([(r[2], r[3]) for r in rows])
    vals = [r[5] for r in rows if (r[2], r[3]) == key]
    println(rpad("$(key[1]) $(key[2])", 36), vals[1], " (", length(vals), " values)")
end

# %% [markdown]
# ## Summary
#
# The metrics of this notebook, the number of checks per status, and every check that
# did not pass (FAIL: error-level, fails the run; WARN: warn-level; KNOWN: listed in
# `tools/known_issues.csv`, with its issue; SKIP: feature absent from the installed version).
# All of them are also written to `results/cross-binding_all-classes_julia.checks.csv` and `.metrics.csv`.

# %%
suite_summary(s)

# %%
suite_finalize!(s)
