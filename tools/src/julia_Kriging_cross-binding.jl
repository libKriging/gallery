# %% [markdown]
# # Julia · Kriging — cross-binding consistency: the Julia side
#
# Same protocol as `python_Kriging_cross-binding-reference.ipynb` (see there), on
# the shared datasets of `data/`; writes `results/xbind_julia.csv`, compared with
# the Python values by `tools/report.py`.
#
# The module is `JLibKriging` (the packaged binding, [JLibKriging.jl](https://github.com/libKriging/JLibKriging.jl)),
# imported under a short name so that its exported accessors (`X`, `y`, `theta`, ...)
# do not clash with the variables of the notebook.

# %%
import JLibKriging as LK
include("tools/julia/checks.jl")
s = Suite("julia_Kriging_cross-binding")
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
open(joinpath(RESULTS_DIR, "xbind_julia.csv"), "w") do io
    println(io, "lang,dataset,quantity,index,value")
    for r in rows; println(io, join(r, ",")); end
end
println(length(rows), " values written")

# %%
finalize!(s)
