# %% [markdown]
# # Julia 01 — Cross-binding consistency, and a comparison with GaussianProcesses.jl
#
# 1. Same protocol as `notebooks/python/11_cross_binding.ipynb` (see there), on
#    the shared datasets of `data/`; writes `results/xbind_julia.csv`, compared
#    with the Python values by `tools/report.py`.
# 2. Default fits against [GaussianProcesses.jl](https://github.com/STOR-i/GaussianProcesses.jl)
#    (Matérn 5/2 ARD, constant mean, MLE), when it is installed (warn-level).
#
# The module is `JLibKriging` (the packaged binding, [JLibKriging.jl](https://github.com/libKriging/JLibKriging.jl)),
# imported under a short name so that its exported accessors (`X`, `y`, `theta`, ...)
# do not clash with the variables of the notebook.

# %%
import JLibKriging as LK
include("checks.jl")
s = Suite("jl01_cross_binding")
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

# %% [markdown]
# ## Default fits vs GaussianProcesses.jl

# %%
have_gp = try
    @eval import GaussianProcesses as GP
    @eval import Optim
    true
catch
    false
end
for name in ("branin", "hartmann3", "borehole")
    Xd = read_data(name, "X"); yd = vec(read_data(name, "y")); Xt = read_data(name, "Xt"); yt = vec(read_data(name, "yt"))
    t = @elapsed k = LK.Kriging(yd, Xd, "matern5_2")
    q_lk = q2(yt, LK.predict(k, Xt; return_stdev=false).mean)
    add_metric!(s, name, "JLibKriging"; q2=q_lk, fit_s=t)
    t = @elapsed k10 = LK.Kriging(yd, Xd, "matern5_2"; optim="BFGS10")
    q_lk10 = q2(yt, LK.predict(k10, Xt; return_stdev=false).mean)
    add_metric!(s, name, "JLibKriging BFGS10"; q2=q_lk10, fit_s=t)
    add_check!(s, "$name: Q2(JLibKriging BFGS10) > 0.9", q_lk10 > 0.9; value=q_lk10, expected="> 0.9")
    if have_gp
        try
            d = size(Xd, 2)
            t = @elapsed begin
                gp = GP.GP(permutedims(Xd), yd, GP.MeanConst(sum(yd) / length(yd)),
                           GP.Mat52Ard(zeros(d), log(sum((yd .- sum(yd) / length(yd)) .^ 2) / length(yd)) / 2), -8.0)
                GP.optimize!(gp; noise=false)
            end
            mg, _ = GP.predict_f(gp, permutedims(Xt))
            q_gp = q2(yt, mg)
            add_metric!(s, name, "GaussianProcesses.jl"; q2=q_gp, fit_s=t)
            add_check!(s, "$name: Q2(JLibKriging) - Q2(GaussianProcesses.jl)", q_lk - q_gp > -0.02;
                       value=q_lk - q_gp, expected="> -0.02", severity="warn")
        catch e
            add_check!(s, "$name: GaussianProcesses.jl fit", false; value=string(typeof(e)), severity="warn")
        end
    end
end
have_gp || add_check!(s, "GaussianProcesses.jl available", false; value="absent", severity="warn")
foreach(r -> println(join(r[3:6], "  ")), s.metrics)

# %%
finalize!(s)
