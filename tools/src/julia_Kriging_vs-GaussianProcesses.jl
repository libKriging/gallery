# %% [markdown]
# # Julia · Kriging vs GaussianProcesses.jl
#
# Default fits on the shared datasets of `data/` (Branin, Hartmann 3, Borehole):
# JLibKriging (`matern5_2`, constant trend, single start and 10 restarts) against
# [GaussianProcesses.jl](https://github.com/STOR-i/GaussianProcesses.jl)
# (Matérn 5/2 ARD, constant mean, MLE), warn-level comparison of Q².
#
# The module is `JLibKriging` (the packaged binding, [JLibKriging.jl](https://github.com/libKriging/JLibKriging.jl)),
# imported under a short name (as GaussianProcesses.jl): both export `predict`,
# `optimize!`, ... so every call is qualified.

# %%
import JLibKriging as LK
include("tools/julia/checks.jl")
s = Suite("julia_Kriging_vs-GaussianProcesses")

# %% [markdown]
# ## Installing GaussianProcesses.jl
#
# Added to the active Julia environment (with Optim.jl, its optimizer). Without
# it, the comparison is skipped (warn-level).

# %%
import Pkg
have_gp = try
    Pkg.add(["GaussianProcesses", "Optim"])
    @eval import GaussianProcesses as GP
    @eval import Optim
    true
catch e
    println("GaussianProcesses.jl could not be installed: ", sprint(showerror, e))
    false
end

# %% [markdown]
# ## Default fits

# %%
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
