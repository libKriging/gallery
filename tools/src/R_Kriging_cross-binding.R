# %% [markdown]
# # R · Kriging — cross-binding consistency: the R side
#
# Same protocol as `python_Kriging_cross-binding-reference.ipynb` (see there), on the
# shared datasets of `data/`; writes `results/xbind_R.csv`, compared with the
# Python values by `tools/report.py`.

# %%
suppressPackageStartupMessages(library(rlibkriging))
source("tools/r/checks.R")
s <- new_suite("R_Kriging_cross-binding")
cat("rlibkriging", as.character(packageVersion("rlibkriging")), "\n")
rows <- list()
put <- function(dataset, quantity, values) {
  v <- as.numeric(values)
  rows[[length(rows) + 1]] <<- data.frame(lang = "R", dataset = dataset, quantity = quantity,
                                          index = seq_along(v), value = sprintf("%.17g", v))
}

# %%
for (name in c("branin", "hartmann3", "borehole")) {
  X <- read_data(name, "X"); y <- as.numeric(read_data(name, "y")); Xt <- read_data(name, "Xt")[1:50, , drop = FALSE]
  d <- ncol(X)
  theta0 <- matrix(0.3, nrow = 1, ncol = d)
  kf <- Kriging(y, X, "matern5_2", regmodel = "constant", optim = "none",
                parameters = list(theta = theta0, is_theta_estim = FALSE))
  put(name, "fixed_sigma2", kf$sigma2())
  put(name, "fixed_beta", kf$beta())
  put(name, "fixed_ll", logLikelihood(kf))
  put(name, "fixed_loo", leaveOneOut(kf))
  p <- predict(kf, Xt, return_stdev = TRUE)
  put(name, "fixed_mean", p$mean)
  put(name, "fixed_stdev", p$stdev)
  llf <- logLikelihoodFun(kf, theta0, return_grad = TRUE)
  put(name, "llfun", llf$logLikelihood)
  put(name, "llfun_grad", llf$logLikelihoodGrad)
  check_close(s, paste0(name, ": logLikelihood() == logLikelihoodFun(theta0)"), logLikelihood(kf), llf$logLikelihood, rtol = 1e-10)
  k <- Kriging(y, X, "matern5_2")
  put(name, "fit_theta", k$theta())
  put(name, "fit_ll", logLikelihood(k))
  put(name, "fit_mean", predict(k, Xt, return_stdev = FALSE)$mean)
  put(name, "sim", simulate(kf, nsim = 3, seed = 123, x = Xt[1:5, , drop = FALSE]))
}
out <- do.call(rbind, rows)
write.csv(out, file.path(results_dir, "xbind_R.csv"), row.names = FALSE, quote = FALSE)
cat(nrow(out), "values written\n")

# %%
finalize(s)
