# %% [markdown]
# # R · all classes — cross-binding consistency: the R side
#
# Same protocol as `cross-binding-reference_all-classes_python.ipynb` (see there), on the
# shared datasets of `data/`; writes `results/xbind_R.csv`, compared with the
# Python values by `tools/report.py`.

# %%
suppressPackageStartupMessages(library(rlibkriging))
source("tools/r/checks.R")
s <- new_suite("cross-binding_all-classes_R")
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

# %% [markdown]
# ## Other classes and fit methods (Branin)

# %%
name <- "branin"
X <- read_data(name, "X"); y <- as.numeric(read_data(name, "y")); Xt <- read_data(name, "Xt")[1:50, , drop = FALSE]
theta0 <- matrix(0.3, nrow = 1, ncol = 2)
s2 <- mean((y - mean(y))^2); nug <- 0.01 * s2   # population variance, as numpy.var
fixed <- list(theta = theta0, is_theta_estim = FALSE, sigma2 = s2, is_sigma2_estim = FALSE)
kn <- Kriging(y, X, "matern5_2", noise = "nugget", optim = "none",
              parameters = c(fixed, list(nugget = nug, is_nugget_estim = FALSE)))
p <- predict(kn, Xt, return_stdev = TRUE)
put(name, "nugget_fixed_mean", p$mean); put(name, "nugget_fixed_stdev", p$stdev)
kh <- Kriging(y, X, "matern5_2", noise = rep(nug, length(y)), optim = "none", parameters = fixed)
p <- predict(kh, Xt, return_stdev = TRUE)
put(name, "noise_fixed_mean", p$mean); put(name, "noise_fixed_stdev", p$stdev)
kf <- Kriging(y, X, "matern5_2", noise = "nugget")
put(name, "nugget_fit_ll", logLikelihood(kf)); put(name, "nugget_fit_mean", predict(kf, Xt, return_stdev = FALSE)$mean)
for (meth in c("vecchia", "nystrom")) tryCatch({  # method= is absent from older releases
  ka <- Kriging(y, X, "matern5_2", optim = "none", parameters = list(theta = theta0, is_theta_estim = FALSE),
                method = paste0(meth, "(10)"))
  put(name, paste0(meth, "_fixed_ll"), ka$logLikelihood(method = meth))
  put(name, paste0(meth, "_fixed_mean"), predict(ka, Xt, return_stdev = FALSE, method = meth)$mean)
}, error = function(e) message(meth, " skipped (reported as missing): ", conditionMessage(e)))
kw <- WarpKriging(y, X, warping = c("kumaraswamy", "kumaraswamy"), kernel = "matern5_2")
put(name, "warp_fit_mean", predict(kw, Xt, return_stdev = FALSE)$mean)
km <- MLPKriging(y, X, hidden_dims = c(8), d_out = 2, activation = "tanh", kernel = "matern5_2", optim = "BFGS10")
put(name, "mlp_fit_mean", predict(km, Xt, return_stdev = FALSE)$mean)
kk <- NestedKriging(y, X, kernel = "matern5_2", nb_groups = 2, aggregation = "NK", seed = 1)
put(name, "nested_fit_mean", predict(kk, Xt, return_stdev = FALSE)$mean)

# %%
out <- do.call(rbind, rows)
write.csv(out, file.path(results_dir, "xbind_R.csv"), row.names = FALSE, quote = FALSE)
cat(nrow(out), "values written\n")

# %%
finalize(s)
