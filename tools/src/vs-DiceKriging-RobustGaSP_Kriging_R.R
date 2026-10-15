# %% [markdown]
# # R · Kriging vs DiceKriging and RobustGaSP
#
# **Idea.** libKriging's `Kriging` follows DiceKriging: at the same hyperparameters it
# must give DiceKriging's numbers, and its default fit must be as accurate as the R
# references.
#
# **Reference.** DiceKriging (exact equality) and RobustGaSP (accuracy, cost).
#
# DiceKriging is the reference implementation libKriging's `Kriging` follows
# (same product-form kernels and range parametrization, same concentrated
# likelihood). Hence two levels of comparison:
#
# 1. **Same hyperparameters → same numbers** (error-level): the concentrated
#    log-likelihood at a given θ (`DiceKriging::logLikFun` vs
#    `logLikelihoodFun`), and the universal kriging mean / stdev at fixed
#    (θ, σ², β) (`predict(type = "UK")`).
# 2. **Default fits** on the shared datasets of `data/` (Branin, Hartmann 3,
#    Borehole), with RobustGaSP as a third package: Q², fit time
#    (warn-level comparisons).

# %% [markdown]
# ## Installing DiceKriging and RobustGaSP
#
# From CRAN (the configured repository, else cloud.r-project.org), into the
# first library of `.libPaths()`, unless already installed. DiceKriging is
# required; without RobustGaSP its comparison is skipped (warn-level).

# %%
repos <- getOption("repos")
if (is.null(repos) || any(repos == "@CRAN@")) repos <- c(CRAN = "https://cloud.r-project.org")
for (p in c("DiceKriging", "RobustGaSP")) {
  if (!requireNamespace(p, quietly = TRUE)) {
    message("installing ", p)
    tryCatch(install.packages(p, repos = repos, lib = .libPaths()[1]),
             error = function(e) message(p, " could not be installed: ", conditionMessage(e)))
  }
}

# %%
suppressPackageStartupMessages({
  library(rlibkriging)
  library(DiceKriging)
})
has_rgasp <- requireNamespace("RobustGaSP", quietly = TRUE)
source("tools/r/checks.R")
s <- new_suite("vs-DiceKriging-RobustGaSP_Kriging_R")
cat("rlibkriging", as.character(packageVersion("rlibkriging")), "| DiceKriging", as.character(packageVersion("DiceKriging")),
    "| RobustGaSP", if (has_rgasp) as.character(packageVersion("RobustGaSP")) else "absent", "\n")
DATASETS <- c("branin", "hartmann3", "borehole")

# %% [markdown]
# ## 1. Same hyperparameters, same numbers

# %%
for (name in DATASETS) {
  X <- read_data(name, "X"); y <- as.numeric(read_data(name, "y")); Xt <- read_data(name, "Xt")
  d <- ncol(X)
  colnames(X) <- colnames(Xt) <- paste0("x", 1:d)
  for (kernel in c("gauss", "matern5_2", "matern3_2", "exp")) {
    theta <- seq(0.2, 0.6, length.out = d)
    k <- Kriging(y, X, kernel, optim = "none", parameters = list(theta = matrix(theta, nrow = 1), is_theta_estim = FALSE))
    km0 <- km(~1, design = data.frame(X), response = y, covtype = kernel, coef.cov = theta,
              coef.var = k$sigma2(), coef.trend = as.numeric(k$beta()), control = list(trace = FALSE))
    # concentrated log-likelihood at theta (DiceKriging profiles sigma2 and beta the same way)
    km_fit <- km(~1, design = data.frame(X), response = y, covtype = kernel, control = list(trace = FALSE))
    ll_dk <- DiceKriging::logLikFun(theta, km_fit)
    ll_lk <- logLikelihoodFun(k, matrix(theta, nrow = 1))$logLikelihood
    check_close(s, sprintf("%s/%s: logLikelihood == DiceKriging logLikFun", name, kernel), ll_lk, ll_dk, rtol = 1e-8)
    p_dk <- predict(km0, newdata = data.frame(Xt), type = "UK", checkNames = FALSE)
    p_lk <- predict(k, Xt, return_stdev = TRUE)
    check_close(s, sprintf("%s/%s: UK mean == DiceKriging", name, kernel), p_lk$mean, p_dk$mean, rtol = 1e-6, atol = 1e-8 * sd(y))
    check_close(s, sprintf("%s/%s: UK stdev == DiceKriging", name, kernel), p_lk$stdev, p_dk$sd, rtol = 1e-5, atol = 1e-8 * sd(y))
  }
}

# %% [markdown]
# ## 2. Default fits

# %%
results <- list()
for (name in DATASETS) {
  X <- read_data(name, "X"); y <- as.numeric(read_data(name, "y")); Xt <- read_data(name, "Xt"); yt <- as.numeric(read_data(name, "yt"))
  colnames(X) <- colnames(Xt) <- paste0("x", 1:ncol(X))
  fits <- list(
    rlibkriging = function() { k <- Kriging(y, X, "matern5_2"); function(Z) predict(k, Z)$mean },
    `rlibkriging BFGS10` = function() { k <- Kriging(y, X, "matern5_2", optim = "BFGS10"); function(Z) predict(k, Z)$mean },
    DiceKriging = function() {
      m <- km(~1, design = data.frame(X), response = y, covtype = "matern5_2", control = list(trace = FALSE))
      function(Z) predict(m, newdata = data.frame(Z), type = "UK", checkNames = FALSE)$mean
    })
  if (has_rgasp) fits$RobustGaSP <- function() {
    m <- RobustGaSP::rgasp(design = X, response = y, kernel_type = "matern_5_2")
    function(Z) RobustGaSP::predict(m, Z)$mean
  }
  for (pkg in names(fits)) {
    t0 <- Sys.time()
    r <- tryCatch(fits[[pkg]](), error = function(e) e)
    t_fit <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    if (inherits(r, "error")) {
      if (startsWith(pkg, "rlibkriging")) add_check(s, paste0(name, "/", pkg, ": fit"), FALSE, detail = conditionMessage(r))
      next
    }
    m <- as.numeric(r(Xt))
    add_metric(s, name, pkg, q2 = q2(yt, m), rmse = rmse(yt, m), fit_s = t_fit)
    results[[length(results) + 1]] <- data.frame(dataset = name, package = pkg, q2 = q2(yt, m), fit_s = t_fit)
  }
}
res <- do.call(rbind, results)
print(reshape(res[, c("dataset", "package", "q2", "fit_s")], idvar = "dataset", timevar = "package", direction = "wide"))
op <- par(mfrow = c(1, 2), mar = c(4, 4, 2, 1))
q <- tapply(res$q2, list(res$package, res$dataset), mean)
barplot(pmax(q, 0), beside = TRUE, legend.text = rownames(q), args.legend = list(x = "bottomright", cex = .7),
        ylim = c(0, 1), ylab = "Q2 (test set)", main = "Default fits: accuracy")
t <- tapply(res$fit_s, list(res$package, res$dataset), mean)
barplot(t, beside = TRUE, log = "y", ylab = "fit time (s)", main = "Default fits: cost")
par(op)

# %%
for (name in DATASETS) {
  r <- res[res$dataset == name, ]
  best_other <- max(r$q2[!startsWith(r$package, "rlibkriging")])
  check_greater(s, paste0(name, ": Q2(rlibkriging) - best other package"), r$q2[r$package == "rlibkriging"] - best_other, -0.02, severity = "warn")
  check_greater(s, paste0(name, ": Q2(rlibkriging BFGS10) > 0.9"), r$q2[r$package == "rlibkriging BFGS10"], 0.9)
}

# %% [markdown]
# ## Summary
#
# The metrics of this notebook, the number of checks per status, and every check that
# did not pass (FAIL: error-level, fails the run; WARN: warn-level; KNOWN: listed in
# `tools/known_issues.csv`, with its issue; SKIP: feature absent from the installed version).
# All of them are also written to `results/vs-DiceKriging-RobustGaSP_Kriging_R.checks.csv` and `.metrics.csv`.

# %%
summary_suite(s)

# %%
finalize(s)
