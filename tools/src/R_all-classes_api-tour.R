# %% [markdown]
# # R · all classes — every libKriging model class and fit method through rlibkriging
#
# One problem (Branin on [0,1]², Q² on 500 test points), every class and every fit
# method of the R binding, each with the checks that hold for it whatever the
# data:
#
# | model | checks (error-level unless stated) |
# |---|---|
# | `Kriging` (noise-free) | interpolation, Q² floor, conditional simulations through the data, `update` = fit on all points (same θ), `save`/`load` |
# | `Kriging(noise = "nugget")` | nugget > 0, smoothing next to the design points, single-start optimum as good as the multistart one (warn), `save`/`load` |
# | `Kriging(noise = <variances>)` | the stored variances, finite predictions |
# | `Kriging(method = "vecchia(m)")`, `"nystrom(k)"` | `k$method()`, `predict(method = ...)` finite, Q² floor (warn) |
# | `WarpKriging` (continuous and categorical warpings) | finite predictions, Q² floor (warn), `save`/`load` |
# | `MLPKriging` | finite predictions, Q² floor (warn), default optimizer as good as a multistart BFGS (warn), `save`/`load` |
# | `NestedKriging` (`NK`, `PoE`) | `NK` interpolates, stdev ≥ 0, `save`/`load` |
#
# A class missing from the installed rlibkriging (release channel) is recorded as
# SKIP.

# %%
suppressPackageStartupMessages(library(rlibkriging))
source("tools/r/checks.R")
s <- new_suite("R_all-classes_api-tour")
cat("rlibkriging", as.character(packageVersion("rlibkriging")), "\n")

branin <- function(U) {  # on [0,1]^2
  x1 <- 15 * U[, 1] - 5; x2 <- 15 * U[, 2]
  (x2 - 5.1 / (4 * pi^2) * x1^2 + 5 / pi * x1 - 6)^2 + 10 * (1 - 1 / (8 * pi)) * cos(x1) + 10
}
set.seed(1)
lhs <- function(n, d) sapply(1:d, function(j) (sample(n) - runif(n)) / n)
X <- lhs(30, 2); y <- branin(X)
Xt <- matrix(runif(1000), ncol = 2); yt <- branin(Xt)
tmp <- tempdir()
has_class <- function(name) exists(name, mode = "function")
skip <- function(s, check, reason) {
  s$checks[[length(s$checks) + 1]] <- data.frame(suite = s$name, lang = "R", check = check, severity = "info",
    status = "SKIP", value = "", expected = "", detail = reason, stringsAsFactors = FALSE)
  message("[SKIP] ", check, ": ", reason)
}
roundtrip_same_mean <- function(s, label, k, Xn) {
  f <- file.path(tmp, paste0(gsub("[^A-Za-z0-9]", "_", label), ".json"))
  check_runs(s, paste0(label, ": save + load"), {
    save(k, f)
    k2 <- rlibkriging::load(f)
    check_close(s, paste0(label, ": load(save(k)) predicts the same mean"),
                predict(k2, Xn, return_stdev = FALSE)$mean, predict(k, Xn, return_stdev = FALSE)$mean, rtol = 1e-10)
  })
}

# %% [markdown]
# ## `Kriging`, noise-free

# %%
k <- Kriging(y, X, "matern5_2")
print(k)
p <- predict(k, Xt, return_stdev = TRUE)
add_metric(s, "branin", "Kriging", q2 = q2(yt, p$mean), rmse = rmse(yt, p$mean))
check_greater(s, "Kriging: Q2", q2(yt, p$mean), 0.95)
check_less(s, "Kriging: interpolates (max |m(X) - y| / sd(y))",
           max(abs(predict(k, X, return_stdev = FALSE)$mean - y)) / sd(y), 1e-6)
sims <- simulate(k, nsim = 5, seed = 2, x = X)
check_less(s, "Kriging: conditional simulations go through the data (max |sim(X) - y| / sigma)",
           max(abs(sims - y)) / sqrt(k$sigma2()), 1e-4)
check_greater(s, "Kriging: stdev > 0 off the design", min(p$stdev), 0)
# update with 5 new points (no refit) = a fit on the 35 points at the same theta
Xu <- lhs(5, 2); yu <- branin(Xu)
ku <- copy(k)
update(ku, yu, Xu, refit = FALSE)
k35 <- Kriging(c(y, yu), rbind(X, Xu), "matern5_2", optim = "none",
               parameters = list(theta = matrix(k$theta(), nrow = 1), is_theta_estim = FALSE,
                                 sigma2 = k$sigma2(), is_sigma2_estim = FALSE))
check_close(s, "Kriging: update(refit = FALSE) == fit on all points (same theta, sigma2)",
            predict(ku, Xt, return_stdev = FALSE)$mean, predict(k35, Xt, return_stdev = FALSE)$mean,
            rtol = 1e-6, atol = 1e-8 * sd(y))
roundtrip_same_mean(s, "Kriging", k, Xt)

# %% [markdown]
# ## `Kriging` with noise: estimated nugget, known variances
#
# Noise of standard deviation 5% of sd(y). The nugget model is fitted with 10
# restarts (`optim = "BFGS10"`); the default single start is compared with it
# (warn-level: a lower likelihood means a local optimum). As DiceKriging's
# nugget, the predictor returns y exactly *at* the design points (the nugget is
# a micro-scale variation): smoothing is checked next to them.

# %%
sd_noise <- 0.05 * sd(y)
yn <- y + rnorm(length(y), sd = sd_noise)
kn <- Kriging(yn, X, "matern5_2", noise = "nugget", optim = "BFGS10")
pn <- predict(kn, Xt, return_stdev = TRUE)
add_metric(s, "branin noisy", "Kriging nugget BFGS10", q2 = q2(yt, pn$mean), nugget = kn$nugget(),
           ll = logLikelihood(kn))
check_greater(s, "Kriging(nugget): estimated nugget > 0", kn$nugget(), 0)
check_greater(s, "Kriging(nugget): Q2", q2(yt, pn$mean), 0.95)
near <- pmin(X + 1e-7, 1)
check_greater(s, "Kriging(nugget): smooths the noisy data next to the design points (rms residual / noise sd)",
              rmse(yn, predict(kn, near, return_stdev = FALSE)$mean) / sd_noise, 0.3)
kn1 <- Kriging(yn, X, "matern5_2", noise = "nugget")
add_metric(s, "branin noisy", "Kriging nugget BFGS", q2 = q2(yt, predict(kn1, Xt, return_stdev = FALSE)$mean),
           nugget = kn1$nugget(), ll = logLikelihood(kn1))
check_greater(s, "Kriging(nugget): LL(single start) - LL(10 restarts)", logLikelihood(kn1) - logLikelihood(kn), -1,
              severity = "warn")
roundtrip_same_mean(s, "Kriging(nugget)", kn, Xt)
kh <- Kriging(yn, X, "matern5_2", noise = rep(sd_noise^2, length(yn)))
check_close(s, "Kriging(noise = variances): noise() returns them", kh$noise(), rep(sd_noise^2, length(yn)),
            rtol = 1e-12)
ph <- predict(kh, Xt, return_stdev = TRUE)
check_less(s, "Kriging(noise = variances): non-finite predictions", sum(!is.finite(c(ph$mean, ph$stdev))), 1)
add_metric(s, "branin noisy", "Kriging known noise", q2 = q2(yt, ph$mean))

# %% [markdown]
# ## `Kriging` with an approximate fit: Vecchia, Nyström (n = 400)

# %%
Xl <- lhs(400, 2); yl <- branin(Xl)
for (m in c("vecchia(20)", "nystrom(60)")) {
  short <- sub("\\(.*", "", m)
  check_runs(s, paste0("Kriging(method = '", m, "'): fit + predict"), {
    ka <- Kriging(yl, Xl, "matern5_2", method = m)
    add_check(s, paste0("Kriging(method = '", m, "'): method()"), identical(ka$method(), m), ka$method(), m)
    pa <- predict(ka, Xt, return_stdev = TRUE, method = short)
    check_less(s, paste0("Kriging(method = '", m, "'): non-finite predictions"),
               sum(!is.finite(c(pa$mean, pa$stdev))), 1)
    add_metric(s, "branin n=400", paste("Kriging", m), q2 = q2(yt, pa$mean))
    check_greater(s, paste0("Kriging(method = '", m, "'): Q2"), q2(yt, pa$mean), 0.95, severity = "warn")
  })
}

# %% [markdown]
# ## `WarpKriging`: continuous warpings, then a categorical input
#
# The categorical case: x₂ restricted to 5 levels (coded 0, …, 4), embedded in ℝ².

# %%
if (has_class("WarpKriging")) {
  check_runs(s, "WarpKriging(kumaraswamy, kumaraswamy): fit + predict", {
    kw <- WarpKriging(y, X, warping = c("kumaraswamy", "kumaraswamy"), kernel = "matern5_2")
    pw <- predict(kw, Xt, return_stdev = TRUE)
    check_less(s, "WarpKriging(kumaraswamy): non-finite predictions", sum(!is.finite(c(pw$mean, pw$stdev))), 1)
    add_metric(s, "branin", "WarpKriging kumaraswamy", q2 = q2(yt, pw$mean))
    check_greater(s, "WarpKriging(kumaraswamy): Q2", q2(yt, pw$mean), 0.9, severity = "warn")
    roundtrip_same_mean(s, "WarpKriging(kumaraswamy)", kw, Xt)
  })
  lev <- function(U) cbind(U[, 1], round(U[, 2] * 4))      # level codes 0..4
  unlev <- function(L) cbind(L[, 1], L[, 2] / 4)
  Xc <- lev(lhs(40, 2)); yc <- branin(unlev(Xc))
  Xtc <- lev(Xt); ytc <- branin(unlev(Xtc))
  check_runs(s, "WarpKriging(none, categorical(5,2)): fit + predict", {
    kc <- WarpKriging(yc, Xc, warping = c("none", "categorical(5,2)"), kernel = "matern5_2")
    pc <- predict(kc, Xtc, return_stdev = TRUE)
    check_less(s, "WarpKriging(categorical): non-finite predictions", sum(!is.finite(c(pc$mean, pc$stdev))), 1)
    add_metric(s, "branin, x2 categorical", "WarpKriging categorical", q2 = q2(ytc, pc$mean))
    check_greater(s, "WarpKriging(categorical): Q2", q2(ytc, pc$mean), 0.8, severity = "warn")
    roundtrip_same_mean(s, "WarpKriging(categorical)", kc, Xtc)
  })
} else skip(s, "WarpKriging", "not in this rlibkriging")

# %% [markdown]
# ## `MLPKriging`: a joint feature map ℝ² → ℝ² before the kernel

# %%
if (has_class("MLPKriging")) {
  check_runs(s, "MLPKriging: fit + predict", {
    km <- MLPKriging(y, X, hidden_dims = c(8), d_out = 2, activation = "tanh", kernel = "matern5_2", optim = "BFGS10")
    pm <- predict(km, Xt, return_stdev = TRUE)
    check_less(s, "MLPKriging: non-finite predictions", sum(!is.finite(c(pm$mean, pm$stdev))), 1)
    add_metric(s, "branin", "MLPKriging tanh [8] -> 2, BFGS10", q2 = q2(yt, pm$mean))
    check_greater(s, "MLPKriging: Q2", q2(yt, pm$mean), 0.9, severity = "warn")
    roundtrip_same_mean(s, "MLPKriging", km, Xt)
    # the default optimizer ("BFGS+Adam")
    kd <- MLPKriging(y, X, hidden_dims = c(8), d_out = 2, activation = "tanh", kernel = "matern5_2")
    qd <- q2(yt, predict(kd, Xt, return_stdev = FALSE)$mean)
    add_metric(s, "branin", "MLPKriging tanh [8] -> 2, default optim", q2 = qd)
    check_greater(s, "MLPKriging: Q2(default optim) - Q2(BFGS10)", qd - q2(yt, pm$mean), -0.05, severity = "warn")
  })
} else skip(s, "MLPKriging", "not in this rlibkriging")

# %% [markdown]
# ## `NestedKriging`: 4 groups of 100 points, `NK` and `PoE` aggregations

# %%
if (has_class("NestedKriging")) {
  for (agg in c("NK", "PoE")) {
    check_runs(s, paste0("NestedKriging(", agg, "): fit + predict"), {
      kk <- NestedKriging(yl, Xl, kernel = "matern5_2", nb_groups = 4, aggregation = agg, seed = 3)
      pk <- predict(kk, Xt, return_stdev = TRUE)
      check_less(s, paste0("NestedKriging(", agg, "): stdev < 0"), sum(pk$stdev < 0), 1)
      add_metric(s, "branin n=400", paste("NestedKriging", agg), q2 = q2(yt, pk$mean))
      check_greater(s, paste0("NestedKriging(", agg, "): Q2"), q2(yt, pk$mean), 0.95, severity = "warn")
      if (agg == "NK")
        check_less(s, "NestedKriging(NK): interpolates (max |m(X) - y| / sd(y))",
                   max(abs(predict(kk, Xl, return_stdev = FALSE)$mean - yl)) / sd(yl), 1e-3)
      roundtrip_same_mean(s, paste0("NestedKriging(", agg, ")"), kk, Xt)
    })
  }
} else skip(s, "NestedKriging", "not in this rlibkriging")

# %%
me <- do.call(rbind, s$metrics)
print(me[me$metric == "q2", c("case", "package", "value")], row.names = FALSE)

# %%
finalize(s)
