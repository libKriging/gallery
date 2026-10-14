# %% [markdown]
# # R · `Kriging` with noise — estimated nugget, known variances
#
# Branin on [0,1]² (n = 30 LHS points unless stated, Q² on 500 uniform test
# points), through rlibkriging. Checks (error-level unless stated):
#
# * `noise = "nugget"` (10 restarts): nugget > 0, Q² > 0.95 on the noise-free function, smoothing next to the design points
# * the default single start reaches the multistart likelihood (warn-level)
# * `save` then `load`: the same predicted mean
# * `noise = <variances>`: the stored variances, finite predictions

# %%
suppressPackageStartupMessages(library(rlibkriging))
source("tools/r/checks.R")
s <- new_suite("noisy-data_Kriging-nugget-noise_R")
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

# %%
me <- do.call(rbind, s$metrics)
print(me[me$metric == "q2", c("case", "package", "value")], row.names = FALSE)

# %%
finalize(s)
