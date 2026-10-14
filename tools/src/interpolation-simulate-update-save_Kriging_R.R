# %% [markdown]
# # R · `Kriging`, noise-free — interpolation, conditional simulation, update, save/load
#
# Branin on [0,1]² (n = 30 LHS points unless stated, Q² on 500 uniform test
# points), through rlibkriging. Checks (error-level unless stated):
#
# * interpolation (max |m(X) − y| / sd(y) < 1e-6) and Q² > 0.95 on 500 test points
# * conditional simulations through the data (relative to σ)
# * stdev > 0 off the design
# * `update` without refit = a fit on all the points at the same θ, σ²
# * `save` then `load`: the same predicted mean

# %%
suppressPackageStartupMessages(library(rlibkriging))
source("tools/r/checks.R")
s <- new_suite("interpolation-simulate-update-save_Kriging_R")
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

# %%
me <- do.call(rbind, s$metrics)
print(me[me$metric == "q2", c("case", "package", "value")], row.names = FALSE)

# %%
finalize(s)
