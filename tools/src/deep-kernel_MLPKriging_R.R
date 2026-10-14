# %% [markdown]
# # R · `MLPKriging` — a joint feature map ℝ² → ℝ² before the kernel
#
# Branin on [0,1]² (n = 30 LHS points unless stated, Q² on 500 uniform test
# points), through rlibkriging. Checks (error-level unless stated):
#
# * 10 restarts (`BFGS10`): finite predictions, Q² > 0.9 (warn-level)
# * `save` then `load`: the same predicted mean
# * the default optimizer as good as `BFGS10` (warn-level; see libKriging/libKriging#382)
#
# A class missing from the installed rlibkriging (release channel) is recorded as SKIP.

# %%
suppressPackageStartupMessages(library(rlibkriging))
source("tools/r/checks.R")
s <- new_suite("deep-kernel_MLPKriging_R")
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

# %%
me <- do.call(rbind, s$metrics)
print(me[me$metric == "q2", c("case", "package", "value")], row.names = FALSE)

# %%
finalize(s)
