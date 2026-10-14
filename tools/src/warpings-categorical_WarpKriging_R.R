# %% [markdown]
# # R · `WarpKriging` — continuous warpings, a categorical input
#
# Branin on [0,1]² (n = 30 LHS points unless stated, Q² on 500 uniform test
# points), through rlibkriging. Checks (error-level unless stated):
#
# * `kumaraswamy` on both inputs, then x₂ restricted to 5 levels with `categorical(5,2)`: finite predictions
# * Q² floors (warn-level)
# * `save` then `load`: the same predicted mean
#
# A class missing from the installed rlibkriging (release channel) is recorded as SKIP.

# %%
suppressPackageStartupMessages(library(rlibkriging))
source("tools/r/checks.R")
s <- new_suite("warpings-categorical_WarpKriging_R")
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

# %%
me <- do.call(rbind, s$metrics)
print(me[me$metric == "q2", c("case", "package", "value")], row.names = FALSE)

# %%
finalize(s)
