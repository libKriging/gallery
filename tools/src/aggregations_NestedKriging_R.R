# %% [markdown]
# # R · `NestedKriging` — `NK` and `PoE` aggregations, 4 groups of 100 points
#
# Branin on [0,1]² (n = 30 LHS points unless stated, Q² on 500 uniform test
# points), through rlibkriging. Checks (error-level unless stated):
#
# * stdev ≥ 0, Q² > 0.95 (warn-level)
# * `NK` interpolates the data (it is a proper kriging predictor)
# * `save` then `load`: the same predicted mean
#
# A class missing from the installed rlibkriging (release channel) is recorded as SKIP.

# %%
suppressPackageStartupMessages(library(rlibkriging))
source("tools/r/checks.R")
s <- new_suite("aggregations_NestedKriging_R")
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
# ## `NestedKriging`: 4 groups of 100 points, `NK` and `PoE` aggregations

# %%
Xl <- lhs(400, 2); yl <- branin(Xl)
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
