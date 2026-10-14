# %% [markdown]
# # R · `Kriging` with an approximate fit — Vecchia, Nyström (n = 400)
#
# Branin on [0,1]² (n = 30 LHS points unless stated, Q² on 500 uniform test
# points), through rlibkriging. Checks (error-level unless stated):
#
# * the fit method is reported by `method()`
# * `predict` with the fit method: finite mean and stdev
# * Q² > 0.95 (warn-level)

# %%
suppressPackageStartupMessages(library(rlibkriging))
source("tools/r/checks.R")
s <- new_suite("large-n_Kriging-vecchia-nystrom_R")
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

# %%
me <- do.call(rbind, s$metrics)
print(me[me$metric == "q2", c("case", "package", "value")], row.names = FALSE)

# %%
finalize(s)
