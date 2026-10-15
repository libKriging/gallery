# Helpers of the libKriging gallery notebooks (R), in one file: source("tools/R/gallery.R").
#
# The same functions, under the same names, as tools/python/gallery.py (where they are
# methods of Suite: suite.check_close(...) is check_close(s, ...) here), and the same
# CSV outputs (results/<suite>.checks.csv, .metrics.csv), aggregated with the other
# languages by tools/report.py.
#
#   suite(name)                                  a suite of checks, named after the notebook
#   check(s, check, ok, value, expected, ...)    record a check (severity "error" or "warn")
#   check_true, check_less, check_greater, check_between, check_close, check_finite,
#   check_runs, check_skip                       the usual checks
#   metric(s, case, package, name = value, ...)  record metrics
#   suite_summary(s)                             end-of-notebook summary
#   suite_finalize(s)                            write the CSV files, stop if an error-level check failed
#   read_data(name, stem), rmse, q2, nlpd, coverage
#
# Statuses: PASS, FAIL (error-level), WARN (warn-level), SKIP (feature absent from the
# installed version), KNOWN (failing, listed in tools/known_issues.csv with its issue).
# Paths are relative to the repository root ($LKGALLERY_ROOT, else the working directory,
# where the notebooks are executed).

.gallery_root <- Sys.getenv("LKGALLERY_ROOT", ".")
.known_file <- file.path(.gallery_root, "tools", "known_issues.csv")
if (!file.exists(.known_file)) stop("run the notebooks from the repository root (", .known_file, " not found)")
.known <- read.csv(.known_file, stringsAsFactors = FALSE)
results_dir <- Sys.getenv("LKGALLERY_RESULTS", file.path(.gallery_root, "results"))
dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)
.STATUSES <- c("PASS", "FAIL", "WARN", "KNOWN", "SKIP")

# ---- suite and checks -----------------------------------------------------------------

suite <- function(name) {
  s <- new.env()
  s$name <- name
  s$checks <- list()
  s$metrics <- list()
  s$t0 <- Sys.time()
  s
}

.fmt <- function(v) if (is.null(v) || length(v) == 0) "" else if (is.numeric(v)) format(signif(v[1], 6)) else as.character(v[1])

.record <- function(s, check, severity, status, value, expected, detail) {
  s$checks[[length(s$checks) + 1]] <- data.frame(suite = s$name, lang = "R", check = check, severity = severity,
    status = status, value = .fmt(value), expected = .fmt(expected), detail = detail, stringsAsFactors = FALSE)
}

check <- function(s, check, ok, value = NULL, expected = NULL, detail = "", severity = "error") {
  ok <- isTRUE(ok)
  status <- if (ok) "PASS" else if (severity == "error") "FAIL" else "WARN"
  k <- .known[.known$suite == s$name & .known$check == check, ]
  if (!ok && nrow(k) > 0) { status <- "KNOWN"; detail <- paste(k$issue[1], detail, sep = " | ") }
  if (!ok) message(sprintf("[%s] %s: value=%s expected %s %s", status, check, .fmt(value), .fmt(expected), detail))
  .record(s, check, severity, status, value, expected, detail)
  invisible(ok)
}

check_true <- function(s, check, cond, detail = "", severity = "error")
  check(s, check, isTRUE(cond), cond, TRUE, detail, severity)

check_less <- function(s, check, value, bound, detail = "", severity = "error")
  check(s, check, is.finite(value) && value < bound, value, paste("<", bound), detail, severity)

check_greater <- function(s, check, value, bound, detail = "", severity = "error")
  check(s, check, is.finite(value) && value > bound, value, paste(">", bound), detail, severity)

check_between <- function(s, check, value, lo, hi, detail = "", severity = "error")
  check(s, check, is.finite(value) && value >= lo && value <= hi, value, sprintf("in [%g, %g]", lo, hi), detail, severity)

check_close <- function(s, check, a, b, rtol = 1e-6, atol = 0, detail = "", severity = "error") {
  a <- as.numeric(a); b <- as.numeric(b)
  if (length(a) != length(b))
    return(check(s, check, FALSE, paste("length", length(a)), paste("length", length(b)), detail, severity))
  gap <- abs(a - b)
  ok <- all(is.finite(a)) && all(gap <= atol + rtol * abs(b))
  rel <- max(gap / pmax(abs(b), 1e-300))
  check(s, check, ok, rel, sprintf("rtol %g, atol %g", rtol, atol), detail, severity)
}

check_finite <- function(s, check, a, detail = "", severity = "error")
  check(s, check, length(a) > 0 && all(is.finite(a)), paste(sum(!is.finite(a)), "non-finite"), "all finite", detail, severity)

check_runs <- function(s, check, expr, severity = "error") {
  # check_runs(s, "fit", { ... }): an error becomes a failed check instead of stopping the notebook
  r <- tryCatch({ force(expr); NULL }, error = function(e) conditionMessage(e))
  check(s, check, is.null(r), if (is.null(r)) "ok" else "error", "no error", if (is.null(r)) "" else substr(r, 1, 200), severity)
}

check_skip <- function(s, check, reason) {
  .record(s, check, "info", "SKIP", "", "", reason)
  message("[SKIP] ", check, ": ", reason)
}

# ---- metrics and results --------------------------------------------------------------

metric <- function(s, case, package, ...) {
  v <- list(...)
  for (n in names(v)) s$metrics[[length(s$metrics) + 1]] <- data.frame(suite = s$name, lang = "R", case = case,
    package = package, metric = n, value = .fmt(v[[n]]), stringsAsFactors = FALSE)
}

suite_summary <- function(s) {
  # end-of-notebook summary: metrics, count per status, every check that did not pass
  if (length(s$metrics) > 0) {
    me <- do.call(rbind, s$metrics)
    me$value <- as.numeric(me$value)
    print(reshape(me[, c("case", "package", "metric", "value")], idvar = c("case", "package"),
                  timevar = "metric", direction = "wide"), row.names = FALSE)
  }
  ch <- do.call(rbind, s$checks)
  print(table(factor(ch$status, levels = .STATUSES)))
  other <- ch[ch$status != "PASS", c("status", "severity", "check", "value", "expected", "detail")]
  if (nrow(other) > 0) print(other, row.names = FALSE)
  invisible(ch)
}

suite_finalize <- function(s) {
  metric(s, "_suite", "_", elapsed_s = as.numeric(difftime(Sys.time(), s$t0, units = "secs")))
  ch <- do.call(rbind, s$checks)
  me <- do.call(rbind, s$metrics)
  write.csv(ch, file.path(results_dir, paste0(s$name, ".checks.csv")), row.names = FALSE)
  write.csv(me, file.path(results_dir, paste0(s$name, ".metrics.csv")), row.names = FALSE)
  print(table(factor(ch$status, levels = .STATUSES)))
  failed <- ch$check[ch$status == "FAIL"]
  if (length(failed) > 0)
    stop(sprintf("%d check(s) failed in %s: %s", length(failed), s$name, paste(head(failed, 20), collapse = "; ")))
  invisible(ch)
}

# ---- data and metrics -----------------------------------------------------------------

read_data <- function(name, stem)  # data/<name>_<stem>.csv (no header) as a matrix
  as.matrix(read.csv(file.path(.gallery_root, "data", paste0(name, "_", stem, ".csv")), header = FALSE))

rmse <- function(y, m) sqrt(mean((y - m)^2))
q2 <- function(y, m) 1 - sum((y - m)^2) / sum((y - mean(y))^2)
nlpd <- function(y, m, s) { s <- pmax(s, 1e-12); mean(0.5 * log(2 * pi * s^2) + 0.5 * ((y - m) / s)^2) }
coverage <- function(y, m, s, level = 0.95) mean(abs(y - m) <= qnorm(0.5 + level / 2) * s)
