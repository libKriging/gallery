# Same CSV outputs as lkgallery/checks.py (see there), for the R notebooks.
# Statuses: PASS, FAIL (error-level), WARN (warn-level), SKIP, KNOWN (listed in tools/known_issues.csv).
# Paths are relative to the repository root, where the notebooks are executed.
results_dir <- Sys.getenv("LKGALLERY_RESULTS", "results")
dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)
known_file <- file.path("tools", "known_issues.csv")
if (!file.exists(known_file)) stop("run the notebooks from the repository root (", known_file, " not found)")
.known <- read.csv(known_file, stringsAsFactors = FALSE)

new_suite <- function(name) {
  s <- new.env()
  s$name <- name
  s$checks <- list()
  s$metrics <- list()
  s$t0 <- Sys.time()
  s
}

.fmt <- function(v) if (is.null(v) || length(v) == 0) "" else if (is.numeric(v)) format(signif(v[1], 6)) else as.character(v[1])

add_check <- function(s, check, ok, value = NULL, expected = NULL, detail = "", severity = "error") {
  ok <- isTRUE(ok)
  status <- if (ok) "PASS" else if (severity == "error") "FAIL" else "WARN"
  k <- .known[.known$suite == s$name & .known$check == check, ]
  if (!ok && nrow(k) > 0) { status <- "KNOWN"; detail <- paste(k$issue[1], detail, sep = " | ") }
  if (!ok) message(sprintf("[%s] %s: value=%s expected %s %s", status, check, .fmt(value), .fmt(expected), detail))
  s$checks[[length(s$checks) + 1]] <- data.frame(suite = s$name, lang = "R", check = check, severity = severity,
    status = status, value = .fmt(value), expected = .fmt(expected), detail = detail, stringsAsFactors = FALSE)
  invisible(ok)
}

check_close <- function(s, check, a, b, rtol = 1e-6, atol = 0, detail = "", severity = "error") {
  a <- as.numeric(a); b <- as.numeric(b)
  if (length(a) != length(b)) return(add_check(s, check, FALSE, paste("length", length(a)), paste("length", length(b)), detail, severity))
  gap <- abs(a - b)
  ok <- all(is.finite(a)) && all(gap <= atol + rtol * abs(b))
  rel <- max(gap / pmax(abs(b), 1e-300))
  add_check(s, check, ok, rel, sprintf("rtol %g, atol %g", rtol, atol), detail, severity)
}

check_less <- function(s, check, value, bound, detail = "", severity = "error")
  add_check(s, check, is.finite(value) && value < bound, value, paste("<", bound), detail, severity)

check_greater <- function(s, check, value, bound, detail = "", severity = "error")
  add_check(s, check, is.finite(value) && value > bound, value, paste(">", bound), detail, severity)

check_runs <- function(s, check, expr, severity = "error") {
  r <- tryCatch({ force(expr); NULL }, error = function(e) conditionMessage(e))
  add_check(s, check, is.null(r), if (is.null(r)) "ok" else "error", "no error", if (is.null(r)) "" else substr(r, 1, 200), severity)
}

add_metric <- function(s, case, package, ...) {
  v <- list(...)
  for (n in names(v)) s$metrics[[length(s$metrics) + 1]] <- data.frame(suite = s$name, lang = "R", case = case,
    package = package, metric = n, value = .fmt(v[[n]]), stringsAsFactors = FALSE)
}

summary_suite <- function(s) {
  # end-of-notebook summary: metrics, count per status, every check that did not pass
  if (length(s$metrics) > 0) {
    me <- do.call(rbind, s$metrics)
    me$value <- as.numeric(me$value)
    print(reshape(me[, c("case", "package", "metric", "value")], idvar = c("case", "package"),
                  timevar = "metric", direction = "wide"), row.names = FALSE)
  }
  ch <- do.call(rbind, s$checks)
  print(table(factor(ch$status, levels = c("PASS", "FAIL", "WARN", "KNOWN", "SKIP"))))
  other <- ch[ch$status != "PASS", c("status", "severity", "check", "value", "expected", "detail")]
  if (nrow(other) > 0) print(other, row.names = FALSE)
  invisible(ch)
}

finalize <- function(s) {
  add_metric(s, "_suite", "_", elapsed_s = as.numeric(difftime(Sys.time(), s$t0, units = "secs")))
  ch <- do.call(rbind, s$checks)
  me <- do.call(rbind, s$metrics)
  write.csv(ch, file.path(results_dir, paste0(s$name, ".checks.csv")), row.names = FALSE)
  write.csv(me, file.path(results_dir, paste0(s$name, ".metrics.csv")), row.names = FALSE)
  print(table(factor(ch$status, levels = c("PASS", "FAIL", "WARN", "KNOWN", "SKIP"))))
  failed <- ch$check[ch$status == "FAIL"]
  if (length(failed) > 0) stop(sprintf("%d check(s) failed in %s: %s", length(failed), s$name, paste(head(failed, 20), collapse = "; ")))
  invisible(ch)
}

rmse <- function(y, m) sqrt(mean((y - m)^2))
q2 <- function(y, m) 1 - sum((y - m)^2) / sum((y - mean(y))^2)
read_data <- function(name, stem) as.matrix(read.csv(file.path("data", paste0(name, "_", stem, ".csv")), header = FALSE))
