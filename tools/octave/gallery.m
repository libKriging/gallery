% Helpers of the libKriging gallery notebooks (Octave), in one script of functions:
% source("tools/octave/gallery.m") (the leading "1;" makes it a script, so that it can
% define several functions).
%
% The same functions, under the same names, as tools/python/gallery.py (where they are
% methods of Suite: suite.check_close(...) is s = check_close(s, ...) here; a suite is a
% struct passed by value), and the same CSV outputs (results/<suite>.checks.csv,
% .metrics.csv), aggregated with the other languages by tools/report.py.
%
%   s = suite(name)                                   a suite of checks, named after the notebook
%   s = check(s, check, ok, value, expected, detail, severity)
%   s = check_true / check_less / check_greater / check_between / check_close /
%       check_finite / check_skip (s, ...)            the usual checks
%   s = metric(s, case, package, name, value)         record a metric
%   suite_summary(s)                                  end-of-notebook summary
%   suite_finalize(s)                                 write the CSV files, error if an error-level check failed
%   read_data(name, stem), rmse, q2
%
% Paths are relative to the repository root ($LKGALLERY_ROOT, else the working
% directory, where the notebooks are executed).
1;

function r = gallery_root()
  r = getenv("LKGALLERY_ROOT"); if isempty(r), r = "."; end
end

% ---- suite and checks ------------------------------------------------------------------

function s = suite(name)
  s.name = name;
  s.checks = cell(0, 8);
  s.metrics = cell(0, 6);
  s.t0 = tic();
  s.known = cell(0, 3);
  f = fullfile(gallery_root(), "tools", "known_issues.csv");
  if ~exist(f, "file"), error("run the notebooks from the repository root (%s not found)", f); end
  fid = fopen(f, "r"); fgetl(fid);
  while true
    l = fgetl(fid);
    if ~ischar(l), break; end
    p = gallery_csv_fields(l);
    if numel(p) >= 3, s.known(end + 1, :) = p(1:3); end
  end
  fclose(fid);
end

function s = check(s, name, ok, value, expected, detail, severity)
  if nargin < 4, value = ""; end
  if nargin < 5, expected = ""; end
  if nargin < 6, detail = ""; end
  if nargin < 7, severity = "error"; end
  if ok, status = "PASS"; elseif strcmp(severity, "error"), status = "FAIL"; else, status = "WARN"; end
  if ~ok && ~isempty(s.known)
    hit = strcmp(s.known(:, 1), s.name) & strcmp(s.known(:, 2), name);
    if any(hit), status = "KNOWN"; detail = [s.known{find(hit, 1), 3} " | " detail]; end
  end
  if isnumeric(value) || islogical(value), value = sprintf("%.6g", double(value)); end
  if ~ok, printf("[%s] %s: value=%s expected %s %s\n", status, name, value, expected, detail); end
  s.checks(end + 1, :) = {s.name, "octave", name, severity, status, value, expected, detail};
end

function s = check_true(s, name, cond, detail, severity)
  if nargin < 4, detail = ""; end
  if nargin < 5, severity = "error"; end
  s = check(s, name, logical(cond), logical(cond), "1", detail, severity);
end

function s = check_less(s, name, value, bound, detail, severity)
  if nargin < 5, detail = ""; end
  if nargin < 6, severity = "error"; end
  s = check(s, name, isfinite(value) && value < bound, value, sprintf("< %g", bound), detail, severity);
end

function s = check_greater(s, name, value, bound, detail, severity)
  if nargin < 5, detail = ""; end
  if nargin < 6, severity = "error"; end
  s = check(s, name, isfinite(value) && value > bound, value, sprintf("> %g", bound), detail, severity);
end

function s = check_between(s, name, value, lo, hi, detail, severity)
  if nargin < 6, detail = ""; end
  if nargin < 7, severity = "error"; end
  s = check(s, name, isfinite(value) && value >= lo && value <= hi, value, sprintf("in [%g, %g]", lo, hi), detail, severity);
end

function s = check_close(s, name, a, b, rtol, atol, detail, severity)
  if nargin < 5, rtol = 1e-6; end
  if nargin < 6, atol = 0; end
  if nargin < 7, detail = ""; end
  if nargin < 8, severity = "error"; end
  a = double(a(:)); b = double(b(:));
  if numel(a) ~= numel(b)
    s = check(s, name, false, sprintf("length %d", numel(a)), sprintf("length %d", numel(b)), detail, severity);
    return;
  end
  gap = abs(a - b);
  ok = all(isfinite(a)) && all(gap <= atol + rtol * abs(b));
  rel = max(gap ./ max(abs(b), 1e-300));
  s = check(s, name, ok, rel, sprintf("rtol %g, atol %g", rtol, atol), detail, severity);
end

function s = check_finite(s, name, a, detail, severity)
  if nargin < 4, detail = ""; end
  if nargin < 5, severity = "error"; end
  s = check(s, name, ~isempty(a) && all(isfinite(a(:))), sprintf("%d non-finite", sum(~isfinite(a(:)))), "all finite", detail, severity);
end

function s = check_skip(s, name, reason)
  s.checks(end + 1, :) = {s.name, "octave", name, "info", "SKIP", "", "", reason};
  printf("[SKIP] %s: %s\n", name, reason);
end

% ---- metrics and results ---------------------------------------------------------------

function s = metric(s, case_name, package, name, value)
  s.metrics(end + 1, :) = {s.name, "octave", case_name, package, name, sprintf("%.6g", double(value))};
end

function suite_summary(s)
  for i = 1:size(s.metrics, 1)
    printf("%-28s %-28s %-12s %s\n", s.metrics{i, 3}, s.metrics{i, 4}, s.metrics{i, 5}, s.metrics{i, 6});
  end
  st = s.checks(:, 5);
  printf("%d checks: PASS %d, FAIL %d, WARN %d, KNOWN %d, SKIP %d\n", numel(st), sum(strcmp(st, "PASS")), ...
         sum(strcmp(st, "FAIL")), sum(strcmp(st, "WARN")), sum(strcmp(st, "KNOWN")), sum(strcmp(st, "SKIP")));
  for i = find(~strcmp(st, "PASS"))'
    printf("%-6s %-6s %s: value %s, expected %s %s\n", s.checks{i, 5}, s.checks{i, 4}, s.checks{i, 3}, ...
           s.checks{i, 6}, s.checks{i, 7}, s.checks{i, 8});
  end
end

function suite_finalize(s)
  s = metric(s, "_suite", "_", "elapsed_s", toc(s.t0));
  d = getenv("LKGALLERY_RESULTS"); if isempty(d), d = fullfile(gallery_root(), "results"); end
  if ~exist(d, "dir"), mkdir(d); end
  gallery_write_csv(fullfile(d, [s.name ".checks.csv"]), {"suite", "lang", "check", "severity", "status", "value", "expected", "detail"}, s.checks);
  gallery_write_csv(fullfile(d, [s.name ".metrics.csv"]), {"suite", "lang", "case", "package", "metric", "value"}, s.metrics);
  st = s.checks(:, 5);
  printf("%s: PASS %d, FAIL %d, WARN %d, KNOWN %d, SKIP %d\n", s.name, sum(strcmp(st, "PASS")), sum(strcmp(st, "FAIL")), ...
         sum(strcmp(st, "WARN")), sum(strcmp(st, "KNOWN")), sum(strcmp(st, "SKIP")));
  failed = s.checks(strcmp(st, "FAIL"), 3);
  if ~isempty(failed)
    error("%d check(s) failed in %s: %s", numel(failed), s.name, strjoin(failed', "; "));
  end
end

% ---- data and metrics ------------------------------------------------------------------

function A = read_data(name, stem)  % data/<name>_<stem>.csv (no header)
  A = dlmread(fullfile(gallery_root(), "data", [name "_" stem ".csv"]), ",");
end

function r = rmse(y, m)
  r = sqrt(mean((y(:) - m(:)).^2));
end

function r = q2(y, m)
  r = 1 - sum((y(:) - m(:)).^2) / sum((y(:) - mean(y(:))).^2);
end

% ---- internals -------------------------------------------------------------------------

function gallery_write_csv(f, header, rows)
  fid = fopen(f, "w");
  fprintf(fid, "%s\n", strjoin(header, ","));
  for i = 1:size(rows, 1)
    q = cellfun(@(c) ["\"" strrep(char(c), "\"", "\"\"") "\""], rows(i, :), "UniformOutput", false);
    fprintf(fid, "%s\n", strjoin(q, ","));
  end
  fclose(fid);
end

function f = gallery_csv_fields(l)
  % one CSV line into fields (RFC 4180 quoting: "a,b" and "" inside quotes)
  f = {}; cur = ""; q = false; i = 1;
  while i <= numel(l)
    c = l(i);
    if q && c == '"' && i < numel(l) && l(i + 1) == '"'
      cur(end + 1) = '"'; i = i + 1;
    elseif c == '"'
      q = ~q;
    elseif c == "," && ~q
      f{end + 1} = cur; cur = "";
    else
      cur(end + 1) = c;
    end
    i = i + 1;
  end
  f{end + 1} = cur;
end
