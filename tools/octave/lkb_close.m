function s = lkb_close(s, check, a, b, rtol, atol, detail, severity)
  if nargin < 5, rtol = 1e-6; end
  if nargin < 6, atol = 0; end
  if nargin < 7, detail = ""; end
  if nargin < 8, severity = "error"; end
  a = double(a(:)); b = double(b(:));
  if numel(a) ~= numel(b)
    s = lkb_check(s, check, false, sprintf("length %d", numel(a)), sprintf("length %d", numel(b)), detail, severity);
    return;
  end
  gap = abs(a - b);
  ok = all(isfinite(a)) && all(gap <= atol + rtol * abs(b));
  rel = max(gap ./ max(abs(b), 1e-300));
  s = lkb_check(s, check, ok, rel, sprintf("rtol %g, atol %g", rtol, atol), detail, severity);
end
