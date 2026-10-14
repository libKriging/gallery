function s = lkb_check(s, check, ok, value, expected, detail, severity)
  if nargin < 4, value = ""; end
  if nargin < 5, expected = ""; end
  if nargin < 6, detail = ""; end
  if nargin < 7, severity = "error"; end
  if ok, status = "PASS"; elseif strcmp(severity, "error"), status = "FAIL"; else, status = "WARN"; end
  if ~ok && ~isempty(s.known)
    hit = strcmp(s.known(:, 1), s.name) & strcmp(s.known(:, 2), check);
    if any(hit), status = "KNOWN"; detail = [s.known{find(hit, 1), 3} " | " detail]; end
  end
  if isnumeric(value) || islogical(value), value = sprintf("%.6g", double(value)); end
  if ~ok, printf("[%s] %s: value=%s expected %s %s\n", status, check, value, expected, detail); end
  s.checks(end + 1, :) = {s.name, "octave", check, severity, status, value, expected, detail};
end
