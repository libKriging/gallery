function s = lkb_suite(name)
  % Same CSV outputs as lkbench/checks.py (see there), for the Octave notebooks.
  % A suite is a struct passed around by value: s = lkb_close(s, ...).
  % Paths are relative to the repository root, where the notebooks are executed.
  s.name = name;
  s.checks = {};
  s.metrics = {};
  s.t0 = tic();
  s.known = cell(0, 3);
  f = fullfile("tools", "known_issues.csv");
  if ~exist(f, "file"), error("run the notebooks from the repository root (%s not found)", f); end
  fid = fopen(f, "r"); fgetl(fid);
  while true
    l = fgetl(fid);
    if ~ischar(l), break; end
    p = csv_fields(l);
    if numel(p) >= 3, s.known(end + 1, :) = p(1:3); end
  end
  fclose(fid);
end

function f = csv_fields(l)
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
