function s = lkb_suite(name)
  % Same CSV outputs as lkbench/checks.py (see there), for the Octave notebooks.
  % A suite is a struct passed around by value: s = lkb_close(s, ...).
  s.name = name;
  s.checks = {};
  s.metrics = {};
  s.t0 = tic();
  s.known = {};
  f = fullfile("..", "..", "known_issues.csv");
  if exist(f, "file")
    fid = fopen(f, "r"); fgetl(fid);
    while true
      l = fgetl(fid);
      if ~ischar(l), break; end
      p = strsplit(l, ",");
      if numel(p) >= 3, s.known(end + 1, :) = {p{1}, p{2}, strjoin(p(3:end), ",")}; end
    end
    fclose(fid);
  end
end
