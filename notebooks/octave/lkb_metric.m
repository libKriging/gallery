function s = lkb_metric(s, case_name, package, metric, value)
  s.metrics(end + 1, :) = {s.name, "octave", case_name, package, metric, sprintf("%.6g", double(value))};
end
