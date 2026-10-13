function lkb_finalize(s)
  s = lkb_metric(s, "_suite", "_", "elapsed_s", toc(s.t0));
  d = getenv("LKBENCH_RESULTS"); if isempty(d), d = fullfile("..", "..", "results"); end
  if ~exist(d, "dir"), mkdir(d); end
  write_csv(fullfile(d, [s.name ".checks.csv"]), {"suite", "lang", "check", "severity", "status", "value", "expected", "detail"}, s.checks);
  write_csv(fullfile(d, [s.name ".metrics.csv"]), {"suite", "lang", "case", "package", "metric", "value"}, s.metrics);
  st = s.checks(:, 5);
  printf("%s: PASS %d, FAIL %d, WARN %d, KNOWN %d, SKIP %d\n", s.name, sum(strcmp(st, "PASS")), sum(strcmp(st, "FAIL")), ...
         sum(strcmp(st, "WARN")), sum(strcmp(st, "KNOWN")), sum(strcmp(st, "SKIP")));
  failed = s.checks(strcmp(st, "FAIL"), 3);
  if ~isempty(failed)
    error("%d check(s) failed in %s: %s", numel(failed), s.name, strjoin(failed', "; "));
  end
end

function write_csv(f, header, rows)
  fid = fopen(f, "w");
  fprintf(fid, "%s\n", strjoin(header, ","));
  for i = 1:size(rows, 1)
    q = cellfun(@(c) ["\"" strrep(char(c), "\"", "\"\"") "\""], rows(i, :), "UniformOutput", false);
    fprintf(fid, "%s\n", strjoin(q, ","));
  end
  fclose(fid);
end
