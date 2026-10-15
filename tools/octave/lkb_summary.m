function lkb_summary(s)
  % End-of-notebook summary: metrics, count per status, every check that did not pass.
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
