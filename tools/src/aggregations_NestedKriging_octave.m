% %% [markdown]
% # Octave · `NestedKriging` — `NK` and `PoE` aggregations, 4 groups of 100 points
%
% Branin on [0,1]² (n = 30 LHS points unless stated, Q² on 500 uniform test
% points), through mLibKriging. Checks (error-level unless stated):
%
% * stdev ≥ 0, Q² > 0.95 (warn-level)
% * `NK` interpolates the data (it is a proper kriging predictor)
% * `save` then `load`: the same predicted mean
%
% The same checks as `aggregations_NestedKriging_R.ipynb`, with the Octave binding.
%
% A class missing from the installed mLibKriging (release channel) is recorded as SKIP.

% %%
addpath(fullfile(pwd, "tools", "octave"));
s = lkb_suite("aggregations_NestedKriging_octave");
q2 = @(y, m) 1 - sum((y - m).^2) / sum((y - mean(y)).^2);
branin = @(U) ((15 * U(:, 2)) - 5.1 / (4 * pi^2) * (15 * U(:, 1) - 5).^2 + 5 / pi * (15 * U(:, 1) - 5) - 6).^2 ...
              + 10 * (1 - 1 / (8 * pi)) * cos(15 * U(:, 1) - 5) + 10;
rand("seed", 1); randn("seed", 1);
function U = lhs(n, d)
  U = zeros(n, d);
  for j = 1:d, U(:, j) = (randperm(n)' - rand(n, 1)) / n; end
end
X = lhs(30, 2); y = branin(X);
Xt = rand(500, 2); yt = branin(Xt);
tmp = tempname(); mkdir(tmp);
has_class = @(name) exist(name) == 2;   % a class file on the path

function s = roundtrip(s, label, k, Xn, loader)
  f = fullfile(tempdir(), [regexprep(label, "[^A-Za-z0-9]", "_") ".json"]);
  try
    k.save(f);
    k2 = loader(f);
    m1 = k.predict(Xn, false, false, false);
    m2 = k2.predict(Xn, false, false, false);
    s = lkb_close(s, [label ": load(save(k)) predicts the same mean"], m2, m1, 1e-10);
  catch err
    s = lkb_check(s, [label ": save + load"], false, "error", "no error", err.message);
  end
end

function s = skip(s, check, reason)
  s.checks(end + 1, :) = {s.name, "octave", check, "info", "SKIP", "", "", reason};
  printf("[SKIP] %s: %s\n", check, reason);
end

% %% [markdown]
% ## `NestedKriging`: 4 groups of 100 points, `NK` and `PoE` aggregations

% %%
Xl = lhs(400, 2); yl = branin(Xl);
if has_class("NestedKriging")
  for agg = {"NK", "PoE"}
    agg = agg{1}; label = ["NestedKriging(" agg ")"];
    try
      kk = NestedKriging(yl, Xl, "matern5_2", 4, agg, "kmeans", 3);
      [mk, sk] = kk.predict(Xt);   % stdev returned when asked for (2 outputs)
      s = lkb_check(s, [label ": stdev >= 0"], all(sk >= 0), min(sk), ">= 0");
      s = lkb_metric(s, "branin n=400", ["NestedKriging " agg], "q2", q2(yt, mk));
      s = lkb_check(s, [label ": Q2"], q2(yt, mk) > 0.95, q2(yt, mk), "> 0.95", "", "warn");
      if strcmp(agg, "NK")
        e = max(abs(kk.predict(Xl) - yl)) / std(yl);
        s = lkb_check(s, "NestedKriging(NK): interpolates (max |m(X) - y| / sd(y))", e < 1e-3, e, "< 1e-3");
      end
      f = fullfile(tmp, ["nested_" agg ".json"]);
      kk.save(f); k2 = load_kriging(f);
      s = lkb_close(s, [label ": load(save(k)) predicts the same mean"], k2.predict(Xt), kk.predict(Xt), 1e-10);
    catch err
      s = lkb_check(s, [label ": fit + predict"], false, "error", "no error", err.message);
    end
  end
else
  s = skip(s, "NestedKriging", "not in this mLibKriging");
end

% %%
q = strcmp(s.metrics(:, 5), "q2");
disp(s.metrics(q, [3, 4, 6]))

% %%
lkb_finalize(s);
