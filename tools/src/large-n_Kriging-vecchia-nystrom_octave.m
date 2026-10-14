% %% [markdown]
% # Octave · `Kriging` with an approximate fit — Vecchia, Nyström (n = 400)
%
% Branin on [0,1]² (n = 30 LHS points unless stated, Q² on 500 uniform test
% points), through mLibKriging. Checks (error-level unless stated):
%
% * the fit method is reported by `method()`
% * `predict` with the fit method: finite mean and stdev
% * Q² > 0.95 (warn-level)
%
% The same checks as `large-n_Kriging-vecchia-nystrom_R.ipynb`, with the Octave binding.

% %%
addpath(fullfile(pwd, "tools", "octave"));
s = lkb_suite("large-n_Kriging-vecchia-nystrom_octave");
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
% ## `Kriging` with an approximate fit: Vecchia, Nyström (n = 400)

% %%
Xl = lhs(400, 2); yl = branin(Xl);
for mth = {"vecchia(20)", "nystrom(60)"}
  mth = mth{1}; short = regexprep(mth, "\\(.*", "");
  label = ["Kriging(method = '" mth "')"];
  try
    ka = Kriging(yl, Xl, "matern5_2", "constant", false, "BFGS", "LL", [], "none", [], mth);
    got = ka.method();
    s = lkb_check(s, [label ": method()"], strcmp(got, mth), got, mth);
    [ma, sa] = ka.predict(Xt, true, false, false, short);
    s = lkb_check(s, [label ": finite predictions"], all(isfinite([ma; sa])), sum(~isfinite([ma; sa])), "0 non-finite");
    s = lkb_metric(s, "branin n=400", ["Kriging " mth], "q2", q2(yt, ma));
    s = lkb_check(s, [label ": Q2"], q2(yt, ma) > 0.95, q2(yt, ma), "> 0.95", "", "warn");
  catch err
    s = lkb_check(s, [label ": fit + predict"], false, "error", "no error", err.message);
  end
end

% %%
q = strcmp(s.metrics(:, 5), "q2");
disp(s.metrics(q, [3, 4, 6]))

% %%
lkb_finalize(s);
