% %% [markdown]
% # Octave · `WarpKriging` — continuous warpings, a categorical input
%
% Branin on [0,1]² (n = 30 LHS points unless stated, Q² on 500 uniform test
% points), through mLibKriging. Checks (error-level unless stated):
%
% * `kumaraswamy` on both inputs, then x₂ restricted to 5 levels with `categorical(5,2)`: finite predictions
% * Q² floors (warn-level)
% * `save` then `load`: the same predicted mean
%
% The same checks as `warpings-categorical_WarpKriging_R.ipynb`, with the Octave binding.
%
% A class missing from the installed mLibKriging (release channel) is recorded as SKIP.

% %%
addpath(fullfile(pwd, "tools", "octave"));
s = lkb_suite("warpings-categorical_WarpKriging_octave");
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
% ## `WarpKriging`: continuous warpings, then a categorical input
%
% The categorical case: x₂ restricted to 5 levels (coded 0, …, 4), embedded in ℝ².

% %%
if has_class("WarpKriging")
  try
    kw = WarpKriging(y, X, {"kumaraswamy", "kumaraswamy"}, "matern5_2");
    [mw, sw] = kw.predict(Xt, true, false, false);
    s = lkb_check(s, "WarpKriging(kumaraswamy): finite predictions", all(isfinite([mw; sw])), sum(~isfinite([mw; sw])), "0 non-finite");
    s = lkb_metric(s, "branin", "WarpKriging kumaraswamy", "q2", q2(yt, mw));
    s = lkb_check(s, "WarpKriging(kumaraswamy): Q2", q2(yt, mw) > 0.9, q2(yt, mw), "> 0.9", "", "warn");
    s = roundtrip(s, "WarpKriging(kumaraswamy)", kw, Xt, @load_kriging);
  catch err
    s = lkb_check(s, "WarpKriging(kumaraswamy): fit + predict", false, "error", "no error", err.message);
  end
  lev = @(U) [U(:, 1), round(U(:, 2) * 4)];
  unlev = @(L) [L(:, 1), L(:, 2) / 4];
  Xc = lev(lhs(40, 2)); yc = branin(unlev(Xc));
  Xtc = lev(Xt); ytc = branin(unlev(Xtc));
  try
    kc = WarpKriging(yc, Xc, {"none", "categorical(5,2)"}, "matern5_2");
    [mc, sc] = kc.predict(Xtc, true, false, false);
    s = lkb_check(s, "WarpKriging(categorical): finite predictions", all(isfinite([mc; sc])), sum(~isfinite([mc; sc])), "0 non-finite");
    s = lkb_metric(s, "branin, x2 categorical", "WarpKriging categorical", "q2", q2(ytc, mc));
    s = lkb_check(s, "WarpKriging(categorical): Q2", q2(ytc, mc) > 0.8, q2(ytc, mc), "> 0.8", "", "warn");
    s = roundtrip(s, "WarpKriging(categorical)", kc, Xtc, @load_kriging);
  catch err
    s = lkb_check(s, "WarpKriging(categorical): fit + predict", false, "error", "no error", err.message);
  end
else
  s = skip(s, "WarpKriging", "not in this mLibKriging");
end

% %%
q = strcmp(s.metrics(:, 5), "q2");
disp(s.metrics(q, [3, 4, 6]))

% %%
lkb_finalize(s);
