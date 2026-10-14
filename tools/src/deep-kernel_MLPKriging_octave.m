% %% [markdown]
% # Octave · `MLPKriging` — a joint feature map ℝ² → ℝ² before the kernel
%
% Branin on [0,1]² (n = 30 LHS points unless stated, Q² on 500 uniform test
% points), through mLibKriging. Checks (error-level unless stated):
%
% * 10 restarts (`BFGS10`): finite predictions, Q² > 0.9 (warn-level)
% * `save` then `load`: the same predicted mean
% * the default optimizer as good as `BFGS10` (warn-level; see libKriging/libKriging#382)
%
% The same checks as `deep-kernel_MLPKriging_R.ipynb`, with the Octave binding.
%
% A class missing from the installed mLibKriging (release channel) is recorded as SKIP.

% %%
addpath(fullfile(pwd, "tools", "octave"));
s = lkb_suite("deep-kernel_MLPKriging_octave");
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
% ## `MLPKriging`: a joint feature map ℝ² → ℝ² before the kernel

% %%
if has_class("MLPKriging")
  try
    km = MLPKriging(y, X, [8], 2, "tanh", "matern5_2", "constant", false, "BFGS10");
    [mm, sm] = km.predict(Xt, true, false, false);
    s = lkb_check(s, "MLPKriging: finite predictions", all(isfinite([mm; sm])), sum(~isfinite([mm; sm])), "0 non-finite");
    s = lkb_metric(s, "branin", "MLPKriging tanh [8] -> 2, BFGS10", "q2", q2(yt, mm));
    s = lkb_check(s, "MLPKriging: Q2", q2(yt, mm) > 0.9, q2(yt, mm), "> 0.9", "", "warn");
    s = roundtrip(s, "MLPKriging", km, Xt, @load_kriging);
    kd = MLPKriging(y, X, [8], 2, "tanh", "matern5_2");   % default optimizer
    qd = q2(yt, kd.predict(Xt, false, false, false));
    s = lkb_metric(s, "branin", "MLPKriging tanh [8] -> 2, default optim", "q2", qd);
    s = lkb_check(s, "MLPKriging: Q2(default optim) - Q2(BFGS10)", qd - q2(yt, mm) > -0.05, qd - q2(yt, mm), "> -0.05", "", "warn");
  catch err
    s = lkb_check(s, "MLPKriging: fit + predict", false, "error", "no error", err.message);
  end
else
  s = skip(s, "MLPKriging", "not in this mLibKriging");
end

% %%
q = strcmp(s.metrics(:, 5), "q2");
disp(s.metrics(q, [3, 4, 6]))

% %%
lkb_finalize(s);
