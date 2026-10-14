% %% [markdown]
% # Octave · `Kriging`, noise-free — interpolation, conditional simulation, update, save/load
%
% Branin on [0,1]² (n = 30 LHS points unless stated, Q² on 500 uniform test
% points), through mLibKriging. Checks (error-level unless stated):
%
% * interpolation (max |m(X) − y| / sd(y) < 1e-6) and Q² > 0.95 on 500 test points
% * conditional simulations through the data (relative to σ)
% * stdev > 0 off the design
% * `update` without refit = a fit on all the points at the same θ, σ²
% * `save` then `load`: the same predicted mean
%
% The same checks as `interpolation-simulate-update-save_Kriging_R.ipynb`, with the Octave binding.

% %%
addpath(fullfile(pwd, "tools", "octave"));
s = lkb_suite("interpolation-simulate-update-save_Kriging_octave");
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
% ## `Kriging`, noise-free

% %%
k = Kriging(y, X, "matern5_2");
[m, sd] = k.predict(Xt, true, false, false);
s = lkb_metric(s, "branin", "Kriging", "q2", q2(yt, m));
s = lkb_check(s, "Kriging: Q2", q2(yt, m) > 0.95, q2(yt, m), "> 0.95");
mX = k.predict(X, false, false, false);
e = max(abs(mX - y)) / std(y);
s = lkb_check(s, "Kriging: interpolates (max |m(X) - y| / sd(y))", e < 1e-6, e, "< 1e-6");
S = k.simulate(int32(5), int32(2), X, false);
e = max(max(abs(S - y))) / sqrt(k.sigma2());
s = lkb_check(s, "Kriging: conditional simulations go through the data (max |sim(X) - y| / sigma)", e < 1e-4, e, "< 1e-4");
s = lkb_check(s, "Kriging: stdev > 0 off the design", min(sd) > 0, min(sd), "> 0");
Xu = lhs(5, 2); yu = branin(Xu);
ku = k.copy();
ku.update(yu, Xu, false);
th = k.theta();
k35 = Kriging([y; yu], [X; Xu], "matern5_2", "constant", false, "none", "LL", ...
              Params("theta", th(:)', "is_theta_estim", false, "sigma2", k.sigma2(), "is_sigma2_estim", false));
s = lkb_close(s, "Kriging: update(refit = false) == fit on all points (same theta, sigma2)", ...
              ku.predict(Xt, false, false, false), k35.predict(Xt, false, false, false), 1e-6, 1e-8 * std(y));
s = roundtrip(s, "Kriging", k, Xt, @load_kriging);

% %%
q = strcmp(s.metrics(:, 5), "q2");
disp(s.metrics(q, [3, 4, 6]))

% %%
lkb_finalize(s);
