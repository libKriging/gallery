% %% [markdown]
% # Octave · `Kriging` with noise — estimated nugget, known variances
%
% Branin on [0,1]² (n = 30 LHS points unless stated, Q² on 500 uniform test
% points), through mLibKriging. Checks (error-level unless stated):
%
% * `noise = "nugget"` (10 restarts): nugget > 0, Q² > 0.95 on the noise-free function, smoothing next to the design points
% * the default single start reaches the multistart likelihood (warn-level)
% * `save` then `load`: the same predicted mean
% * `noise = <variances>`: the stored variances, finite predictions
%
% The same checks as `noisy-data_Kriging-nugget-noise_R.ipynb`, with the Octave binding.

% %%
addpath(fullfile(pwd, "tools", "octave"));
s = lkb_suite("noisy-data_Kriging-nugget-noise_octave");
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
% ## `Kriging` with noise: estimated nugget, known variances
%
% Noise of standard deviation 5% of sd(y). The nugget model is fitted with 10
% restarts (`"BFGS10"`); the default single start is compared with it
% (warn-level: a lower likelihood means a local optimum). As DiceKriging's
% nugget, the predictor returns y exactly *at* the design points (the nugget is
% a micro-scale variation): smoothing is checked next to them.

% %%
sd_noise = 0.05 * std(y);
yn = y + sd_noise * randn(size(y));
kn = Kriging(yn, X, "matern5_2", "constant", false, "BFGS10", "LL", [], "nugget");
mn = kn.predict(Xt, false, false, false);
nug = kn.nugget(); ll10 = kn.logLikelihood();
s = lkb_metric(s, "branin noisy", "Kriging nugget BFGS10", "q2", q2(yt, mn));
s = lkb_check(s, "Kriging(nugget): estimated nugget > 0", nug > 0, nug, "> 0");
s = lkb_check(s, "Kriging(nugget): Q2", q2(yt, mn) > 0.95, q2(yt, mn), "> 0.95");
r = sqrt(mean((kn.predict(min(X + 1e-7, 1), false, false, false) - yn).^2)) / sd_noise;
s = lkb_check(s, "Kriging(nugget): smooths the noisy data next to the design points (rms residual / noise sd)", r > 0.3, r, "> 0.3");
kn1 = Kriging(yn, X, "matern5_2", "constant", false, "BFGS", "LL", [], "nugget");
ll1 = kn1.logLikelihood();
s = lkb_metric(s, "branin noisy", "Kriging nugget BFGS", "q2", q2(yt, kn1.predict(Xt, false, false, false)));
s = lkb_check(s, "Kriging(nugget): LL(single start) - LL(10 restarts)", ll1 - ll10 > -1, ll1 - ll10, "> -1", "", "warn");
s = roundtrip(s, "Kriging(nugget)", kn, Xt, @load_kriging);
v = sd_noise^2 * ones(size(yn));
kh = Kriging(yn, X, "matern5_2", "constant", false, "BFGS", "LL", [], "heterogeneous", v);
s = lkb_close(s, "Kriging(noise = variances): noise() returns them", kh.noise(), v, 1e-12);
[mh, sh] = kh.predict(Xt, true, false, false);
s = lkb_check(s, "Kriging(noise = variances): finite predictions", all(isfinite([mh; sh])), sum(~isfinite([mh; sh])), "0 non-finite");
s = lkb_metric(s, "branin noisy", "Kriging known noise", "q2", q2(yt, mh));

% %%
q = strcmp(s.metrics(:, 5), "q2");
disp(s.metrics(q, [3, 4, 6]))

% %%
lkb_finalize(s);
