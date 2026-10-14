% %% [markdown]
% # Octave · all classes — every libKriging model class and fit method through mLibKriging
%
% Same tour as `R_all-classes_api-tour.ipynb` (see there for the checks): Branin
% on [0,1]², every class and fit method of the Octave binding (positional
% arguments), each with the checks that hold for it whatever the data. A class
% missing from the installed mLibKriging is recorded as SKIP. mLibKriging must
% be on the path (`OCTAVE_PATH` or `addpath`), see `tools/install/octave.sh`.

% %%
addpath(fullfile(pwd, "tools", "octave"));
s = lkb_suite("octave_all-classes_api-tour");
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

% %% [markdown]
% ## `NestedKriging`: 4 groups of 100 points, `NK` and `PoE` aggregations

% %%
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
