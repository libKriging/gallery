% %% [markdown]
% # Octave · all classes — cross-binding consistency: the Octave side
%
% **Idea.** The Octave side of the cross-binding protocol: mLibKriging must give the
% numbers of pylibkriging on the same data, for every model class and fit method.
%
% **Reference.** The Python values (`cross-binding-reference_all-
% classes_python.ipynb`), compared by `tools/report.py`.
%
% Same protocol as `cross-binding-reference_all-classes_python.ipynb` (see there), on the
% shared datasets of `data/`; writes `results/xbind_octave.csv`, compared with
% the Python values by `tools/report.py`. mLibKriging must be on the path
% (`OCTAVE_PATH` or `addpath`).

% %%
source("tools/octave/gallery.m");
s = suite("cross-binding_all-classes_octave");
out = {};
function out = put(out, dataset, quantity, values)
  v = double(values(:));
  for i = 1:numel(v)
    out(end + 1, :) = {"octave", dataset, quantity, i, sprintf("%.17g", v(i))};
  end
end

% %%
for name = {"branin", "hartmann3", "borehole"}
  name = name{1};
  X = read_data(name, "X"); y = read_data(name, "y"); Xt = read_data(name, "Xt")(1:50, :);
  d = columns(X);
  theta0 = 0.3 * ones(1, d);
  kf = Kriging(y, X, "matern5_2", "constant", false, "none", "LL", Params("theta", theta0, "is_theta_estim", false));
  v = kf.sigma2(); out = put(out, name, "fixed_sigma2", v);
  v = kf.beta(); out = put(out, name, "fixed_beta", v);
  ll = kf.logLikelihood();
  out = put(out, name, "fixed_ll", ll);
  loo = kf.leaveOneOut();
  out = put(out, name, "fixed_loo", loo);
  [m, sd] = kf.predict(Xt, true, false, false);
  out = put(out, name, "fixed_mean", m);
  out = put(out, name, "fixed_stdev", sd);
  [llf, grad] = kf.logLikelihoodFun(theta0', true, false);
  out = put(out, name, "llfun", llf);
  out = put(out, name, "llfun_grad", grad);
  s = check_close(s, [name ": logLikelihood() == logLikelihoodFun(theta0)"], ll, llf, 1e-10);
  k = Kriging(y, X, "matern5_2");
  v = k.theta(); out = put(out, name, "fit_theta", v);
  llk = k.logLikelihood();
  out = put(out, name, "fit_ll", llk);
  [mk, ~] = k.predict(Xt, false, false, false);
  out = put(out, name, "fit_mean", mk);
  S = kf.simulate(int32(3), int32(123), Xt(1:5, :), false);
  out = put(out, name, "sim", S);
end

% %% [markdown]
% ## Other classes and fit methods (Branin)

% %%
name = "branin";
X = read_data(name, "X"); y = read_data(name, "y"); Xt = read_data(name, "Xt")(1:50, :);
theta0 = [0.3, 0.3];
s2 = mean((y - mean(y)).^2); nug = 0.01 * s2;   % population variance, as numpy.var
kn = Kriging(y, X, "matern5_2", "constant", false, "none", "LL", ...
             Params("theta", theta0, "is_theta_estim", false, "sigma2", s2, "is_sigma2_estim", false, ...
                    "nugget", nug, "is_nugget_estim", false), "nugget");
[m, sd] = kn.predict(Xt, true, false, false);
out = put(out, name, "nugget_fixed_mean", m); out = put(out, name, "nugget_fixed_stdev", sd);
kh = Kriging(y, X, "matern5_2", "constant", false, "none", "LL", ...
             Params("theta", theta0, "is_theta_estim", false, "sigma2", s2, "is_sigma2_estim", false), ...
             "heterogeneous", nug * ones(size(y)));
[m, sd] = kh.predict(Xt, true, false, false);
out = put(out, name, "noise_fixed_mean", m); out = put(out, name, "noise_fixed_stdev", sd);
kf = Kriging(y, X, "matern5_2", "constant", false, "BFGS", "LL", [], "nugget");
ll = kf.logLikelihood(); out = put(out, name, "nugget_fit_ll", ll);
m = kf.predict(Xt, false, false, false); out = put(out, name, "nugget_fit_mean", m);
for meth = {"vecchia", "nystrom"}
  meth = meth{1};
  try  % method is absent from older releases
    ka = Kriging(y, X, "matern5_2", "constant", false, "none", "LL", Params("theta", theta0, "is_theta_estim", false), ...
                 "none", [], [meth "(10)"]);
    ll = ka.logLikelihood(meth); out = put(out, name, [meth "_fixed_ll"], ll);
    m = ka.predict(Xt, false, false, false, meth); out = put(out, name, [meth "_fixed_mean"], m);
  catch err
    printf("%s skipped (reported as missing): %s\n", meth, err.message);
  end
end
kw = WarpKriging(y, X, {"kumaraswamy", "kumaraswamy"}, "matern5_2");
m = kw.predict(Xt, false, false, false); out = put(out, name, "warp_fit_mean", m);
km = MLPKriging(y, X, [8], 2, "tanh", "matern5_2", "constant", false, "BFGS10");
m = km.predict(Xt, false, false, false); out = put(out, name, "mlp_fit_mean", m);
kk = NestedKriging(y, X, "matern5_2", 2, "NK", "kmeans", 1);
m = kk.predict(Xt); out = put(out, name, "nested_fit_mean", m);

% %%
d = getenv("LKGALLERY_RESULTS"); if isempty(d), d = "results"; end
if ~exist(d, "dir"), mkdir(d); end
fid = fopen(fullfile(d, "xbind_octave.csv"), "w");
fprintf(fid, "lang,dataset,quantity,index,value\n");
for i = 1:size(out, 1)
  fprintf(fid, "%s,%s,%s,%d,%s\n", out{i, :});
end
fclose(fid);
printf("%d values written\n", size(out, 1));

% %% [markdown]
% ## The values written
%
% First value of each quantity and dataset (each compared with Python by the
% report; plots are left to the Python reference, Octave's graphics toolkit not
% being available on every runner).

% %%
keys = strcat(out(:, 2), " ", out(:, 3));
[u, first] = unique(keys, "first");
for i = 1:numel(u)
  printf("%-36s %s (%d values)\n", u{i}, out{first(i), 5}, sum(strcmp(keys, u{i})));
end

% %% [markdown]
% ## Summary
%
% The metrics of this notebook, the number of checks per status, and every check that
% did not pass (FAIL: error-level, fails the run; WARN: warn-level; KNOWN: listed in
% `tools/known_issues.csv`, with its issue; SKIP: feature absent from the installed version).
% All of them are also written to `results/cross-binding_all-classes_octave.checks.csv` and `.metrics.csv`.

% %%
suite_summary(s);

% %%
suite_finalize(s);
